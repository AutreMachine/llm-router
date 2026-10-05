"""Machine/model registry and priority queue.

All code runs in the single asyncio loop: no critical section
contains an ``await``, so no lock is needed.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import Config, MachineConfig, ModelConfig

log = logging.getLogger("llm_router.scheduler")

PRIORITIES = {"high": 0, "medium": 1, "low": 2}
PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}


class ModelNotFound(Exception):
    pass


class NoHealthyBackend(Exception):
    pass


class QueueFull(Exception):
    pass


class QueueTimeout(Exception):
    pass


def parse_priority(value, default: str = "medium") -> int:
    if value is None or value == "":
        return PRIORITIES[default]
    v = str(value).strip().lower()
    if v not in PRIORITIES:
        raise ValueError(f"Invalid priority '{value}' (expected: low, medium, high)")
    return PRIORITIES[v]


@dataclass
class Machine:
    cfg: MachineConfig
    active: int = 0
    healthy: bool = True
    last_error: Optional[str] = None
    last_check: float = 0.0
    # Observed exponential moving averages (tokens/s)
    ema_prefill: Optional[float] = None
    ema_gen: Optional[float] = None

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def available(self) -> bool:
        return self.cfg.enabled and self.healthy

    def update_perf(self, prefill: Optional[float], gen: Optional[float], alpha: float = 0.2) -> None:
        if prefill:
            self.ema_prefill = prefill if self.ema_prefill is None else alpha * prefill + (1 - alpha) * self.ema_prefill
        if gen:
            self.ema_gen = gen if self.ema_gen is None else alpha * gen + (1 - alpha) * self.ema_gen


@dataclass
class Deployment:
    """A model hosted on a given machine."""
    machine: Machine
    model: ModelConfig
    active: int = 0

    def has_free_slot(self) -> bool:
        m = self.machine
        if not m.available or m.active >= m.cfg.max_concurrent:
            return False
        if self.model.max_concurrent is not None and self.active >= self.model.max_concurrent:
            return False
        return True


@dataclass
class Lease:
    deployment: Deployment
    released: bool = False

    @property
    def machine(self) -> Machine:
        return self.deployment.machine


_seq = itertools.count()


@dataclass
class Waiter:
    model: str
    priority: int
    exclude: frozenset[str]
    tag: Optional[str] = None
    enqueued_at: float = field(default_factory=time.monotonic)
    seq: int = field(default_factory=lambda: next(_seq))
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class Scheduler:
    def __init__(self, config: Config):
        self.server = config.server
        self.machines: dict[str, Machine] = {}
        self.deployments: dict[str, list[Deployment]] = {}
        self.queue: list[Waiter] = []
        for mc in config.machines:
            self.machines[mc.name] = Machine(mc)
        self._rebuild()

    # ------------------------------------------------- dynamic pool management
    def _rebuild(self) -> None:
        """Rebuilds the model -> deployments index, preserving active counters."""
        old = {(d.machine.name, d.model.name): d for deps in self.deployments.values() for d in deps}
        new: dict[str, list[Deployment]] = {}
        for machine in self.machines.values():
            for model in machine.cfg.models:
                dep = old.get((machine.name, model.name))
                if dep is not None and dep.machine is machine:
                    dep.model = model
                else:
                    dep = Deployment(machine, model)
                new.setdefault(model.name, []).append(dep)
        self.deployments = new
        self._fail_orphans()
        self.dispatch()

    def _validate(self, machines: list[MachineConfig]) -> None:
        from .config import check_model_types
        check_model_types(machines)

    def add_machine(self, mc: MachineConfig) -> Machine:
        if mc.name in self.machines:
            raise ValueError(f"Machine '{mc.name}' already exists")
        self._validate([m.cfg for m in self.machines.values()] + [mc])
        machine = Machine(mc, healthy=True)
        self.machines[mc.name] = machine
        self._rebuild()
        log.info("Machine added: %s (%s)", mc.name, mc.url)
        return machine

    def update_machine(self, name: str, mc: MachineConfig) -> Machine:
        machine = self.machines.get(name)
        if machine is None:
            raise KeyError(name)
        if mc.name != name:
            raise ValueError("A machine's name cannot be changed")
        self._validate([m.cfg for m in self.machines.values() if m.name != name] + [mc])
        url_changed = mc.url != machine.cfg.url
        machine.cfg = mc
        if url_changed:
            machine.ema_prefill = machine.ema_gen = None
        self._rebuild()
        log.info("Machine updated: %s", name)
        return machine

    def remove_machine(self, name: str) -> Machine:
        """In-flight requests on the machine will run to completion."""
        machine = self.machines.pop(name, None)
        if machine is None:
            raise KeyError(name)
        self._rebuild()
        log.info("Machine removed: %s (%d in-flight request(s) left to finish)", name, machine.active)
        return machine

    def set_enabled(self, name: str, enabled: bool) -> Machine:
        machine = self.machines[name]
        machine.cfg.enabled = enabled
        if enabled:
            self.dispatch()
        else:
            self._fail_orphans()
        return machine

    # ------------------------------------------------------------------ infos
    def model_type(self, model: str) -> Optional[str]:
        deps = self.deployments.get(model)
        return deps[0].model.type if deps else None

    def effective_priority(self, w: Waiter, now: float) -> int:
        if self.server.aging_seconds > 0:
            boost = math.floor((now - w.enqueued_at) / self.server.aging_seconds)
            return max(0, w.priority - boost)
        return w.priority

    # -------------------------------------------------------------- sélection
    def _pick(self, model: str, exclude: frozenset[str]) -> Optional[Deployment]:
        """The most performant free machine (tie-break: least loaded)."""
        candidates = [d for d in self.deployments.get(model, [])
                      if d.machine.name not in exclude and d.has_free_slot()]
        if not candidates:
            return None
        return max(candidates, key=lambda d: (
            d.machine.cfg.performance,
            -d.machine.active / d.machine.cfg.max_concurrent,
        ))

    def _lease(self, dep: Deployment) -> Lease:
        dep.active += 1
        dep.machine.active += 1
        return Lease(dep)

    def dispatch(self) -> None:
        """Assigns free machines to waiting requests, by priority."""
        if not self.queue:
            return
        now = time.monotonic()
        self.queue = [w for w in self.queue if not w.future.done()]
        ordered = sorted(self.queue, key=lambda w: (self.effective_priority(w, now), w.seq))
        served = set()
        for w in ordered:
            dep = self._pick(w.model, w.exclude)
            if dep is None:
                continue  # nothing free for THIS model; check the next ones
            w.future.set_result(self._lease(dep))
            served.add(w.seq)
        if served:
            self.queue = [w for w in self.queue if w.seq not in served]

    # ----------------------------------------------------------------- API
    def check_model(self, model: str, exclude: frozenset[str] = frozenset()) -> None:
        deps = self.deployments.get(model)
        if not deps:
            raise ModelNotFound(model)
        if not any(d.machine.available and d.machine.name not in exclude for d in deps):
            raise NoHealthyBackend(model)

    async def acquire(self, model: str, priority: int, *, exclude: frozenset[str] = frozenset(),
                      timeout: Optional[float] = None, is_disconnected=None,
                      tag: Optional[str] = None) -> tuple[Lease, float]:
        """Returns (lease, seconds spent in queue)."""
        self.check_model(model, exclude)

        # Fast-path if a machine is free AND no higher-priority
        # (or equal-priority, arrived earlier) waiter is queued for this model.
        now = time.monotonic()
        if not any(w.model == model and self.effective_priority(w, now) <= priority for w in self.queue):
            dep = self._pick(model, exclude)
            if dep is not None:
                return self._lease(dep), 0.0

        if len(self.queue) >= self.server.max_queue_size:
            raise QueueFull()

        w = Waiter(model=model, priority=priority, exclude=exclude, tag=tag)
        self.queue.append(w)
        log.info("Queued: model=%s prio=%s (queue=%d)", model, PRIORITY_NAMES[priority], len(self.queue))
        self.dispatch()

        timeout = self.server.queue_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        try:
            while True:
                done, _ = await asyncio.wait({w.future}, timeout=min(1.0, max(0.0, deadline - time.monotonic())))
                if done:
                    return w.future.result(), time.monotonic() - w.enqueued_at
                if is_disconnected is not None and await is_disconnected():
                    raise asyncio.CancelledError("client disconnected")
                if time.monotonic() >= deadline:
                    raise QueueTimeout()
                # The machine may have gone down while waiting
                self.check_model(model, exclude)
        except BaseException:
            if w.future.done() and not w.future.cancelled() and w.future.exception() is None:
                self.release(w.future.result())  # attribuée au même moment : on rend la place
            else:
                w.future.cancel()
            self.queue = [x for x in self.queue if x is not w]
            raise

    def release(self, lease: Lease) -> None:
        if lease.released:
            return
        lease.released = True
        lease.deployment.active -= 1
        lease.machine.active -= 1
        self.dispatch()

    def set_health(self, machine: Machine, healthy: bool, error: Optional[str] = None) -> None:
        changed = machine.healthy != healthy
        machine.healthy = healthy
        machine.last_error = error
        machine.last_check = time.time()
        if changed:
            log.warning("Machine %s: %s%s", machine.name, "OK" if healthy else "OFFLINE",
                        f" ({error})" if error else "")
        if healthy:
            self.dispatch()
        else:
            self._fail_orphans()

    def _fail_orphans(self) -> None:
        """Queue entries that no remaining machine can serve are failed."""
        for w in list(self.queue):
            try:
                self.check_model(w.model, w.exclude)
            except (NoHealthyBackend, ModelNotFound) as e:
                if not w.future.done():
                    w.future.set_exception(e)

    # -------------------------------------------------------------- statut
    def machine_info(self, m: Machine) -> dict:
        deps = {d.model.name: d for ds in self.deployments.values() for d in ds if d.machine is m}
        return {
            "name": m.name, "url": m.cfg.url, "enabled": m.cfg.enabled, "healthy": m.healthy,
            "last_error": m.last_error, "last_check": m.last_check or None,
            "performance": m.cfg.performance, "active": m.active, "max_concurrent": m.cfg.max_concurrent,
            "supports_stream_options": m.cfg.supports_stream_options, "has_api_key": bool(m.cfg.api_key),
            "observed_prefill_tps": round(m.ema_prefill, 1) if m.ema_prefill else None,
            "observed_gen_tps": round(m.ema_gen, 1) if m.ema_gen else None,
            "models": [{
                "name": mo.name, "type": mo.type, "backend_name": mo.backend_name,
                "max_concurrent": mo.max_concurrent,
                "active": deps[mo.name].active if mo.name in deps else 0,
            } for mo in m.cfg.models],
        }

    def snapshot(self) -> dict:
        now = time.monotonic()
        return {
            "machines": [self.machine_info(m) for m in self.machines.values()],
            "queue": [{
                "tag": w.tag, "model": w.model, "priority": PRIORITY_NAMES[w.priority],
                "effective_priority": PRIORITY_NAMES[self.effective_priority(w, now)],
                "waiting_s": round(now - w.enqueued_at, 2),
            } for w in sorted(self.queue, key=lambda w: (self.effective_priority(w, now), w.seq))],
        }
