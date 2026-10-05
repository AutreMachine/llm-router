"""Shared state between the public API and the admin console."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .config import Config, MachineConfig, machine_from_dict, save_machines
from .metrics import MetricsStore
from .proxy import PerfMeter
from .scheduler import Machine, Scheduler

log = logging.getLogger("llm_router")


@dataclass
class ActiveRequest:
    """In-flight request (queued or running), visible in the admin console."""
    id: str
    endpoint: str
    model: str
    priority: str
    stream: bool
    client: Optional[str]
    client_ip: Optional[str]
    created: float = field(default_factory=time.time)
    state: str = "queued"              # queued | running
    machine: Optional[str] = None
    started: Optional[float] = None
    meter: Optional[PerfMeter] = None
    attempt: int = 1

    def to_dict(self, now: float, queue_pos: Optional[int] = None, eff_prio: Optional[str] = None) -> dict:
        d = {
            "id": self.id, "endpoint": self.endpoint, "model": self.model, "priority": self.priority,
            "effective_priority": eff_prio or self.priority, "stream": self.stream, "client": self.client,
            "client_ip": self.client_ip, "state": self.state, "machine": self.machine, "attempt": self.attempt,
            "created": self.created, "age_s": round(now - self.created, 2), "queue_position": queue_pos,
            "queue_wait_s": round((self.started or now) - self.created, 2),
            "running_s": round(now - self.started, 2) if self.started else None,
            "phase": None, "tokens": None, "live_tps": None, "ttft_ms": None,
        }
        m = self.meter
        if self.state == "running" and m is not None:
            pc = time.perf_counter()
            if m.t_first is None:
                d["phase"] = "prefill" if self.endpoint != "embeddings" else "embedding"
            else:
                d["phase"] = "generation"
                d["tokens"] = m.token_chunks
                d["ttft_ms"] = round((m.t_first - m.t_start) * 1000, 1)
                span = (m.t_last or m.t_first) - m.t_first
                if m.token_chunks > 1 and span > 0:
                    d["live_tps"] = round((m.token_chunks - 1) / span, 1)
                d["since_last_token_s"] = round(pc - (m.t_last or m.t_first), 2)
        return d


class RouterCore:
    def __init__(self, config: Config):
        self.config = config
        srv = config.server
        self.scheduler = Scheduler(config)
        self.metrics = MetricsStore(str(config.resolve(srv.db_path)),
                                    str(config.resolve(srv.log_file)) if srv.log_file else None)
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(srv.request_timeout, connect=srv.connect_timeout),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=100),
        )
        self.active: dict[str, ActiveRequest] = {}
        self.api_keys = srv.parsed_keys()
        self.started_at = time.time()
        self._health_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self.config.server.health_check_interval > 0:
            self._health_task = asyncio.create_task(self._health_loop())
        log.info("Router ready: %d machine(s), models: %s", len(self.scheduler.machines),
                 ", ".join(sorted(self.scheduler.deployments)) or "(none)")

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
        await self.http.aclose()
        self.metrics.close()

    # ------------------------------------------------------------- health
    @staticmethod
    def _auth_headers(api_key: Optional[str]) -> dict:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def probe(self, url: str, api_key: Optional[str] = None) -> dict:
        """Queries <url>/models: used for health checks and model detection."""
        t0 = time.perf_counter()
        r = await self.http.get(f"{url.rstrip('/')}/models", headers=self._auth_headers(api_key),
                                timeout=self.config.server.connect_timeout)
        r.raise_for_status()
        data = r.json()
        models = [x.get("id") for x in (data.get("data") or data.get("models") or []) if isinstance(x, dict)]
        return {"models": [m for m in models if m], "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    async def check_machine(self, m: Machine) -> None:
        try:
            await self.probe(m.cfg.url, m.cfg.api_key)
            self.scheduler.set_health(m, True)
        except Exception as e:  # noqa: BLE001
            self.scheduler.set_health(m, False, f"{type(e).__name__}: {e}")

    async def _health_loop(self) -> None:
        while True:
            machines = [m for m in self.scheduler.machines.values() if m.cfg.enabled]
            await asyncio.gather(*(self.check_machine(m) for m in machines), return_exceptions=True)
            await asyncio.sleep(self.config.server.health_check_interval)

    # ------------------------------------------------------------- pool
    def _persist(self) -> None:
        path = self.config.resolve(self.config.server.machines_file)
        if path is None:
            log.warning("server.machines_file not defined: change will not be persisted")
            return
        save_machines(path, [m.cfg for m in self.scheduler.machines.values()])
        log.info("Machine pool saved to %s", path)

    async def add_machine(self, raw: dict) -> dict:
        mc = machine_from_dict(raw)
        machine = self.scheduler.add_machine(mc)
        self._persist()
        if mc.enabled:
            await self.check_machine(machine)
        return self.scheduler.machine_info(machine)

    async def update_machine(self, name: str, raw: dict) -> dict:
        current = self.scheduler.machines[name]
        raw = dict(raw, name=name)
        if raw.get("api_key") is None:          # field absent / null: key unchanged
            raw["api_key"] = current.cfg.api_key
        mc = machine_from_dict(raw)
        machine = self.scheduler.update_machine(name, mc)
        self._persist()
        if mc.enabled:
            await self.check_machine(machine)
        return self.scheduler.machine_info(machine)

    def remove_machine(self, name: str) -> None:
        self.scheduler.remove_machine(name)
        self._persist()

    async def set_enabled(self, name: str, enabled: bool) -> dict:
        machine = self.scheduler.set_enabled(name, enabled)
        self._persist()
        if enabled:
            await self.check_machine(machine)
        return self.scheduler.machine_info(machine)

    # ------------------------------------------------------------- live tracking
    def active_snapshot(self) -> list[dict]:
        now = time.time()
        snap = self.scheduler.snapshot()["queue"]
        pos = {q["tag"]: (i + 1, q["effective_priority"]) for i, q in enumerate(snap)}
        out = []
        for r in self.active.values():
            p, eff = pos.get(r.id, (None, None))
            out.append(r.to_dict(now, p, eff))
        out.sort(key=lambda d: (d["state"] != "running", d["queue_position"] or 0, d["created"]))
        return out
