"""Web administration console (local network only, no authentication)."""
from __future__ import annotations

import ipaddress
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from .core import RouterCore

log = logging.getLogger("llm_router.admin")
STATIC = Path(__file__).parent / "static"


class LanOnlyMiddleware:
    """Rejects any IP outside the allowed networks (safety net if the port is accidentally exposed)."""

    def __init__(self, inner, networks: list[str]):
        self.inner = inner
        self.networks = [ipaddress.ip_network(n, strict=False) for n in networks]

    def allowed(self, host: Optional[str]) -> bool:
        if not host:
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return any(ip in n for n in self.networks)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            host = (scope.get("client") or (None, 0))[0]
            if not self.allowed(host):
                log.warning("Console access denied to %s", host)
                return await JSONResponse({"detail": "Access restricted to local network"}, status_code=403)(
                    scope, receive, send)
        await self.inner(scope, receive, send)


def create_admin_app(core: RouterCore) -> FastAPI:
    sched, metrics = core.scheduler, core.metrics
    app = FastAPI(title="LLM Router — console", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.core = core
    app.add_middleware(LanOnlyMiddleware, networks=core.config.admin.allowed_networks)

    def bad_request(e: Exception):
        raise HTTPException(status_code=400, detail=str(e))

    def get_machine(name: str):
        if name not in sched.machines:
            raise HTTPException(status_code=404, detail=f"Unknown machine: {name}")
        return sched.machines[name]

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "admin.html", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------- live
    @app.get("/api/overview")
    async def overview():
        machines = list(sched.machines.values())
        active = list(core.active.values())
        return {
            "uptime_s": round(time.time() - core.started_at),
            "running": sum(a.state == "running" for a in active),
            "queued": len(sched.queue),
            "machines_total": len(machines),
            "machines_up": sum(m.available for m in machines),
            "slots_used": sum(m.active for m in machines),
            "slots_total": sum(m.cfg.max_concurrent for m in machines if m.available),
            "models": sorted(sched.deployments),
            "last_hour": await metrics.overview(1),
            "auth_enabled": bool(core.api_keys),
        }

    @app.get("/api/active")
    async def active():
        return {"requests": core.active_snapshot(),
                "machines": [sched.machine_info(m) for m in sched.machines.values()]}

    # ------------------------------------------------------------- historique
    @app.get("/api/requests")
    async def requests(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                       model: Optional[str] = None, machine: Optional[str] = None,
                       priority: Optional[str] = None, client: Optional[str] = None,
                       endpoint: Optional[str] = None, status: Optional[str] = None,
                       hours: Optional[float] = None, q: Optional[str] = None):
        since = time.time() - hours * 3600 if hours else None
        return await metrics.query(limit, offset, model=model, machine=machine, priority=priority,
                                   client=client, endpoint=endpoint, status=status, since=since, q=q)

    @app.get("/api/stats")
    async def stats(hours: float = 24, model: Optional[str] = None, machine: Optional[str] = None):
        return {"hours": hours, "stats": await metrics.stats(hours, model, machine)}

    # ------------------------------------------------------------- machines
    @app.get("/api/machines")
    async def list_machines():
        return {"machines": [sched.machine_info(m) for m in sched.machines.values()]}

    @app.post("/api/machines", status_code=201)
    async def add_machine(payload: dict = Body(...)):
        try:
            return await core.add_machine(payload)
        except (ValueError, TypeError) as e:
            bad_request(e)

    @app.put("/api/machines/{name}")
    async def update_machine(name: str, payload: dict = Body(...)):
        get_machine(name)
        try:
            return await core.update_machine(name, payload)
        except (ValueError, TypeError) as e:
            bad_request(e)

    @app.delete("/api/machines/{name}")
    async def delete_machine(name: str):
        m = get_machine(name)
        running = m.active
        core.remove_machine(name)
        return {"removed": name, "running_requests_left_to_finish": running}

    @app.post("/api/machines/{name}/enable")
    async def enable(name: str):
        get_machine(name)
        return await core.set_enabled(name, True)

    @app.post("/api/machines/{name}/disable")
    async def disable(name: str):
        get_machine(name)
        return await core.set_enabled(name, False)

    @app.post("/api/machines/{name}/check")
    async def check(name: str):
        m = get_machine(name)
        await core.check_machine(m)
        return sched.machine_info(m)

    @app.post("/api/probe")
    async def probe(payload: dict = Body(...)):
        """Tests a backend URL and lists its models (assists the add-machine form)."""
        url = (payload.get("url") or "").strip()
        api_key = payload.get("api_key") or None
        if not api_key and payload.get("machine") in sched.machines:
            api_key = sched.machines[payload["machine"]].cfg.api_key
        if not url.startswith(("http://", "https://")):
            bad_request(ValueError("Invalid URL"))
        try:
            return {"ok": True, **(await core.probe(url, api_key))}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return app
