"""Public OpenAI-compatible API (internet-facing, protected by API key)."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .core import ActiveRequest, RouterCore
from .metrics import RequestRecord
from .proxy import PerfMeter, StreamAggregator, parse_sse_line
from .scheduler import (PRIORITY_NAMES, Lease, Machine, ModelNotFound, NoHealthyBackend, QueueFull,
                        QueueTimeout, parse_priority)

log = logging.getLogger("llm_router.api")

ENDPOINT_KIND = {"chat/completions": "llm", "completions": "llm", "embeddings": "embedding"}
PUBLIC_PATHS = {"/health"}


def oai_error(status: int, message: str, type_: str, code: Optional[str] = None, headers=None) -> JSONResponse:
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": type_, "param": None, "code": code}})


class BackendUnreachable(Exception):
    pass


class ApiKeyMiddleware:
    """Checks 'Authorization: Bearer <key>' (or 'X-API-Key: <key>') and stores the key name in the scope.

    Pure ASGI middleware: BaseHTTPMiddleware would break client disconnect detection.
    """

    def __init__(self, inner, core: RouterCore):
        self.inner = inner
        self.core = core
        self.failures: dict[str, list[float]] = {}

    def _match(self, token: str) -> Optional[str]:
        found = None
        for k in self.core.api_keys:  # constant-time comparison, no early exit
            if hmac.compare_digest(token.encode(), k.key.encode()):
                found = k.name
        return found

    def _throttled(self, ip: str) -> bool:
        now = time.monotonic()
        hits = [t for t in self.failures.get(ip, []) if now - t < 60]
        self.failures[ip] = hits
        return len(hits) >= 20  # 20 failures / minute / IP

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS or not self.core.api_keys:
            scope.setdefault("state", {})["client_name"] = None
            return await self.inner(scope, receive, send)
        ip = (scope.get("client") or ("?", 0))[0]
        if self._throttled(ip):
            return await oai_error(429, "Too many authentication failures, please try again later",
                                   "rate_limit_error")(scope, receive, send)
        headers = dict(scope["headers"])
        auth = headers.get(b"authorization", b"").decode("latin-1")
        token = auth[7:].strip() if auth[:7].lower() == "bearer " else headers.get(b"x-api-key", b"").decode("latin-1")
        name = self._match(token) if token else None
        if name is None:
            self.failures.setdefault(ip, []).append(time.monotonic())
            log.warning("API key rejected from %s on %s", ip, scope["path"])
            return await oai_error(401, "Missing or invalid API key", "invalid_request_error",
                                   "invalid_api_key")(scope, receive, send)
        scope.setdefault("state", {})["client_name"] = name
        await self.inner(scope, receive, send)


class Ctx:
    """Context for a proxied request."""

    def __init__(self, rec: RequestRecord, ar: ActiveRequest, public: str):
        self.rec, self.ar, self.public = rec, ar, public
        self.lease: Optional[Lease] = None
        self.headers: dict = {}


def create_api_app(core: RouterCore) -> FastAPI:
    srv = core.config.server
    sched, metrics, http = core.scheduler, core.metrics, core.http

    app = FastAPI(title="LLM Router API", version="2.0.0", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.core = core
    app.add_middleware(ApiKeyMiddleware, core=core)

    # ------------------------------------------------------------ helpers
    async def finish(ctx: Ctx) -> None:
        """End of request lifecycle: remove from live tracking and log."""
        core.active.pop(ctx.rec.request_id, None)
        await metrics.record(ctx.rec)

    def backend_headers(m: Machine) -> dict:
        h = {"Content-Type": "application/json"}
        if m.cfg.api_key:
            h["Authorization"] = f"Bearer {m.cfg.api_key}"
        return h

    async def open_upstream(lease: Lease, path: str, body: dict, stream: bool) -> httpx.Response:
        m = lease.machine
        req = http.build_request("POST", f"{m.cfg.url}/{path}", json=body, headers=backend_headers(m))
        try:
            return await http.send(req, stream=stream)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
            raise BackendUnreachable(f"{type(e).__name__}: {e}") from e

    def rewrite_model(obj: dict, public: str) -> dict:
        if isinstance(obj, dict) and "model" in obj:
            obj["model"] = public
        return obj

    def new_meter(ctx: Ctx) -> PerfMeter:
        meter = PerfMeter()
        ctx.ar.meter = meter
        return meter

    async def record_success(ctx: Ctx, meter: PerfMeter, status: int = 200) -> None:
        for k, v in meter.finish().items():
            setattr(ctx.rec, k, v)
        ctx.rec.status = status
        if status < 400:
            ctx.lease.machine.update_perf(ctx.rec.prefill_tps, ctx.rec.gen_tps)
        await finish(ctx)

    async def finish_error(ctx: Ctx, resp: httpx.Response, meter: PerfMeter):
        content = await resp.aread()
        await resp.aclose()
        sched.release(ctx.lease)
        ctx.rec.status = resp.status_code
        ctx.rec.total_ms = meter.finish()["total_ms"]
        ctx.rec.error = content[:500].decode("utf-8", "replace")
        await finish(ctx)
        try:
            return JSONResponse(status_code=resp.status_code, content=json.loads(content), headers=ctx.headers)
        except json.JSONDecodeError:
            return oai_error(resp.status_code, ctx.rec.error, "server_error", headers=ctx.headers)

    # ------------------------------------------------------------ request
    async def handle(request: Request, path: str):
        request_id = uuid.uuid4().hex[:12]
        try:
            body = await request.json()
            assert isinstance(body, dict)
        except Exception:  # noqa: BLE001
            return oai_error(400, "Invalid JSON body", "invalid_request_error")

        model = body.get("model")
        if not model:
            return oai_error(400, "Missing 'model' field", "invalid_request_error")
        try:
            prio = parse_priority(request.headers.get("x-priority") or body.get("priority"), srv.default_priority)
        except ValueError as e:
            return oai_error(400, str(e), "invalid_request_error")
        body.pop("priority", None)

        mtype = sched.model_type(model)
        if mtype is None:
            return oai_error(404, f"Model '{model}' is not available on any machine",
                             "invalid_request_error", "model_not_found")
        if mtype != ENDPOINT_KIND[path]:
            return oai_error(400, f"Model '{model}' is of type '{mtype}', incompatible with /v1/{path}",
                             "invalid_request_error")

        client_stream = bool(body.get("stream")) and path != "embeddings"
        client_name = request.scope.get("state", {}).get("client_name")
        client_ip = request.client.host if request.client else None
        prio_name = PRIORITY_NAMES[prio]
        rec = RequestRecord(request_id=request_id, endpoint=path, model=model, machine=None, priority=prio_name,
                            stream=client_stream, status=0, client=client_name, client_ip=client_ip)
        ar = ActiveRequest(id=request_id, endpoint=path, model=model, priority=prio_name, stream=client_stream,
                           client=client_name, client_ip=client_ip)
        core.active[request_id] = ar
        ctx = Ctx(rec, ar, model)
        try:
            return await relay(request, ctx, path, body, prio, client_stream)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — safety net: the request must not stay "in progress"
            log.exception("[%s] unexpected error", request_id)
            if ctx.lease:
                sched.release(ctx.lease)
            rec.status, rec.error = 500, f"{type(e).__name__}: {e}"
            await finish(ctx)
            return oai_error(500, "Internal router error", "server_error")

    async def relay(request: Request, ctx: Ctx, path: str, body: dict, prio: int, client_stream: bool):
        rec, ar, model = ctx.rec, ctx.ar, ctx.public
        exclude: frozenset[str] = frozenset()

        for attempt in range(srv.max_retries + 1):
            ar.attempt, ar.state, ar.machine, ar.started, ar.meter = attempt + 1, "queued", None, None, None
            # ---- 1. acquire a machine (immediately or after queuing)
            try:
                lease, waited = await sched.acquire(model, prio, exclude=exclude, tag=rec.request_id,
                                                    is_disconnected=request.is_disconnected)
            except (NoHealthyBackend, ModelNotFound) as e:
                rec.status, rec.error = 503, f"no machine available ({type(e).__name__})"
                await finish(ctx)
                return oai_error(503, f"No online machine serves '{model}'", "server_error", "no_backend")
            except QueueFull:
                rec.status, rec.error = 429, "queue full"
                await finish(ctx)
                return oai_error(429, "Queue is full", "rate_limit_error", "queue_full")
            except QueueTimeout:
                rec.status, rec.error = 503, "queue wait timeout exceeded"
                await finish(ctx)
                return oai_error(503, "Queue wait timeout exceeded", "server_error", "queue_timeout")
            except asyncio.CancelledError:
                rec.status, rec.error = 499, "client disconnected while waiting"
                await asyncio.shield(finish(ctx))
                raise

            ctx.lease = lease
            ar.state, ar.machine, ar.started = "running", lease.machine.name, time.time()
            rec.machine = lease.machine.name
            rec.queue_wait_ms += waited * 1000
            upstream = dict(body, model=lease.deployment.model.upstream_name)
            ctx.headers = {"X-Router-Machine": lease.machine.name, "X-Router-Request-Id": rec.request_id,
                           "X-Router-Queue-Wait-Ms": f"{rec.queue_wait_ms:.0f}"}

            # ---- 2. proxy
            try:
                if path == "embeddings":
                    return await do_embeddings(ctx, upstream)
                use_stream = client_stream or srv.internal_streaming
                injected_usage = False
                if use_stream:
                    upstream["stream"] = True
                    so = dict(upstream.get("stream_options") or {})
                    if lease.machine.cfg.supports_stream_options and not so.get("include_usage"):
                        so["include_usage"] = True
                        upstream["stream_options"] = so
                        injected_usage = True
                if client_stream:
                    return await do_stream(ctx, path, upstream, injected_usage)
                if use_stream:
                    return await do_aggregate(ctx, path, upstream)
                return await do_plain(ctx, path, upstream)
            except BackendUnreachable as e:
                sched.set_health(lease.machine, False, str(e))
                sched.release(lease)
                ctx.lease = None
                exclude = exclude | {lease.machine.name}
                log.warning("[%s] %s unreachable (attempt %d): %s", rec.request_id, lease.machine.name,
                            attempt + 1, e)
                rec.error = f"failover: {lease.machine.name} unreachable ({e})"
                continue
            except httpx.HTTPError as e:  # read timeout, etc.: no retry
                sched.release(lease)
                rec.status = 504 if isinstance(e, httpx.TimeoutException) else 502
                rec.error = f"{type(e).__name__}: {e}"
                await finish(ctx)
                return oai_error(rec.status, f"Backend error ({lease.machine.name}): {rec.error}",
                                 "server_error", headers=ctx.headers)
            except BaseException:
                sched.release(lease)
                raise

        rec.status = 502
        await finish(ctx)
        return oai_error(502, f"All machines unreachable for '{model}': {rec.error}", "server_error",
                         "backend_unreachable")

    # --- streaming client
    async def do_stream(ctx: Ctx, path: str, upstream: dict, injected_usage: bool):
        meter = new_meter(ctx)
        resp = await open_upstream(ctx.lease, path, upstream, stream=True)
        if resp.status_code >= 400:
            return await finish_error(ctx, resp, meter)

        async def gen():
            status = 200
            try:
                async for line in resp.aiter_lines():
                    kind, obj = parse_sse_line(line)
                    if kind == "data":
                        meter.feed(obj)
                        if injected_usage and not obj.get("choices") and obj.get("usage"):
                            continue  # usage chunk injected by us: not requested by the client
                        yield f"data: {json.dumps(rewrite_model(obj, ctx.public), ensure_ascii=False)}\n\n"
                    elif kind == "done":
                        yield "data: [DONE]\n\n"
                    elif line.strip():
                        yield line + "\n\n"
            except asyncio.CancelledError:
                status = 499
                ctx.rec.error = "client disconnected"
                raise
            except httpx.HTTPError as e:
                status = 502
                ctx.rec.error = f"stream interrupted: {type(e).__name__}: {e}"
                yield f"data: {json.dumps({'error': {'message': ctx.rec.error, 'type': 'server_error'}})}\n\n"
            finally:
                await resp.aclose()
                sched.release(ctx.lease)
                await asyncio.shield(record_success(ctx, meter, status))

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={**ctx.headers, "Cache-Control": "no-cache"})

    # --- non-streaming client, internal stream (accurate prefill measurement)
    async def do_aggregate(ctx: Ctx, path: str, upstream: dict):
        meter = new_meter(ctx)
        resp = await open_upstream(ctx.lease, path, upstream, stream=True)
        if resp.status_code >= 400:
            return await finish_error(ctx, resp, meter)
        agg = StreamAggregator("chat" if path == "chat/completions" else "completion")
        try:
            async for line in resp.aiter_lines():
                kind, obj = parse_sse_line(line)
                if kind == "data":
                    meter.feed(obj)
                    agg.feed(obj)
        except httpx.HTTPError as e:
            ctx.rec.error = f"stream interrupted: {type(e).__name__}: {e}"
            await record_success(ctx, meter, 502)
            return oai_error(502, ctx.rec.error, "server_error", headers=ctx.headers)
        finally:
            await resp.aclose()
            sched.release(ctx.lease)
        await record_success(ctx, meter)
        return JSONResponse(rewrite_model(agg.result(), ctx.public), headers=ctx.headers)

    # --- end-to-end non-streaming
    async def do_plain(ctx: Ctx, path: str, upstream: dict):
        meter = new_meter(ctx)
        try:
            resp = await open_upstream(ctx.lease, path, upstream, stream=False)
        finally:
            sched.release(ctx.lease)
        if resp.status_code >= 400:
            ctx.rec.status, ctx.rec.error = resp.status_code, resp.text[:500]
            ctx.rec.total_ms = meter.finish()["total_ms"]
            await finish(ctx)
            try:
                return JSONResponse(status_code=resp.status_code, content=resp.json(), headers=ctx.headers)
            except json.JSONDecodeError:
                return oai_error(resp.status_code, ctx.rec.error, "server_error", headers=ctx.headers)
        data = resp.json()
        meter.read_usage(data)
        await record_success(ctx, meter)
        return JSONResponse(rewrite_model(data, ctx.public), headers=ctx.headers)

    # --- embeddings: prefill only
    async def do_embeddings(ctx: Ctx, upstream: dict):
        upstream.pop("stream", None)
        rec = ctx.rec
        new_meter(ctx)
        t0 = time.perf_counter()
        try:
            resp = await open_upstream(ctx.lease, "embeddings", upstream, stream=False)
        finally:
            sched.release(ctx.lease)
        dt = time.perf_counter() - t0
        rec.total_ms = dt * 1000
        if resp.status_code >= 400:
            rec.status, rec.error = resp.status_code, resp.text[:500]
            await finish(ctx)
            try:
                return JSONResponse(status_code=resp.status_code, content=resp.json(), headers=ctx.headers)
            except json.JSONDecodeError:
                return oai_error(resp.status_code, rec.error, "server_error", headers=ctx.headers)
        data = resp.json()
        usage = data.get("usage") or {}
        rec.prompt_tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
        if rec.prompt_tokens and dt > 0:
            rec.prefill_tps = rec.prompt_tokens / dt
        rec.metrics_source = "total_only"
        rec.status = 200
        ctx.lease.machine.update_perf(rec.prefill_tps, None)
        await finish(ctx)
        return JSONResponse(rewrite_model(data, ctx.public), headers=ctx.headers)

    # ------------------------------------------------------------- routes
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await handle(request, "chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle(request, "completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await handle(request, "embeddings")

    @app.get("/v1/models")
    async def list_models():
        data = []
        for name, deps in sorted(sched.deployments.items()):
            data.append({
                "id": name, "object": "model", "created": 0, "owned_by": "llm-router",
                "type": deps[0].model.type,
                "available": any(d.machine.available for d in deps),
            })
        return {"object": "list", "data": data}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
