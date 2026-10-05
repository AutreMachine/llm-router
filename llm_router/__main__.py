"""Entry point.

    python -m llm_router -c config.yaml      # starts the API (port 8000) + admin console (port 8001)
    python -m llm_router --gen-key           # generates an API key
"""
import argparse
import asyncio
import contextlib
import logging
import secrets
import signal
import sys

import uvicorn

from .admin import create_admin_app
from .api import create_api_app
from .config import load_config, Config
from .core import RouterCore

log = logging.getLogger("llm_router")


class _Server(uvicorn.Server):
    """Two servers in the same process: signals are handled once, here."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def run(cfg: Config, log_level: str) -> None:
    core = RouterCore(cfg)
    srv, adm = cfg.server, cfg.admin
    servers = [_Server(uvicorn.Config(
        create_api_app(core), host=srv.host, port=srv.port, log_level=log_level, access_log=False,
        ssl_certfile=srv.ssl_certfile, ssl_keyfile=srv.ssl_keyfile,
        proxy_headers=bool(srv.forwarded_allow_ips), forwarded_allow_ips=srv.forwarded_allow_ips,
    ))]
    if adm.enabled:
        servers.append(_Server(uvicorn.Config(
            create_admin_app(core), host=adm.host, port=adm.port, log_level=log_level, access_log=False,
            proxy_headers=False,
        )))

    def stop(*_):
        for s in servers:
            s.should_exit = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows
            loop.add_signal_handler(sig, stop)

    await core.start()
    scheme = "https" if srv.ssl_certfile else "http"
    log.info("OpenAI API   : %s://%s:%d/v1 (%s)", scheme, srv.host, srv.port,
             f"{len(core.api_keys)} API key(s)" if core.api_keys else "NO AUTHENTICATION")
    if adm.enabled:
        log.info("Admin console: http://%s:%d/ (allowed networks: %s)", adm.host, adm.port,
                 ", ".join(adm.allowed_networks))
    try:
        await asyncio.gather(*(s.serve() for s in servers))
    finally:
        await core.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="Load balancer / OpenAI-compatible router for LLMs and embeddings")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("--log-level", default="info")
    p.add_argument("--gen-key", action="store_true", help="print a new random API key and exit")
    args = p.parse_args()

    if args.gen_key:
        print("sk-router-" + secrets.token_urlsafe(32))
        return

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # only warning log level
    cfg = load_config(args.config)
    if not cfg.server.parsed_keys() and not cfg.server.allow_no_auth:
        sys.exit("No API key configured (server.api_keys or LLM_ROUTER_API_KEYS).\n"
                 "Generate one with: python -m llm_router --gen-key\n"
                 "Or set server.allow_no_auth: true (not recommended if the API is exposed to the Internet).")
    if (cfg.admin.enabled and cfg.admin.port == cfg.server.port
            and cfg.admin.host in (cfg.server.host, "0.0.0.0")):
        sys.exit("admin.port must be different from server.port")
    #check config
    
    asyncio.run(run(cfg, args.log_level))


if __name__ == "__main__":
    main()
