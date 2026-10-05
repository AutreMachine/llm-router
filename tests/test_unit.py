"""Tests unitaires : planificateur (priorités, vieillissement) et authentification."""
import asyncio

import httpx
import pytest

from llm_router.admin import LanOnlyMiddleware
from llm_router.api import create_api_app
from llm_router.core import RouterCore
from llm_router.config import Config, MachineConfig, ModelConfig, ServerConfig
from llm_router.scheduler import PRIORITIES, ModelNotFound, Scheduler


def make_config(tmp_path, **server):
    return Config(
        server=ServerConfig(db_path=str(tmp_path / "m.db"), log_file=None, health_check_interval=0, **server),
        machines=[MachineConfig(name="a", url="http://x/v1", max_concurrent=1, models=[ModelConfig("m")])],
    )


def test_priority_and_aging(tmp_path):
    async def run():
        sched = Scheduler(make_config(tmp_path, aging_seconds=0.3))
        first, _ = await sched.acquire("m", PRIORITIES["low"])
        order = []

        async def wait(tag, prio, delay):
            await asyncio.sleep(delay)
            lease, _ = await sched.acquire("m", PRIORITIES[prio])
            order.append(tag)
            sched.release(lease)

        tasks = [asyncio.create_task(wait("old-low", "low", 0)),
                 asyncio.create_task(wait("new-high", "high", 0.65))]
        await asyncio.sleep(0.7)       # old-low a vieilli de 2 niveaux -> high, arrivée plus tôt
        sched.release(first)
        await asyncio.gather(*tasks)
        assert order == ["old-low", "new-high"]

        with pytest.raises(ModelNotFound):
            await sched.acquire("inconnu", 1)

    asyncio.run(run())


def test_auth(tmp_path):
    async def run():
        key = "secret-0123456789abcdef"
        app = create_api_app(RouterCore(make_config(tmp_path, api_keys=[key])))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            assert (await c.get("/health")).status_code == 200
            assert (await c.get("/v1/models")).status_code == 401
            r = await c.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
            assert r.status_code == 200 and r.json()["data"][0]["id"] == "m"

    asyncio.run(run())


def test_short_key_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_config(tmp_path, api_keys=["court"]).server.parsed_keys()


def test_lan_filter():
    mw = LanOnlyMiddleware(None, ["127.0.0.0/8", "192.168.0.0/16", "fc00::/7"])
    assert mw.allowed("192.168.1.20") and mw.allowed("127.0.0.1") and mw.allowed("::ffff:192.168.1.2")
    assert not mw.allowed("8.8.8.8") and not mw.allowed("2001:db8::1") and not mw.allowed(None)
