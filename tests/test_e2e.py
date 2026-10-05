"""Tests de bout en bout : faux backends + routeur lancés en sous-processus.

pytest -v tests/
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
ROUTER = "http://127.0.0.1:9100"
ADMIN = "http://127.0.0.1:9150"
KEY = "sk-router-test-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {KEY}"}

BACKENDS = [
    # port, name, models, prefill, gen, extra
    (9101, "fast", "llama,shared", 4000, 200, []),
    (9102, "slow", "llama,solo,bge,shared", 1000, 50, []),
    (9103, "lcpp", "lcpp", 3000, 80, ["--timings"]),
]

CONFIG = """
server:
  host: 127.0.0.1
  port: 9100
  api_keys:
    - {{name: tests, key: "sk-router-test-0123456789abcdef"}}
  machines_file: machines.yaml
  queue_timeout: 30
  health_check_interval: 0
  max_retries: 1
  db_path: {tmp}/m.db
  log_file: {tmp}/m.jsonl
machines:
  - name: dead
    url: http://127.0.0.1:9199/v1
    performance: 1000
    max_concurrent: 4
    models: [llama, ghost]
  - name: fast
    url: http://127.0.0.1:9101/v1
    performance: 100
    max_concurrent: 1
    models: [llama, shared]
  - name: slow
    url: http://127.0.0.1:9102/v1
    performance: 30
    max_concurrent: 1
    models:
      - llama
      - solo
      - shared
      - {{name: bge-m3, backend_name: bge, type: embedding}}
  - name: lcpp
    url: http://127.0.0.1:9103/v1
    performance: 50
    max_concurrent: 1
    models: [lcpp]
admin:
  host: 127.0.0.1
  port: 9150
"""


def wait_up(url, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            httpx.get(url, timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{url} ne répond pas")


@pytest.fixture(scope="session", autouse=True)
def stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("router")
    procs = []
    for port, name, models, pf, gen, extra in BACKENDS:
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "tests/fake_backend.py"), "--port", str(port),
                                       "--name", name, "--models", models, "--prefill-tps", str(pf),
                                       "--gen-tps", str(gen), *extra]))
    cfg = tmp / "config.yaml"
    cfg.write_text(CONFIG.format(tmp=tmp))
    log = open(tmp / "router.log", "w")
    procs.append(subprocess.Popen([sys.executable, "-m", "llm_router", "-c", str(cfg)], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT))
    for port, *_ in BACKENDS:
        wait_up(f"http://127.0.0.1:{port}/v1/models")
    wait_up(f"{ROUTER}/health")
    wait_up(f"{ADMIN}/api/overview")
    yield tmp
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    log.close()
    print((tmp / "router.log").read_text())


def chat(content, model="llama", max_tokens=10, **kw):
    return {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, **kw}


def last_records(n=1):
    return httpx.get(f"{ADMIN}/api/requests", params={"limit": n}).json()["rows"]


def post(path, json, **kw):
    return httpx.post(f"{ROUTER}{path}", json=json, headers={**AUTH, **kw.pop("headers", {})},
                      timeout=kw.pop("timeout", 30), **kw)


def machine(name):
    return next(m for m in httpx.get(f"{ADMIN}/api/machines").json()["machines"] if m["name"] == name)


# ---------------------------------------------------------------------------
def test_unknown_model_404():
    r = post("/v1/chat/completions", json=chat("x", model="nope"))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_wrong_endpoint_type():
    r = post("/v1/embeddings", json={"model": "llama", "input": "x"})
    assert r.status_code == 400


def test_failover_from_dead_machine():
    """'dead' est la plus performante mais injoignable : on bascule et on la marque hors ligne."""
    r = post("/v1/chat/completions", json=chat("failover"), timeout=30)
    assert r.status_code == 200, r.text
    assert r.headers["x-router-machine"] == "fast"
    assert machine("dead")["healthy"] is False


def test_model_only_on_dead_machine_503():
    r = post("/v1/chat/completions", json=chat("x", model="ghost"))
    assert r.status_code == 503


def test_non_stream_chat_and_metrics():
    r = post("/v1/chat/completions", json=chat("hello", max_tokens=20), timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "llama"
    assert data["choices"][0]["message"]["content"].startswith("w0 w1")
    assert data["usage"]["completion_tokens"] == 20
    rec = last_records()[0]
    assert rec["metrics_source"] == "measured"
    assert rec["machine"] == "fast"
    assert 100 < rec["gen_tps"] < 260  # le faux backend génère à 200 tok/s
    assert rec["prefill_tps"] > 0


def test_stream_chat_hides_injected_usage():
    chunks = []
    with httpx.stream("POST", f"{ROUTER}/v1/chat/completions", headers=AUTH, json=chat("s", max_tokens=15, stream=True),
                      timeout=30) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text.count("w") == 15
    assert all(c["choices"] for c in chunks), "le chunk d'usage injecté ne doit pas être renvoyé"
    time.sleep(0.2)
    rec = last_records()[0]
    assert rec["stream"] == 1 and rec["completion_tokens"] == 15 and rec["gen_tps"]


def test_stream_with_usage_requested():
    with httpx.stream("POST", f"{ROUTER}/v1/chat/completions", headers=AUTH,
                      json=chat("s", max_tokens=5, stream=True, stream_options={"include_usage": True}),
                      timeout=30) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: {")]
    assert json.loads(lines[-1][6:])["usage"]["completion_tokens"] == 5


def test_completions_endpoint():
    r = post("/v1/completions", json={"model": "llama", "prompt": "abc", "max_tokens": 4}, timeout=30)
    assert r.status_code == 200 and r.json()["choices"][0]["text"] == "w0 w1 w2 w3 "


def test_backend_timings_preferred():
    r = post("/v1/chat/completions", json=chat("t", model="lcpp"), timeout=30)
    assert r.status_code == 200
    rec = last_records()[0]
    assert rec["metrics_source"] == "backend_timings"
    assert rec["prefill_tps"] == 3000 and rec["gen_tps"] == 80


def test_embeddings():
    r = post("/v1/embeddings", json={"model": "bge-m3", "input": ["a" * 400, "b"]}, timeout=30)
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "bge-m3" and len(r.json()["data"]) == 2
    rec = last_records()[0]
    assert rec["prefill_tps"] and rec["machine"] == "slow"


def test_priority_queue_order():
    """'solo' n'existe que sur 'slow' (1 slot). On l'occupe, puis on empile low, medium, high."""
    results = {}

    def send(tag, prio, delay, max_tokens=5):
        time.sleep(delay)
        r = post("/v1/chat/completions", headers={"X-Priority": prio},
                       json=chat(tag, model="solo", max_tokens=max_tokens), timeout=60)
        results[tag] = r

    threads = [threading.Thread(target=send, args=a) for a in [
        ("BLOCK", "low", 0.0, 50),   # ~1 s de génération
        ("L", "low", 0.2), ("M", "medium", 0.3), ("H", "high", 0.4), ("H2", "high", 0.5),
    ]]
    for t in threads:
        t.start()
    time.sleep(0.7)
    act = httpx.get(f"{ADMIN}/api/active").json()["requests"]
    q = [x for x in act if x["state"] == "queued"]
    assert [x["priority"] for x in q] == ["high", "high", "medium", "low"]
    assert [x["queue_position"] for x in q] == [1, 2, 3, 4]
    running = [x for x in act if x["state"] == "running"]
    assert len(running) == 1 and running[0]["machine"] == "slow" and running[0]["client"] == "tests"
    for t in threads:
        t.join()
    assert all(r.status_code == 200 for r in results.values())
    started = httpx.get("http://127.0.0.1:9102/log").json()["started"]
    assert started[-5:] == ["BLOCK", "H", "H2", "M", "L"]
    waits = {tag: int(r.headers["x-router-queue-wait-ms"]) for tag, r in results.items()}
    assert waits["BLOCK"] == 0 and waits["H"] > 0


def test_client_disconnect_removes_from_queue():
    def block():
        post("/v1/chat/completions", json=chat("BLOCK2", model="solo", max_tokens=100), timeout=60)

    t = threading.Thread(target=block)
    t.start()
    time.sleep(0.2)
    with pytest.raises(httpx.ReadTimeout):
        post("/v1/chat/completions", json=chat("GONE", model="solo"), timeout=0.4)
    time.sleep(1.3)
    assert [x for x in httpx.get(f"{ADMIN}/api/active").json()["requests"] if x["state"] == "queued"] == []
    t.join()
    assert "GONE" not in httpx.get("http://127.0.0.1:9102/log").json()["started"]


def test_priority_in_body_and_invalid():
    r = post("/v1/chat/completions", json=chat("p", priority="high"), timeout=30)
    assert r.status_code == 200
    assert last_records()[0]["priority"] == "high"
    r = post("/v1/chat/completions", json=chat("p", priority="urgent"))
    assert r.status_code == 400


def test_load_spreads_to_second_machine():
    """'shared' : fast (1 slot) occupée -> la 2e requête part sur slow au lieu d'attendre."""
    machines = []

    def send():
        r = post("/v1/chat/completions", json=chat("x", model="shared", max_tokens=30), timeout=30)
        machines.append(r.headers["x-router-machine"])

    ts = [threading.Thread(target=send) for _ in range(2)]
    for t in ts:
        t.start()
        time.sleep(0.05)
    for t in ts:
        t.join()
    assert sorted(machines) == ["fast", "slow"]


def test_stats_endpoint():
    s = httpx.get(f"{ADMIN}/api/stats").json()["stats"]
    row = next(x for x in s if x["machine"] == "fast" and x["model"] == "llama")
    assert row["requests"] >= 3 and row["avg_gen_tps"] > 0
    models = httpx.get(f"{ROUTER}/v1/models", headers=AUTH).json()["data"]
    assert {m["id"] for m in models} >= {"llama", "bge-m3", "solo"}


# --------------------------------------------------------------------------- authentification
def test_api_key_required():
    body = chat("x")
    assert httpx.post(f"{ROUTER}/v1/chat/completions", json=body).status_code == 401
    r = httpx.post(f"{ROUTER}/v1/chat/completions", json=body, headers={"Authorization": "Bearer mauvaise-cle-xxxxxxxx"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert httpx.get(f"{ROUTER}/v1/models").status_code == 401
    assert httpx.post(f"{ROUTER}/v1/chat/completions", json=body, headers={"X-API-Key": KEY}, timeout=30).status_code == 200
    assert httpx.get(f"{ROUTER}/health").status_code == 200          # public
    assert httpx.get(f"{ROUTER}/admin/status").status_code in (401, 404)  # pas de console sur le port public


# --------------------------------------------------------------------------- console : suivi live
def test_live_tracking_shows_tokens():
    seen = {}

    def run():
        with httpx.stream("POST", f"{ROUTER}/v1/chat/completions", headers=AUTH,
                          json=chat("live", model="solo", max_tokens=60, stream=True), timeout=30) as r:
            for _ in r.iter_lines():
                pass

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.6)
    act = httpx.get(f"{ADMIN}/api/active").json()
    t.join()
    req = next(x for x in act["requests"] if x["model"] == "solo")
    assert req["state"] == "running" and req["phase"] == "génération"
    assert req["tokens"] > 5 and 30 < req["live_tps"] < 70
    assert next(m for m in act["machines"] if m["name"] == "slow")["active"] == 1
    time.sleep(0.2)
    assert httpx.get(f"{ADMIN}/api/active").json()["requests"] == []


def test_history_filters():
    r = httpx.get(f"{ADMIN}/api/requests", params={"machine": "lcpp"}).json()
    assert r["total"] >= 1 and all(x["machine"] == "lcpp" for x in r["rows"])
    assert "fast" in r["facets"]["machine"]
    err = httpx.get(f"{ADMIN}/api/requests", params={"status": "error"}).json()
    assert all(x["status"] >= 400 for x in err["rows"])
    assert httpx.get(f"{ADMIN}/api/overview").json()["last_hour"]["requests"] > 5


# --------------------------------------------------------------------------- console : gestion du pool
def test_add_update_disable_remove_machine(stack):
    probe = httpx.post(f"{ADMIN}/api/probe", json={"url": "http://127.0.0.1:9101/v1"}).json()
    assert probe["ok"] and "llama" in probe["models"]

    # le modèle n'existe pas encore
    assert post("/v1/chat/completions", json=chat("x", model="neo")).status_code == 404

    new = {"name": "fast-bis", "url": "http://127.0.0.1:9101/v1/", "performance": 80, "max_concurrent": 2,
           "models": [{"name": "neo", "backend_name": "llama", "type": "llm"}]}
    r = httpx.post(f"{ADMIN}/api/machines", json=new)
    assert r.status_code == 201, r.text
    assert r.json()["healthy"] is True and r.json()["url"] == "http://127.0.0.1:9101/v1"
    assert httpx.post(f"{ADMIN}/api/machines", json=new).status_code == 400  # doublon

    r = post("/v1/chat/completions", json=chat("x", model="neo"))
    assert r.status_code == 200 and r.headers["x-router-machine"] == "fast-bis" and r.json()["model"] == "neo"

    # persistance
    saved = (stack / "machines.yaml").read_text()
    assert "fast-bis" in saved and "neo" in saved

    # incohérence de type refusée
    bad = dict(new, name="x2", models=[{"name": "llama", "type": "embedding"}])
    assert httpx.post(f"{ADMIN}/api/machines", json=bad).status_code == 400

    # modification
    r = httpx.put(f"{ADMIN}/api/machines/fast-bis", json=dict(new, performance=5))
    assert r.status_code == 200 and r.json()["performance"] == 5

    # désactivation -> plus servi
    assert httpx.post(f"{ADMIN}/api/machines/fast-bis/disable").json()["enabled"] is False
    assert post("/v1/chat/completions", json=chat("x", model="neo")).status_code == 503
    assert httpx.post(f"{ADMIN}/api/machines/fast-bis/enable").json()["enabled"] is True
    assert post("/v1/chat/completions", json=chat("x", model="neo")).status_code == 200

    # suppression
    assert httpx.delete(f"{ADMIN}/api/machines/fast-bis").status_code == 200
    assert post("/v1/chat/completions", json=chat("x", model="neo")).status_code == 404
    assert "fast-bis" not in (stack / "machines.yaml").read_text()
    assert httpx.delete(f"{ADMIN}/api/machines/fast-bis").status_code == 404


def test_remove_machine_while_request_running():
    """Une requête en cours se termine normalement même si on retire sa machine."""
    httpx.post(f"{ADMIN}/api/machines", json={"name": "tmp", "url": "http://127.0.0.1:9102/v1",
                                                "models": [{"name": "tmpm", "backend_name": "solo"}]})
    res = {}

    def run():
        res["r"] = post("/v1/chat/completions", json=chat("x", model="tmpm", max_tokens=40))

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.3)
    assert httpx.delete(f"{ADMIN}/api/machines/tmp").json()["running_requests_left_to_finish"] == 1
    t.join()
    assert res["r"].status_code == 200
    assert machine("slow")["active"] == 0
