[![Docker Pulls](https://badgen.net/docker/pulls/autremachine/llm-router?icon=docker&label=pulls)](https://hub.docker.com/r/autremachine/llm-router/)
[![Docker Stars](https://badgen.net/docker/stars/autremachine/llm-router?icon=docker&label=stars)](https://hub.docker.com/r/autremachine/llm-router/)
[![Docker Image Size](https://badgen.net/docker/size/autremachine/llm-router?icon=docker&label=image%20size)](https://hub.docker.com/r/autremachine/llm-router/)
![Github issues](https://img.shields.io/github/issues/AutreMachine/llm-router.git)
![Github last-commit](https://img.shields.io/github/last-commit/AutreMachine/llm-router.git)

Have you felt the need to add new machines to your local network to host new LLMs ?
Have you felt frustrated with the price of high end GPUs and dreamt of adding several smaller and cheaper LLM machines in your network to process more requests ?
This old RTX 3090 could well help to serve more clients...

The solution ? Being able to add several LLM-capable servers to your network and call just one address. 

The idea : Grow your LLM farm and just call one endpoint.

<img width="1405" height="517" alt="image" src="https://github.com/user-attachments/assets/8f7c30db-7ca5-4024-b5b2-043cb86dffe9" />


# LLM Router

LLM Router is an OpenAI-compatible load balancer / router for LLMs and embedding models spread across multiple machines
(llama.cpp, Ollama, vLLM, LM Studio… : anything that exposes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`).
. 

Single process, two ports:

| Port | Role | Access | Protection |
|---|---|---|---|
| **8000** | OpenAI API (`/v1/...`) | Internet / LAN | **API key required** |
| **8001** | Web admin console | LAN only | IP filter (private networks), no login |

```
 Clients (Internet)  ──HTTPS──▶  reverse proxy ──▶ :8000  API  ─┐
                                                                ├─ priority queue ──▶ LLM / embedding machines
 Browser (LAN)       ───────────────────────────▶ :8001 console ┘
```

## Installation

```bash
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
python -m llm_router --gen-key        # copy the key into config.yaml > server.api_keys
python -m llm_router -c config/config.yaml
```

Then open `http://<router-ip>:8001/` from the local network.
A "data" folder will be created on the disk to store the metrics.

## Adding a new app in the config
In the config, you can create a new item in the api_keys section with a new key (see how to generate in Installation).
This way, you can monitor to which app a call is linked.

## Docker
You can use this docker-compose file :
```
services:
  llm-router:
    image: autremachine/llm-router:latest
    container_name: llm-router
    volumes:
      - /data/llm-router/data:/data/
      - /data/llm-router/config:/config/
    restart: unless-stopped
    ports:
      - "42000:8000"   # OpenAI-compatible API : modify with your port (here : 42000, port 8000 is the one in the server: section of config.yaml)
      - "42001:8001"   # Admin console : modify with your port (here : 42001, port 8001 is the one in the admin: section of config.yaml))

```
The volume "config" map to the folders containing the config.yaml file, and the "data" contains data produced by the API.

## How routing works

1. Each machine is declared with a **performance score** and a number of **slots** (`max_concurrent`).
2. On each request:
   - if no machine declares the model, response **404** (`model_not_found`);
   - if all machines that have it are offline or disabled, response **503**;
   - if a machine is **free**, the request goes to **the most performant** free machine;
   - otherwise, it goes into the **queue**.
3. The queue is sorted by **priority** (`high` > `medium` > `low`) then by arrival order. A `high` request jumps ahead of all pending `medium`/`low` requests. In-progress requests are never interrupted.
4. When a slot becomes free, it goes to the highest-priority request **that can run on that machine**. A request for a different model therefore does not block the queue.

## Usage (standard OpenAI client)

```python
from openai import OpenAI

client = OpenAI(base_url="https://llm.mydomain.com/v1", api_key="sk-router-…")

r = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"X-Priority": "high"},      # or extra_body={"priority": "high"}
)
emb = client.embeddings.create(model="bge-m3", input=["text 1", "text 2"])
```

Headers added to responses: `X-Router-Machine`, `X-Router-Request-Id`, `X-Router-Queue-Wait-Ms`.

## Public API security (port 8000)

- **Key required** on all routes except `GET /health`. Send it as `Authorization: Bearer <key>` (OpenAI format) or `X-API-Key: <key>`.
- The router **refuses to start without a key**, unless `allow_no_auth: true`. It also rejects the example key and keys shorter than 16 characters.
- Multiple **named** keys are supported (`{name, key}`). The name appears in the console and history: you can see which client sent what, and revoke one key without touching the others.
- Keys are compared in constant time. After 20 failures per minute, an IP receives 429s.
- Keys can also come from the environment variable `LLM_ROUTER_API_KEYS="key1,key2"`, to avoid writing them in the config file.
- The public API exposes neither the names nor the URLs of machines. `/v1/models` only lists model names.

> ⚠️ **Put HTTPS in front.** Over plain HTTP, the key is transmitted in clear text over the Internet. The easiest approach is a reverse proxy:
>
> ```
> # Caddyfile: automatic Let's Encrypt certificate
> llm.mydomain.com {
>     reverse_proxy 127.0.0.1:8000 {
>         flush_interval -1        # SSE streaming without buffering
>     }
> }
> ```
>
> With a proxy, set `server.host: 127.0.0.1` and `forwarded_allow_ips: "127.0.0.1"` so that the history records the real client IP.
> Alternatively, `ssl_certfile` / `ssl_keyfile` make the router serve HTTPS directly.

## Admin console (port 8001)

No authentication. Access is restricted to the `admin.allowed_networks` ranges (private networks by default): a public IP receives a 403, even if the port is accidentally exposed. **Do not forward this port** from your router or firewall.

- **Live**
  - Indicators: running and queued requests, machines online, slots used, average throughput over the last hour.
  - Slot usage per machine.
  - Active request list, updated every second: queue position, priority (including effective priority after aging), client, wait time, phase (prefill or generation), TTFT, generated tokens, and live tok/s.
- **History**
  - All past requests, with queue time, TTFT, total duration, tokens, **prefill tok/s** and **generation tok/s**.
  - Filters by period, model, machine, priority, status, and client; search by ID or error text; pagination.
  - Click a row to show details: IP, metrics source, error message.
- **Machines**
  - **Add** a machine: the *Test / detect models* button queries the backend and suggests the models it announces.
  - **Edit**, **disable** (maintenance), **test**, or **remove** a machine.
  - Removal is seamless: in-flight requests on the machine run to completion; queued requests that have no remaining machine for their model receive a 503.
  - Changes take effect immediately and are saved to `machines.yaml`. At startup, this file overrides the `machines` section of `config.yaml`.
- **Statistics**: per machine and per model, request and error counts, prefill and generation (average, min, max), TTFT, queue wait, tokens.

The underlying JSON API is documented at `http://<ip>:8001/api/docs`, useful for scripting (e.g. `curl -X POST :8001/api/machines/gpu-4090/disable`).

## Performance measurement

| Value | Calculation |
|---|---|
| prefill tok/s | `prompt_tokens / (first token received − request sent)` |
| generation tok/s | `(completion_tokens − 1) / (last token − first token)` |
| embeddings | `prompt_tokens / total duration` |

- To measure prefill, the first token is needed. **Non-streaming** requests are therefore streamed internally to the backend, then reassembled into a standard response (`internal_streaming: true`). The client sees no difference.
- The router requests `stream_options.include_usage` to get the exact token count. If the client had not requested it, this usage chunk is stripped from the stream. If a backend rejects the option, uncheck *Backend accepts stream_options*: the router then counts 1 chunk ≈ 1 token.
- If the backend returns its own `timings` (llama.cpp server), they are used in preference, as they exclude network latency.

Metrics go to the console, to `router_metrics.db` (SQLite), to `router_requests.jsonl` (one JSON line per request), and to the logs.

## Tuning `max_concurrent`

This parameter defines when a machine is "free". Set it to the number of requests the backend truly handles in parallel: `--parallel` / `-np` for llama.cpp, `OLLAMA_NUM_PARALLEL` for Ollama, a higher value for vLLM.
If it is too high, requests pile up in the backend's *internal* queue and priorities no longer apply.
If a machine serves an LLM and an embedding model in two separate processes, use `max_concurrent` at the model level to limit each one individually.

## Robustness

- **Health check**: `GET <url>/models` every `health_check_interval` seconds. A failing machine is no longer selected.
- **Failover**: if a machine is unreachable at send time, it is marked offline and the request is retried on another machine (`max_retries`).
- **Queue**:
  - `max_queue_size`: beyond this, response 429;
  - `queue_timeout`: beyond this, response 503;
  - a request is removed from the queue if its client disconnects.
- **Anti-starvation** (optional): with `aging_seconds: 60`, a request gains one priority level every 60 seconds of waiting.
- The queue is in memory, so it is lost on restart. Run **a single process**: no `--workers` or gunicorn multi-worker setup.

## Tests

```bash
pytest -v tests/
```

The tests spin up fake OpenAI backends (`tests/fake_backend.py`). They cover routing, queue priority ordering, failover, streaming, embeddings, metrics, authentication, live tracking, and hot-adding, editing, and removing machines.

## Structure

```
llm_router/
  config.py      YAML configuration, validation, pool persistence
  scheduler.py   machine/model registry, priority queue, machine selection
  core.py        shared state: scheduler, metrics, active requests, health checks, pool management
  api.py         public OpenAI API + API key middleware
  admin.py       console JSON API + LAN filter
  proxy.py       SSE parsing, prefill / generation measurement, response reassembly
  metrics.py     SQLite + JSONL
  static/admin.html   web interface (HTML/JS, no external dependencies)
```
