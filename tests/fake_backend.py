"""Faux serveur OpenAI pour les tests (simule prefill et génération à vitesse fixe).

python tests/fake_backend.py --port 9001 --prefill-tps 2000 --gen-tps 50 --models m1,m2
"""
import argparse
import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def build(models: list[str], prefill_tps: float, gen_tps: float, name: str, timings: bool) -> FastAPI:
    app = FastAPI()
    started: list[str] = []

    def prompt_tokens(body) -> int:
        text = json.dumps(body.get("messages") or body.get("prompt") or body.get("input"))
        return max(1, len(text) // 4)

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}

    @app.get("/log")
    async def get_log():
        return {"started": started}

    @app.post("/v1/embeddings")
    async def emb(request: Request):
        body = await request.json()
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        n = prompt_tokens(body)
        await asyncio.sleep(n / prefill_tps)
        return {"object": "list", "model": body["model"],
                "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(inputs))],
                "usage": {"prompt_tokens": n, "total_tokens": n}}

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def chat(request: Request):
        body = await request.json()
        if body["model"] not in models:
            return JSONResponse(status_code=404, content={"error": {"message": f"model {body['model']} not found"}})
        chat_mode = request.url.path.endswith("chat/completions")
        tag = ""
        if chat_mode:
            tag = body["messages"][-1]["content"]
        else:
            tag = body.get("prompt", "")
        started.append(tag)
        n_prompt = prompt_tokens(body)
        n_gen = int(body.get("max_tokens") or 20)
        words = [f"w{i} " for i in range(n_gen)]
        cid, created = f"cmpl-{name}-{len(started)}", int(time.time())

        def chunk(text=None, finish=None, role=False):
            if chat_mode:
                delta = {}
                if role:
                    delta["role"] = "assistant"
                if text is not None:
                    delta["content"] = text
                ch = {"index": 0, "delta": delta, "finish_reason": finish}
                obj = "chat.completion.chunk"
            else:
                ch = {"index": 0, "text": text or "", "finish_reason": finish}
                obj = "text_completion"
            return {"id": cid, "object": obj, "created": created, "model": body["model"], "choices": [ch]}

        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt + n_gen}
        tm = {"prompt_n": n_prompt, "prompt_per_second": prefill_tps,
              "predicted_n": n_gen, "predicted_per_second": gen_tps}

        if body.get("stream"):
            async def gen():
                await asyncio.sleep(n_prompt / prefill_tps)
                for i, w in enumerate(words):
                    if i:
                        await asyncio.sleep(1 / gen_tps)
                    yield f"data: {json.dumps(chunk(w, role=(i == 0)))}\n\n"
                last = chunk(finish="stop")
                if timings:
                    last["timings"] = tm
                yield f"data: {json.dumps(last)}\n\n"
                if (body.get("stream_options") or {}).get("include_usage"):
                    u = {"id": cid, "object": "chat.completion.chunk", "created": created,
                         "model": body["model"], "choices": [], "usage": usage}
                    yield f"data: {json.dumps(u)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        await asyncio.sleep(n_prompt / prefill_tps + (n_gen - 1) / gen_tps)
        text = "".join(words)
        choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                  if chat_mode else {"index": 0, "text": text, "finish_reason": "stop"})
        return {"id": cid, "object": "chat.completion" if chat_mode else "text_completion",
                "created": created, "model": body["model"], "choices": [choice], "usage": usage}

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--name", default="fake")
    p.add_argument("--models", required=True)
    p.add_argument("--prefill-tps", type=float, default=1000)
    p.add_argument("--gen-tps", type=float, default=50)
    p.add_argument("--timings", action="store_true", help="renvoie des timings façon llama.cpp")
    a = p.parse_args()
    uvicorn.run(build(a.models.split(","), a.prefill_tps, a.gen_tps, a.name, a.timings),
                host="127.0.0.1", port=a.port, log_level="warning")
