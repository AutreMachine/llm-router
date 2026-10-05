"""Proxy utilities: SSE parsing, prefill/generation measurement, streaming response reassembly."""
from __future__ import annotations

import json
import time
from typing import Any, Optional


def chunk_has_tokens(obj: dict) -> bool:
    """True if the chunk carries at least one generated token (text, reasoning, or tool call)."""
    for ch in obj.get("choices") or []:
        delta = ch.get("delta")
        if delta:
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") \
                    or delta.get("tool_calls") or delta.get("function_call"):
                return True
        elif ch.get("text"):
            return True
    return False


class PerfMeter:
    """Measures TTFT, prefill tok/s and generation tok/s for a request.

    - prefill    = prompt_tokens / (first token - request sent)
    - generation = (completion_tokens - 1) / (last token - first token)
    If the backend returns its own ``timings`` (llama.cpp), they take precedence.
    """

    def __init__(self) -> None:
        self.t_start = time.perf_counter()
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None
        self.t_end: Optional[float] = None
        self.token_chunks = 0
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self.timings: Optional[dict] = None

    def feed(self, obj: dict) -> None:
        now = time.perf_counter()
        if chunk_has_tokens(obj):
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
            self.token_chunks += 1
        self.read_usage(obj)

    def read_usage(self, obj: dict) -> None:
        usage = obj.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = usage.get("prompt_tokens", self.prompt_tokens)
            self.completion_tokens = usage.get("completion_tokens", self.completion_tokens)
        if isinstance(obj.get("timings"), dict):
            self.timings = obj["timings"]

    def finish(self) -> dict[str, Any]:
        self.t_end = time.perf_counter()
        total = self.t_end - self.t_start
        out: dict[str, Any] = {
            "total_ms": total * 1000,
            "ttft_ms": (self.t_first - self.t_start) * 1000 if self.t_first else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prefill_tps": None, "gen_tps": None, "metrics_source": None,
        }
        t = self.timings or {}
        if t.get("prompt_per_second") or t.get("predicted_per_second"):
            out["prefill_tps"] = t.get("prompt_per_second")
            out["gen_tps"] = t.get("predicted_per_second")
            out["prompt_tokens"] = out["prompt_tokens"] or t.get("prompt_n")
            out["completion_tokens"] = out["completion_tokens"] or t.get("predicted_n")
            out["metrics_source"] = "backend_timings"
            return out

        if self.t_first is not None:
            completion = self.completion_tokens
            if completion is None:  # no usage: approximation 1 chunk ≈ 1 token
                completion = self.token_chunks
                out["completion_tokens"] = completion
                out["metrics_source"] = "measured_approx"
            else:
                out["metrics_source"] = "measured"
            ttft = self.t_first - self.t_start
            if self.prompt_tokens and ttft > 0:
                out["prefill_tps"] = self.prompt_tokens / ttft
            gen_time = (self.t_last or self.t_first) - self.t_first
            if completion and completion > 1 and gen_time > 0:
                out["gen_tps"] = (completion - 1) / gen_time
        elif self.completion_tokens and total > 0:
            # Non-streaming request: only the total time is known
            out["gen_tps"] = self.completion_tokens / total
            out["metrics_source"] = "total_only"
        return out


def parse_sse_line(line: str) -> tuple[str, Optional[dict]]:
    """Returns ('data', obj) | ('done', None) | ('other', None)."""
    if not line.startswith("data:"):
        return "other", None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "done", None
    try:
        return "data", json.loads(payload)
    except json.JSONDecodeError:
        return "other", None


class StreamAggregator:
    """Reassembles a non-streaming response (chat.completion / text_completion) from chunks."""

    def __init__(self, kind: str) -> None:
        self.kind = kind  # "chat" | "completion"
        self.base: dict = {}
        self.choices: dict[int, dict] = {}
        self.usage: Optional[dict] = None
        self.timings: Optional[dict] = None

    def feed(self, obj: dict) -> None:
        if not self.base:
            self.base = {k: obj[k] for k in ("id", "created", "model", "system_fingerprint") if k in obj}
        if obj.get("usage"):
            self.usage = obj["usage"]
        if obj.get("timings"):
            self.timings = obj["timings"]
        for ch in obj.get("choices") or []:
            idx = ch.get("index", 0)
            acc = self.choices.setdefault(idx, {"index": idx, "finish_reason": None, "parts": [],
                                                "reasoning": [], "tool_calls": {}, "role": "assistant"})
            if ch.get("finish_reason"):
                acc["finish_reason"] = ch["finish_reason"]
            if self.kind == "completion":
                if ch.get("text"):
                    acc["parts"].append(ch["text"])
                continue
            delta = ch.get("delta") or {}
            if delta.get("role"):
                acc["role"] = delta["role"]
            if delta.get("content"):
                acc["parts"].append(delta["content"])
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                acc["reasoning"].append(r)
            for tc in delta.get("tool_calls") or []:
                t = acc["tool_calls"].setdefault(tc.get("index", 0), {
                    "id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    t["id"] = tc["id"]
                if tc.get("type"):
                    t["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    t["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    t["function"]["arguments"] += fn["arguments"]

    def result(self) -> dict:
        choices = []
        for idx in sorted(self.choices):
            acc = self.choices[idx]
            if self.kind == "completion":
                choices.append({"index": idx, "text": "".join(acc["parts"]),
                                "finish_reason": acc["finish_reason"], "logprobs": None})
                continue
            msg: dict[str, Any] = {"role": acc["role"], "content": "".join(acc["parts"]) or None}
            if acc["reasoning"]:
                msg["reasoning_content"] = "".join(acc["reasoning"])
            if acc["tool_calls"]:
                msg["tool_calls"] = [acc["tool_calls"][i] for i in sorted(acc["tool_calls"])]
            elif msg["content"] is None:
                msg["content"] = ""
            choices.append({"index": idx, "message": msg, "finish_reason": acc["finish_reason"],
                            "logprobs": None})
        out = {
            "id": self.base.get("id"),
            "object": "chat.completion" if self.kind == "chat" else "text_completion",
            "created": self.base.get("created", int(time.time())),
            "model": self.base.get("model"),
            "choices": choices,
        }
        if "system_fingerprint" in self.base:
            out["system_fingerprint"] = self.base["system_fingerprint"]
        if self.usage:
            out["usage"] = self.usage
        if self.timings:
            out["timings"] = self.timings
        return out
