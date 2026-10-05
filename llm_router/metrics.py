"""Performance logging: SQLite (for stats) + JSONL (human-readable / ingestible)."""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

log = logging.getLogger("llm_router.metrics")

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL    NOT NULL,
    request_id        TEXT,
    endpoint          TEXT,
    model             TEXT,
    machine           TEXT,
    priority          TEXT,
    stream            INTEGER,
    status            INTEGER,
    queue_wait_ms     REAL,
    ttft_ms           REAL,
    total_ms          REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    prefill_tps       REAL,
    gen_tps           REAL,
    metrics_source    TEXT,
    error             TEXT,
    client            TEXT,
    client_ip         TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_mm ON requests(machine, model);
"""


@dataclass
class RequestRecord:
    request_id: str
    endpoint: str
    model: str
    machine: Optional[str]
    priority: str
    stream: bool
    status: int
    queue_wait_ms: float = 0.0
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    prefill_tps: Optional[float] = None
    gen_tps: Optional[float] = None
    metrics_source: Optional[str] = None   # "measured" | "backend_timings" | "total_only"
    error: Optional[str] = None
    client: Optional[str] = None          # name of the API key used
    client_ip: Optional[str] = None
    ts: float = 0.0


def _r(x: Optional[float], n: int = 2) -> Optional[float]:
    return None if x is None else round(x, n)


class MetricsStore:
    def __init__(self, db_path: str, jsonl_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        # Migrate databases created by an earlier version
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(requests)")}
        for col in ("client", "client_ip"):
            if col not in cols:
                self._db.execute(f"ALTER TABLE requests ADD COLUMN {col} TEXT")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status)")
        self._db.commit()
        self._jsonl = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None

    def _write(self, rec: RequestRecord) -> None:
        row = asdict(rec)
        with self._lock:
            cols = ",".join(row)
            self._db.execute(f"INSERT INTO requests ({cols}) VALUES ({','.join('?' * len(row))})",
                             [int(v) if isinstance(v, bool) else v for v in row.values()])
            self._db.commit()
            if self._jsonl:
                self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._jsonl.flush()

    async def record(self, rec: RequestRecord) -> None:
        rec.ts = rec.ts or time.time()
        for f in ("queue_wait_ms", "ttft_ms", "total_ms", "prefill_tps", "gen_tps"):
            setattr(rec, f, _r(getattr(rec, f)))
        def f(v, unit=""):
            return "-" if v is None else f"{v}{unit}"
        log.info(
            "%s %s -> %s [%s] status=%s file=%s ttft=%s total=%s prompt=%s compl=%s "
            "prefill=%s gen=%s (%s)%s",
            rec.endpoint, rec.model, f(rec.machine), rec.priority, rec.status, f(rec.queue_wait_ms, "ms"),
            f(rec.ttft_ms, "ms"), f(rec.total_ms, "ms"), f(rec.prompt_tokens), f(rec.completion_tokens),
            f(rec.prefill_tps, " tok/s"), f(rec.gen_tps, " tok/s"), f(rec.metrics_source),
            f" | {rec.error}" if rec.error else "",
        )
        try:
            await asyncio.to_thread(self._write, rec)
        except Exception:  # la journalisation ne doit jamais faire échouer une requête
            log.exception("Failed to write metrics")

    def _stats(self, since: float, model: Optional[str], machine: Optional[str]) -> list[dict]:
        where, args = ["ts >= ?"], [since]
        if model:
            where.append("model = ?"); args.append(model)
        if machine:
            where.append("machine = ?"); args.append(machine)
        sql = f"""
            SELECT machine, model, COUNT(*) AS requests,
                   SUM(status >= 400) AS errors,
                   ROUND(AVG(queue_wait_ms), 1)     AS avg_queue_wait_ms,
                   ROUND(MAX(queue_wait_ms), 1)     AS max_queue_wait_ms,
                   ROUND(AVG(ttft_ms), 1)           AS avg_ttft_ms,
                   ROUND(AVG(prefill_tps), 1)       AS avg_prefill_tps,
                   ROUND(MIN(prefill_tps), 1)       AS min_prefill_tps,
                   ROUND(MAX(prefill_tps), 1)       AS max_prefill_tps,
                   ROUND(AVG(gen_tps), 1)           AS avg_gen_tps,
                   ROUND(MIN(gen_tps), 1)           AS min_gen_tps,
                   ROUND(MAX(gen_tps), 1)           AS max_gen_tps,
                   SUM(prompt_tokens)               AS total_prompt_tokens,
                   SUM(completion_tokens)           AS total_completion_tokens
            FROM requests WHERE {' AND '.join(where)}
            GROUP BY machine, model ORDER BY machine, model"""
        with self._lock:
            cur = self._db.execute(sql, args)
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _recent(self, limit: int) -> list[dict]:
        with self._lock:
            cur = self._db.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _query(self, limit: int, offset: int, filters: dict) -> dict:
        where, args = [], []
        for col in ("model", "machine", "priority", "client", "endpoint"):
            if filters.get(col):
                where.append(f"{col} = ?"); args.append(filters[col])
        st = filters.get("status")
        if st == "ok":
            where.append("status < 400")
        elif st == "error":
            where.append("status >= 400")
        if filters.get("since"):
            where.append("ts >= ?"); args.append(float(filters["since"]))
        if filters.get("q"):
            where.append("(request_id LIKE ? OR error LIKE ?)"); args += [f"%{filters['q']}%"] * 2
        w = f"WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM requests {w}", args).fetchone()[0]
            cur = self._db.execute(f"SELECT * FROM requests {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                                   args + [limit, offset])
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            facets = {c: [r[0] for r in self._db.execute(
                f"SELECT DISTINCT {c} FROM requests WHERE {c} IS NOT NULL ORDER BY {c}")]
                for c in ("model", "machine", "client")}
        return {"total": total, "rows": rows, "facets": facets}

    def _overview(self, since: float) -> dict:
        with self._lock:
            r = self._db.execute("""
                SELECT COUNT(*), SUM(status >= 400), AVG(queue_wait_ms), AVG(gen_tps), AVG(prefill_tps),
                       SUM(COALESCE(prompt_tokens, 0)), SUM(COALESCE(completion_tokens, 0))
                FROM requests WHERE ts >= ?""", (since,)).fetchone()
        keys = ("requests", "errors", "avg_queue_wait_ms", "avg_gen_tps", "avg_prefill_tps",
                "prompt_tokens", "completion_tokens")
        return {k: (round(v, 1) if isinstance(v, float) else (v or 0)) for k, v in zip(keys, r)}

    async def query(self, limit: int = 50, offset: int = 0, **filters) -> dict:
        return await asyncio.to_thread(self._query, limit, offset, filters)

    async def overview(self, hours: float = 1) -> dict:
        return await asyncio.to_thread(self._overview, time.time() - hours * 3600)

    async def stats(self, hours: float = 24, model: Optional[str] = None, machine: Optional[str] = None):
        return await asyncio.to_thread(self._stats, time.time() - hours * 3600, model, machine)

    async def recent(self, limit: int = 50):
        return await asyncio.to_thread(self._recent, limit)

    def close(self) -> None:
        with self._lock:
            self._db.close()
            if self._jsonl:
                self._jsonl.close()
