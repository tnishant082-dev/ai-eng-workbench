from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model TEXT,
                    gateway TEXT,
                    latency_ms REAL,
                    tokens_estimate INTEGER,
                    cost_usd REAL DEFAULT 0,
                    status TEXT,
                    request_json TEXT,
                    response_json TEXT,
                    events_json TEXT
                )
                """
            )
            # migrate older DBs missing cost_usd
            cols = {r[1] for r in conn.execute("PRAGMA table_info(traces)").fetchall()}
            if "cost_usd" not in cols:
                conn.execute("ALTER TABLE traces ADD COLUMN cost_usd REAL DEFAULT 0")
            conn.commit()

    def add(
        self,
        *,
        kind: str,
        model: str,
        gateway: str,
        latency_ms: float,
        tokens_estimate: int,
        status: str,
        request: dict[str, Any],
        response: dict[str, Any],
        events: list[dict[str, Any]] | None = None,
        cost_usd: float = 0.0,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO traces (
                    kind, created_at, model, gateway, latency_ms, tokens_estimate,
                    cost_usd, status, request_json, response_json, events_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    _utcnow(),
                    model,
                    gateway,
                    latency_ms,
                    tokens_estimate,
                    cost_usd,
                    status,
                    json.dumps(request),
                    json.dumps(response),
                    json.dumps(events or []),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, created_at, model, gateway, latency_ms,
                       tokens_estimate, cost_usd, status
                FROM traces ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, trace_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        return {
            "id": d["id"],
            "kind": d["kind"],
            "created_at": d["created_at"],
            "model": d["model"],
            "gateway": d["gateway"],
            "latency_ms": d["latency_ms"],
            "tokens_estimate": d["tokens_estimate"],
            "cost_usd": d.get("cost_usd") or 0,
            "status": d["status"],
            "request": json.loads(d["request_json"]),
            "response": json.loads(d["response_json"]),
            "events": json.loads(d["events_json"] or "[]"),
        }

    def compare(self, a_id: int, b_id: int) -> dict[str, Any] | None:
        a, b = self.get(a_id), self.get(b_id)
        if not a or not b:
            return None
        return {
            "a": {"id": a["id"], "kind": a["kind"], "latency_ms": a["latency_ms"],
                  "tokens_estimate": a["tokens_estimate"], "cost_usd": a["cost_usd"],
                  "status": a["status"], "event_count": len(a["events"])},
            "b": {"id": b["id"], "kind": b["kind"], "latency_ms": b["latency_ms"],
                  "tokens_estimate": b["tokens_estimate"], "cost_usd": b["cost_usd"],
                  "status": b["status"], "event_count": len(b["events"])},
            "delta": {
                "latency_ms": round((b["latency_ms"] or 0) - (a["latency_ms"] or 0), 2),
                "tokens_estimate": (b["tokens_estimate"] or 0) - (a["tokens_estimate"] or 0),
                "cost_usd": round((b["cost_usd"] or 0) - (a["cost_usd"] or 0), 6),
            },
            "events_a": a["events"],
            "events_b": b["events"],
        }
