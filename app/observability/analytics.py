"""Token / cost / error analytics over stored SQLite traces."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.observability.store import TraceStore


def trace_analytics(store: TraceStore, limit: int = 200) -> dict[str, Any]:
    rows = store.list(limit=limit)
    if not rows:
        return {
            "n": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "error_rate": 0.0,
            "by_kind": {},
            "avg_latency_ms": 0.0,
        }
    by_kind: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "tokens": 0, "cost": 0.0, "errors": 0, "latency": 0.0})
    total_tokens = 0
    total_cost = 0.0
    errors = 0
    latency_sum = 0.0
    for r in rows:
        kind = r.get("kind") or "unknown"
        tok = int(r.get("tokens_estimate") or 0)
        cost = float(r.get("cost_usd") or 0)
        lat = float(r.get("latency_ms") or 0)
        status = (r.get("status") or "").lower()
        is_err = status not in ("ok", "ok_with_recovery", "pending_approval", "")
        total_tokens += tok
        total_cost += cost
        latency_sum += lat
        if is_err:
            errors += 1
        b = by_kind[kind]
        b["n"] += 1
        b["tokens"] += tok
        b["cost"] += cost
        b["latency"] += lat
        if is_err:
            b["errors"] += 1
    n = len(rows)
    return {
        "n": n,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "error_rate": round(errors / n, 4) if n else 0.0,
        "avg_latency_ms": round(latency_sum / n, 2) if n else 0.0,
        "by_kind": {
            k: {
                "n": int(v["n"]),
                "tokens": int(v["tokens"]),
                "cost_usd": round(v["cost"], 6),
                "avg_latency_ms": round(v["latency"] / v["n"], 2) if v["n"] else 0.0,
                "errors": int(v["errors"]),
            }
            for k, v in by_kind.items()
        },
    }
