from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.rag.index import CorpusIndex

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


@dataclass
class ToolResult:
    ok: bool
    detail: str
    data: dict[str, Any] | None = None


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> ToolResult:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _eval_node(tree.body)
        return ToolResult(True, f"calculator({expression}) = {value}", {"value": value, "expression": expression})
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"calculator error: {exc}")


def get_time() -> ToolResult:
    now = datetime.now(UTC).isoformat()
    return ToolResult(True, f"utc_now={now}", {"utc": now})


def corpus_search(index: CorpusIndex, query: str, top_k: int = 3) -> ToolResult:
    hits = index.search(query, top_k=top_k)
    rows = [
        {
            "doc_id": h.chunk.doc_id,
            "title": h.chunk.title,
            "path": h.chunk.path,
            "score": round(h.score, 4),
            "snippet": " ".join(h.chunk.text.split())[:180],
        }
        for h in hits
    ]
    detail = f"corpus_search('{query}') → {len(rows)} hits"
    return ToolResult(True, detail, {"hits": rows})


def extract_math_expression(text: str) -> str | None:
    # Prefer explicit calculate phrases, then richest arithmetic span
    m = re.search(
        r"(?:calculat\w*|compute|eval)\s+([0-9\.+\-\*/\(\)\s%]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        expr = re.sub(r"\s+", "", m.group(1)).rstrip('.,;:')
        if re.search(r"[+\-*/]", expr):
            return expr
    candidates = re.findall(r"[0-9]+(?:\s*[+\-*/%]\s*[0-9\(\)]+)+", text)
    if not candidates:
        candidates = re.findall(r"[0-9\.\s\+\-\*/\(\)]{3,}", text)
    best = None
    for c in candidates:
        expr = re.sub(r"\s+", " ", c).strip()
        if re.search(r"[+\-*/]", expr) and re.search(r"\d", expr) and (
            best is None or len(expr) > len(best)
        ):
            best = expr
    if best:
        best = best.rstrip('.,;:')
    return best


TOOL_SPECS = [
    {"name": "calculator", "description": "Evaluate a basic arithmetic expression"},
    {"name": "corpus_search", "description": "Hybrid search over the local markdown corpus"},
    {"name": "get_time", "description": "Return current UTC timestamp"},
]


def validate_tool_call(name: str, args: dict | None = None) -> tuple[bool, str]:
    known = {x["name"] for x in TOOL_SPECS}
    if name not in known:
        return False, f"unknown tool: {name}"
    return True, "ok"


def safe_call(name: str, *, index=None, **kwargs):
    ok, msg = validate_tool_call(name, kwargs or {"_": True})
    if not ok and name not in {x["name"] for x in TOOL_SPECS}:
        return ToolResult(False, msg)
    try:
        if name == "calculator":
            return calculator(str(kwargs.get("expression", "1+1")))
        if name == "get_time":
            return get_time()
        if name == "corpus_search":
            if index is None:
                return ToolResult(False, "corpus_search requires index")
            return corpus_search(index, str(kwargs.get("query", "")), int(kwargs.get("top_k", 3)))
        return ToolResult(False, f"unhandled tool {name}")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, f"tool {name} failed: {exc}")


def pick_tools(task: str) -> list[str]:
    t = task.lower()
    planned: list[str] = []
    if any(k in t for k in ("search", "corpus", "find", "rag", "doc", "faithfulness", "prompt", "eval")):
        planned.append("corpus_search")
    if extract_math_expression(task) or any(k in t for k in ("calculat", "math", "sum", "multiply", "*", "+")):
        planned.append("calculator")
    if any(k in t for k in ("time", "utc", "clock", "date")):
        planned.append("get_time")
    if not planned:
        planned.append("corpus_search")
    return planned
