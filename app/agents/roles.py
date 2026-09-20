from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.memory import ShortTermMemory
from app.agents.tools import (
    TOOL_SPECS,
    calculator,
    corpus_search,
    extract_math_expression,
    get_time,
    pick_tools,
)
from app.rag.index import CorpusIndex


@dataclass
class RoleOutput:
    role: str
    content: str
    data: dict[str, Any]


def planner(task: str, memory: ShortTermMemory) -> RoleOutput:
    tools = pick_tools(task)
    plan = {
        "goal": task.strip(),
        "steps": [
            {"id": 1, "action": "research", "tools": [t for t in tools if t == "corpus_search"]},
            {"id": 2, "action": "execute", "tools": [t for t in tools if t != "corpus_search"]},
            {"id": 3, "action": "critic", "tools": []},
            {"id": 4, "action": "reflect", "tools": []},
        ],
        "available_tools": TOOL_SPECS,
    }
    # ensure execute has something if only search
    if not plan["steps"][1]["tools"] and tools:
        plan["steps"][1]["tools"] = tools
    summary = f"Plan for: {task.strip()} → tools={tools}"
    memory.add("planner", summary, plan=plan)
    return RoleOutput("planner", summary, plan)


def research(task: str, index: CorpusIndex, memory: ShortTermMemory) -> RoleOutput:
    result = corpus_search(index, task, top_k=4)
    memory.add("research", result.detail, data=result.data)
    citations = (result.data or {}).get("hits") or []
    content = result.detail + "\n" + "\n".join(
        f"- [{c['doc_id']}] {c['snippet']}" for c in citations[:3]
    )
    return RoleOutput("research", content, {"tool": "corpus_search", "result": result.data, "ok": result.ok})


def executor(task: str, index: CorpusIndex, memory: ShortTermMemory, tools: list[str]) -> RoleOutput:
    notes = []
    data: dict[str, Any] = {"tool_results": []}
    for name in tools:
        if name == "calculator":
            expr = extract_math_expression(task) or "1+1"
            r = calculator(expr)
        elif name == "get_time":
            r = get_time()
        elif name == "corpus_search":
            r = corpus_search(index, task, top_k=3)
        else:
            continue
        notes.append(r.detail)
        data["tool_results"].append({"tool": name, "ok": r.ok, "detail": r.detail, "data": r.data})
        memory.add("executor", r.detail, tool=name)
    content = "Executed: " + "; ".join(notes) if notes else "No tools executed"
    return RoleOutput("executor", content, data)


def critic(task: str, memory: ShortTermMemory) -> RoleOutput:
    research_items = memory.by_role("research")
    exec_items = memory.by_role("executor")
    issues = []
    if not research_items:
        issues.append("missing research evidence")
    else:
        hits = ((research_items[0].get("data") or {}).get("hits")) or []
        if len(hits) < 1:
            issues.append("weak retrieval")
    if ("calculat" in task.lower() or extract_math_expression(task)) and not any(
        "calculator" in (i.get("content") or "") for i in exec_items
    ):
        issues.append("expected calculator result missing")
    verdict = "pass" if not issues else "revise"
    content = f"Critic verdict={verdict}; issues={issues or ['none']}"
    memory.add("critic", content, verdict=verdict, issues=issues)
    return RoleOutput("critic", content, {"verdict": verdict, "issues": issues})


def reflect(task: str, memory: ShortTermMemory, critic_out: RoleOutput) -> RoleOutput:
    if critic_out.data.get("verdict") == "pass":
        content = "Reflection: evidence and tools look sufficient — finalize answer."
        improve = False
    else:
        content = (
            "Reflection: gaps found ("
            + ", ".join(critic_out.data.get("issues") or [])
            + "). Re-run research with narrowed query."
        )
        improve = True
    memory.add("reflect", content, improve=improve)
    return RoleOutput("reflect", content, {"improve": improve})
