from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.agents.memory import ShortTermMemory
from app.agents.roles import critic, executor, planner, reflect, research
from app.agents.tools import pick_tools, validate_tool_call
from app.llm.base import LLMProvider, estimate_tokens
from app.llm.prompts import load_prompt
from app.rag.index import CorpusIndex


class MultiAgentWorkflow:
    """planner → research → critic → executor with optional reflection loop + approval gate."""

    def __init__(
        self,
        index: CorpusIndex,
        provider: LLMProvider,
        prompts_dir: Path,
        require_human_approval: bool = False,
    ):
        self.index = index
        self.provider = provider
        self.prompts_dir = prompts_dir
        self.require_human_approval = require_human_approval

    def run(self, task: str, max_steps: int = 6, approve: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        events: list[dict[str, Any]] = []
        memory = ShortTermMemory()
        citations: list[dict[str, Any]] = []
        tools_used: list[str] = []
        failures: list[str] = []

        # PLAN
        plan_out = planner(task, memory)
        events.append({"type": "role", "role": "planner", "content": plan_out.content, "data": plan_out.data})
        for step in plan_out.data.get("steps") or []:
            for tname in step.get("tools") or []:
                ok, msg = validate_tool_call(tname, {"query": task, "expression": "1+1"})
                events.append({"type": "tool_validation", "tool": tname, "ok": ok, "detail": msg})

        if self.require_human_approval and not approve:
            latency = (time.perf_counter() - t0) * 1000
            return {
                "status": "pending_approval",
                "answer": "Plan ready — awaiting human approval before research/execute.",
                "plan": plan_out.data,
                "tools_used": [],
                "events": events,
                "citations": [],
                "model": "n/a",
                "provider": self.provider.name,
                "gateway": self.provider.name,
                "latency_ms": round(latency, 2),
                "tokens_estimate": 0,
                "cost_usd": 0.0,
                "memory": memory.items,
                "failures": failures,
            }

        # RESEARCH
        research_out = research(task, self.index, memory)
        events.append({"type": "role", "role": "research", "content": research_out.content})
        tools_used.append("corpus_search")
        for hit in (research_out.data.get("result") or {}).get("hits") or []:
            citations.append(hit)

        # EXECUTE
        tools = pick_tools(task)
        exec_tools = [t for t in tools if t != "corpus_search"] or tools
        exec_out = executor(task, self.index, memory, exec_tools)
        events.append({"type": "role", "role": "executor", "content": exec_out.content, "data": exec_out.data})
        for tr in exec_out.data.get("tool_results") or []:
            if tr["tool"] not in tools_used:
                tools_used.append(tr["tool"])

        # CRITIC + REFLECTION LOOP
        critic_out = critic(task, memory)
        events.append({"type": "role", "role": "critic", "content": critic_out.content, "data": critic_out.data})
        ref_out = reflect(task, memory, critic_out)
        events.append({"type": "role", "role": "reflect", "content": ref_out.content, "data": ref_out.data})

        if ref_out.data.get("improve"):
            # one reflection retry: narrow research
            narrow = f"{task} key evidence"
            research_out2 = research(narrow, self.index, memory)
            events.append({"type": "role", "role": "research_retry", "content": research_out2.content})
            for hit in (research_out2.data.get("result") or {}).get("hits") or []:
                if hit not in citations:
                    citations.append(hit)
            critic_out = critic(task, memory)
            events.append({"type": "role", "role": "critic_retry", "content": critic_out.content, "data": critic_out.data})

        system = load_prompt(self.prompts_dir, "agent_system_v1.txt") or (
            "You are a multi-agent synthesizer. Summarize tool and research findings."
        )

        def mock_builder() -> str:
            parts = [
                f"**Task:** {task.strip()}",
                "",
                "**Plan:** " + plan_out.content,
                "",
                "**Research:**",
                research_out.content,
                "",
                "**Executor:**",
                exec_out.content,
                "",
                "**Critic:** " + critic_out.content,
                "**Reflection:** " + ref_out.content,
                "",
            ]
            if citations:
                parts.append("**Citations:**")
                for c in citations[:4]:
                    parts.append(f"- [{c.get('doc_id')}] {c.get('snippet', '')[:120]}")
            parts.append("")
            parts.append("_Multi-agent workflow finished (mock provider)._")
            return "\n".join(parts)

        user = f"Task: {task}\n\nMemory:\n{memory.as_text()}"
        events.append({"type": "llm_call_start", "provider": self.provider.name})
        result = self.provider.complete(system=system, user=user, mock_builder=mock_builder)
        events.append(
            {
                "type": "llm_call_end",
                "model": result.model,
                "provider": result.provider,
                "tokens": result.tokens_estimate,
                "cost_usd": result.cost_usd,
            }
        )

        latency = (time.perf_counter() - t0) * 1000
        return {
            "status": "ok" if not failures else "ok_with_recovery",
            "answer": result.text,
            "plan": plan_out.data,
            "tools_used": tools_used,
            "events": events,
            "citations": citations,
            "model": result.model,
            "provider": result.provider,
            "gateway": result.provider,
            "latency_ms": round(latency, 2),
            "tokens_estimate": result.tokens_estimate or estimate_tokens(system + user + result.text),
            "cost_usd": result.cost_usd,
            "memory": memory.items,
            "failures": failures,
        }
