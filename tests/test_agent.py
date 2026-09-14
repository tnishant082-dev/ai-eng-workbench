from app.agents.tools import calculator, extract_math_expression, pick_tools
from app.agents.workflow import MultiAgentWorkflow
from app.core.config import get_settings
from app.llm.mock import MockProvider
from app.rag.index import build_index


def test_calculator():
    r = calculator("12 * (3 + 4)")
    assert r.ok
    assert r.data["value"] == 84


def test_pick_tools():
    tools = pick_tools("Search corpus for evals then calculate 2+2")
    assert "corpus_search" in tools
    assert "calculator" in tools


def test_multi_agent_workflow():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    wf = MultiAgentWorkflow(idx, MockProvider(), settings.prompts_dir)
    out = wf.run("Search the corpus for faithfulness evals, then calculate 12 * (3 + 4).")
    assert out["status"] == "ok"
    assert "corpus_search" in out["tools_used"]
    assert "calculator" in out["tools_used"]
    roles = [e.get("role") for e in out["events"] if e.get("type") == "role"]
    assert "planner" in roles and "critic" in roles


def test_pending_approval():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    wf = MultiAgentWorkflow(idx, MockProvider(), settings.prompts_dir, require_human_approval=True)
    out = wf.run("Search corpus", approve=False)
    assert out["status"] == "pending_approval"
