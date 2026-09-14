from app.agents.tools import validate_tool_call, safe_call, pick_tools
from app.agents.workflow import MultiAgentWorkflow
from app.core.config import get_settings
from app.llm.mock import MockProvider
from app.rag.index import build_index


def test_tool_validation():
    ok, _ = validate_tool_call("calculator", {"expression": "1+1"})
    assert ok
    bad, msg = validate_tool_call("nope", {})
    assert not bad


def test_workflow_recovery_path():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    wf = MultiAgentWorkflow(idx, MockProvider(), settings.prompts_dir)
    out = wf.run("Search the corpus for faithfulness, then calculate 12 * (3 + 4).")
    assert out["status"] in ("ok", "ok_with_recovery")
    assert "calculator" in out["tools_used"] or "corpus_search" in out["tools_used"]
    assert out["events"]


def test_approval_gate():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    wf = MultiAgentWorkflow(idx, MockProvider(), settings.prompts_dir, require_human_approval=True)
    out = wf.run("Find docs about RAG", approve=False)
    assert out["status"] == "pending_approval"
