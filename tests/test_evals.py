from pathlib import Path
from app.core.config import get_settings
from app.evals.metrics import context_precision, context_recall, llm_judge_stub
from app.evals.runner import run_evals
from app.llm.mock import MockProvider
from app.rag.index import build_index
from app.rag.service import RagService


def test_run_evals_and_report(tmp_path):
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    rag = RagService(idx, MockProvider(), settings.prompts_dir)
    out = run_evals(rag, settings.golden_path, limit=5, reports_dir=tmp_path)
    assert out["summary"]["total"] == 5
    assert "avg_context_precision" in out["summary"]
    assert out.get("report_paths")
    assert Path(out["report_paths"]["markdown"]).exists()


def test_context_metrics_and_judge():
    assert context_precision(["alpha beta"], ["alpha gamma"]) > 0
    assert context_recall(["alpha beta"], ["alpha beta gamma"]) == 1.0
    j = llm_judge_stub("what is rag", "rag retrieves then generates", ["retrieval augmented"])
    assert j["judge"] == "heuristic_stub"
