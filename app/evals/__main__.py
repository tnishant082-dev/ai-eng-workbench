from pathlib import Path
from __future__ import annotations

import json
import sys

from app.core.config import get_settings
from app.evals.runner import run_evals
from app.llm.factory import get_provider
from app.rag.index import build_index
from app.rag.service import RagService


def main() -> int:
    settings = get_settings()
    index = build_index(settings.corpus_dir, settings.chunk_size, settings.chunk_overlap)
    provider = get_provider(settings)
    rag = RagService(index, provider, settings.prompts_dir)
    out = run_evals(rag, settings.golden_path, reports_dir=Path("reports"))
    print(json.dumps(out["summary"], indent=2))
    failed = [c for c in out["cases"] if not c["passed"]]
    if failed:
        print(f"\nFailed {len(failed)} cases:", file=sys.stderr)
        for c in failed:
            print(f"  - {c['id']}: faith={c['faithfulness']} rel={c['relevance']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
