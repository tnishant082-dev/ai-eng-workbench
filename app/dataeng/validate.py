from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_corpus_frames(corpus_dir: Path) -> dict[str, Any]:
    """Lightweight validation (pandera-style checks without hard dependency)."""
    files = sorted(Path(corpus_dir).rglob("*.md"))
    issues = []
    rows = []
    for p in files:
        text = p.read_text(encoding="utf-8")
        rows.append({"path": str(p.name), "chars": len(text), "has_heading": text.lstrip().startswith("#")})
        if len(text) < 80:
            issues.append({"path": p.name, "rule": "min_length", "detail": "doc shorter than 80 chars"})
        if not text.lstrip().startswith("#"):
            issues.append({"path": p.name, "rule": "heading_required", "detail": "missing H1"})
    df = pd.DataFrame(rows)
    # churn dataset optional check
    churn = Path(corpus_dir).parents[0] / "datasets" / "churn_demo.csv"
    churn_report = None
    if churn.exists():
        cdf = pd.read_csv(churn)
        required = {"tenure_months", "monthly_charges", "total_charges", "support_tickets", "contract", "churn"}
        missing = sorted(required - set(cdf.columns))
        nulls = {c: int(cdf[c].isna().sum()) for c in cdf.columns}
        churn_report = {
            "rows": len(cdf),
            "missing_columns": missing,
            "null_counts": nulls,
            "ok": not missing and all(v == 0 for v in nulls.values()),
        }
        if missing:
            issues.append({"path": str(churn), "rule": "schema", "detail": f"missing {missing}"})
    return {
        "corpus_docs": len(files),
        "issues": issues,
        "ok": len(issues) == 0,
        "churn": churn_report,
        "summary": {
            "avg_chars": float(df["chars"].mean()) if len(df) else 0,
            "with_heading": int(df["has_heading"].sum()) if len(df) else 0,
        },
    }
