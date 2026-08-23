"""Data quality checks over corpus + tabular demo set."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def corpus_quality(corpus_dir: Path) -> dict[str, Any]:
    files = sorted(Path(corpus_dir).rglob("*.md"))
    empty = []
    short = []
    for p in files:
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            empty.append(p.name)
        elif len(text) < 80:
            short.append(p.name)
    return {
        "docs": len(files),
        "empty": empty,
        "short": short,
        "ok": not empty,
        "score": round(1.0 - (len(empty) + 0.5 * len(short)) / max(len(files), 1), 4),
    }


def tabular_quality(csv_path: Path, required_cols: list[str]) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    missing = [c for c in required_cols if c not in df.columns]
    null_frac = {c: round(float(df[c].isna().mean()), 4) for c in df.columns}
    dupes = int(df.duplicated().sum())
    return {
        "rows": int(len(df)),
        "missing_columns": missing,
        "null_fraction": null_frac,
        "duplicate_rows": dupes,
        "ok": not missing and dupes == 0,
    }
