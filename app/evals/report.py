"""Auto HTML + Markdown eval report generator under reports/."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_eval_report(payload: dict[str, Any], reports_dir: Path) -> dict[str, str]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    summary = payload.get("summary") or {}
    cases = payload.get("cases") or []

    md_path = reports_dir / f"eval_report_{stamp}.md"
    html_path = reports_dir / f"eval_report_{stamp}.html"
    json_path = reports_dir / f"eval_report_{stamp}.json"

    lines = [
        f"# Eval Report ({stamp})",
        "",
        "## Summary",
        "",
        f"- total: **{summary.get('total', 0)}**",
        f"- pass_rate: **{summary.get('pass_rate', 0)}**",
        f"- avg_faithfulness: {summary.get('avg_faithfulness', 0)}",
        f"- avg_relevance: {summary.get('avg_relevance', 0)}",
        f"- avg_context_precision: {summary.get('avg_context_precision', 'n/a')}",
        f"- avg_context_recall: {summary.get('avg_context_recall', 'n/a')}",
        f"- avg_latency_ms: {summary.get('avg_latency_ms', 0)}",
        f"- total_cost_usd: {summary.get('total_cost_usd', 0)}",
        "",
        "## Cases",
        "",
        "| id | passed | faith | rel | ctx_p | ctx_r | latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        lines.append(
            f"| {c.get('id','')} | {c.get('passed')} | {c.get('faithfulness')} | "
            f"{c.get('relevance')} | {c.get('context_precision', '')} | "
            f"{c.get('context_recall', '')} | {c.get('latency_ms')} |"
        )
    lines.append("")
    lines.append("_Heuristic metrics on synthetic golden set — not production LLM-judge scores._")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    rows_html = "".join(
        f"<tr><td>{c.get('id','')}</td><td>{c.get('passed')}</td>"
        f"<td>{c.get('faithfulness')}</td><td>{c.get('relevance')}</td>"
        f"<td>{c.get('context_precision','')}</td><td>{c.get('context_recall','')}</td>"
        f"<td>{c.get('latency_ms')}</td></tr>"
        for c in cases
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Eval Report {stamp}</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; background:#0b1220; color:#e2e8f0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #334155; padding: 8px 10px; text-align: left; }}
th {{ background: #1e293b; }}
.card {{ background:#111827; border:1px solid #334155; border-radius:12px; padding:1rem; margin-bottom:1rem; }}
</style></head><body>
<h1>Eval Report <small>{stamp}</small></h1>
<div class="card">
<p>pass_rate=<b>{summary.get('pass_rate', 0)}</b> · faithfulness={summary.get('avg_faithfulness', 0)}
· relevance={summary.get('avg_relevance', 0)} · latency_ms={summary.get('avg_latency_ms', 0)}</p>
<p style="opacity:.7">Heuristic / local — not a hosted SaaS judge.</p>
</div>
<table><thead><tr>
<th>id</th><th>passed</th><th>faith</th><th>rel</th><th>ctx_p</th><th>ctx_r</th><th>latency</th>
</tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path), "json": str(json_path)}
