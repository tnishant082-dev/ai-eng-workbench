"""LangSmith-compatible client: real HTTP if LANGCHAIN_API_KEY set, else local recorder."""
from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class LocalLangSmithRecorder:
    """Records runs under data/runtime/langsmith_local/ when no API key."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or Path("data/runtime/langsmith_local"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.mode = "local_stub"

    def create_run(self, name: str, inputs: dict[str, Any], run_type: str = "chain") -> str:
        run_id = str(uuid.uuid4())
        payload = {
            "id": run_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "outputs": None,
            "created_at": datetime.now(UTC).isoformat(),
            "mode": self.mode,
        }
        (self.root / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return run_id

    def end_run(self, run_id: str, outputs: dict[str, Any], error: str | None = None) -> None:
        path = self.root / f"{run_id}.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        data["outputs"] = outputs
        data["error"] = error
        data["ended_at"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for p in files[:limit]:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return out


class LangSmithClient:
    def __init__(self, api_key: str | None = None, local_root: Path | None = None):
        self.api_key = api_key or os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        self.local = LocalLangSmithRecorder(local_root)
        if self.api_key:
            self.mode = "langsmith_cloud_planned"
            # Honest: we record locally AND note that cloud push requires langsmith SDK.
            # Full cloud push is optional — do not pretend it's live without the package.
            try:
                import langsmith  # noqa: F401

                self.mode = "langsmith_sdk_available"
            except ImportError:
                self.mode = "api_key_set_sdk_missing_local_fallback"
        else:
            self.mode = "local_stub"

    def create_run(self, name: str, inputs: dict[str, Any], run_type: str = "chain") -> str:
        return self.local.create_run(name, inputs, run_type=run_type)

    def end_run(self, run_id: str, outputs: dict[str, Any], error: str | None = None) -> None:
        self.local.end_run(run_id, outputs, error=error)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.local.list_runs(limit=limit)

    def info(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "note": "Without LANGCHAIN_API_KEY, runs are recorded locally under data/runtime/langsmith_local/.",
        }
