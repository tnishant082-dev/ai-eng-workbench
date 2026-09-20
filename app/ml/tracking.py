"""Experiment tracking: MLflow if installed, else filesystem store with same surface."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class FilesystemExperimentStore:
    """MLflow-like API surface backed by experiments/ filesystem."""

    backend = "filesystem"

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, Any] | None = None

    def start_run(self, run_name: str | None = None) -> str:
        run_id = run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._active = {"run_id": run_id, "metrics": {}, "params": {}, "dir": run_dir}
        return run_id

    def log_param(self, key: str, value: Any) -> None:
        if not self._active:
            raise RuntimeError("no active run")
        self._active["params"][key] = value

    def log_metric(self, key: str, value: float) -> None:
        if not self._active:
            raise RuntimeError("no active run")
        self._active["metrics"][key] = value

    def log_artifact_json(self, name: str, payload: dict) -> Path:
        if not self._active:
            raise RuntimeError("no active run")
        path = self._active["dir"] / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def end_run(self) -> dict[str, Any]:
        if not self._active:
            raise RuntimeError("no active run")
        meta = {
            "run_id": self._active["run_id"],
            "params": self._active["params"],
            "metrics": self._active["metrics"],
            "backend": self.backend,
            "ended_at": datetime.now(UTC).isoformat(),
        }
        self.log_artifact_json("run.json", meta)
        out = meta
        self._active = None
        return out


class MLflowExperimentStore:
    backend = "mlflow"

    def __init__(self, tracking_uri: str | None = None):
        import mlflow

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._run = None

    def start_run(self, run_name: str | None = None) -> str:
        self._run = self.mlflow.start_run(run_name=run_name)
        return self._run.info.run_id

    def log_param(self, key: str, value: Any) -> None:
        self.mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float) -> None:
        self.mlflow.log_metric(key, value)

    def log_artifact_json(self, name: str, payload: dict) -> Path:
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / name
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.mlflow.log_artifact(str(tmp))
        return tmp

    def end_run(self) -> dict[str, Any]:
        rid = self._run.info.run_id if self._run else None
        self.mlflow.end_run()
        self._run = None
        return {"run_id": rid, "backend": self.backend}


def get_experiment_store(root: Path, prefer_mlflow: bool = True):
    if prefer_mlflow:
        try:
            return MLflowExperimentStore(), "mlflow"
        except (ImportError, OSError, RuntimeError, ValueError):
            return FilesystemExperimentStore(root), "filesystem"
    return FilesystemExperimentStore(root), "filesystem"
