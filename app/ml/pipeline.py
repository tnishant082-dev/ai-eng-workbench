from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.tracking import get_experiment_store

FEATURE_COLS = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "contract"]
TARGET = "churn"
NUM = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
CAT = ["contract"]


def _load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_dataset(dataset_path: Path, version: str | None = None) -> Path:
    if version:
        cand = Path("data/datasets/versions") / version / "churn_demo.csv"
        if cand.exists():
            return cand
    return Path(dataset_path)


def train_model(dataset_path: Path, experiments_dir: Path, registry_dir: Path, *, tune: bool = True, version: str | None = None) -> dict[str, Any]:
    """Demo-scale sklearn churn classifier. Honest: tiny synthetic CSV, not production."""
    experiments_dir = Path(experiments_dir)
    registry_dir = Path(registry_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_dataset(dataset_path, version=version)
    df = _load_df(data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )
    clf = Pipeline(
        [
            ("pre", pre),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    store, tracking_backend = get_experiment_store(experiments_dir)
    run_id = store.start_run()
    store.log_param("dataset", str(data_path))
    store.log_param("tune", tune)
    best_params = {}
    if tune:
        grid = GridSearchCV(clf, {"model__C": [0.25, 1.0, 2.0], "model__class_weight": ["balanced", None]}, cv=3, scoring="f1", n_jobs=1)
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_
        best_params = {k: (v if v is not None else "None") for k, v in grid.best_params_.items()}
    else:
        clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scale": "demo",
        "best_params": best_params,
        "tracking_backend": tracking_backend,
    }
    run_dir = experiments_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.joblib"
    joblib.dump(clf, model_path)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    store.log_artifact_json("metrics.json", metrics)
    store.end_run()

    # registry pointer
    meta = {
        "run_id": run_id,
        "model_path": str(model_path),
        "metrics": metrics,
        "features": FEATURE_COLS,
        "created_at": datetime.now(UTC).isoformat(),
        "note": "Demo-scale logistic regression on synthetic churn CSV — not a production model.",
    }
    (registry_dir / "current.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (registry_dir / f"{run_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _current_model(registry_dir: Path):
    meta_path = Path(registry_dir) / "current.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = joblib.load(meta["model_path"])
    return model, meta


def predict(registry_dir: Path, records: list[dict[str, Any]], mode: str = "batch") -> dict[str, Any]:
    model, meta = _current_model(registry_dir)
    if model is None:
        raise RuntimeError("No model in registry — run train first")
    df = pd.DataFrame(records)
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"missing feature: {col}")
    proba = model.predict_proba(df[FEATURE_COLS])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "run_id": meta["run_id"],
        "mode": mode,
        "predictions": [
            {"churn_probability": round(float(p), 4), "churn_pred": int(y)}
            for p, y in zip(proba, pred)
        ],
        "note": "Demo-scale online/batch predict",
    }


def list_registry(registry_dir: Path) -> list[dict[str, Any]]:
    registry_dir = Path(registry_dir)
    out = []
    for p in sorted(registry_dir.glob("*.json")):
        if p.name == "current.json":
            continue
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index — demo drift signal."""
    quantiles = np.linspace(0, 1, buckets + 1)
    breaks = np.unique(np.quantile(expected, quantiles))
    if len(breaks) < 3:
        return 0.0
    exp_counts = np.histogram(expected, bins=breaks)[0].astype(float)
    act_counts = np.histogram(actual, bins=breaks)[0].astype(float)
    exp_perc = np.clip(exp_counts / max(exp_counts.sum(), 1), 1e-4, None)
    act_perc = np.clip(act_counts / max(act_counts.sum(), 1), 1e-4, None)
    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def check_drift(dataset_path: Path, sample: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compare a batch to training distribution (PSI + mean shift). Demo-scale."""
    df = _load_df(dataset_path)
    if sample:
        cur = pd.DataFrame(sample)
    else:
        # holdout-like: last 40 rows as "production" batch
        cur = df.tail(40)
        df = df.iloc[:-40]
    report = {"scale": "demo", "features": {}}
    alerts = []
    for col in NUM:
        base = df[col].astype(float).values
        now = cur[col].astype(float).values
        psi = round(_psi(base, now), 4)
        mean_shift = round(float(now.mean() - base.mean()) / (base.std() + 1e-9), 4)
        report["features"][col] = {"psi": psi, "mean_shift": mean_shift}
        if psi > 0.2 or abs(mean_shift) > 0.5:
            alerts.append(col)
    report["alerts"] = alerts
    report["status"] = "drift_suspected" if alerts else "stable"
    return report
