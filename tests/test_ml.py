from app.core.config import get_settings
from app.ml.pipeline import train_model, predict, check_drift, list_registry


def test_train_predict_drift(tmp_path):
    settings = get_settings()
    exp, reg = tmp_path / "exp", tmp_path / "reg"
    meta = train_model(settings.dataset_path, exp, reg, tune=True, version="v1")
    assert meta["metrics"]["scale"] == "demo"
    out = predict(reg, [{"tenure_months": 12, "monthly_charges": 70.0, "total_charges": 800.0, "support_tickets": 1, "contract": "month-to-month"}], mode="online")
    assert out["predictions"]
    assert check_drift(settings.dataset_path)["status"] in ("stable", "drift_suspected")
    assert list_registry(reg)
