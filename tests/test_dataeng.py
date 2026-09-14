from app.core.config import get_settings
from app.dataeng.etl import run_etl
from app.dataeng.quality import corpus_quality


def test_etl_and_quality(tmp_path):
    settings = get_settings()
    cat = tmp_path / "catalog.json"
    lin = tmp_path / "lineage.json"
    out = run_etl(settings.corpus_dir, catalog_path=cat, lineage_path=lin, dataset_path=settings.dataset_path)
    assert out["chunks"] > 0
    assert cat.exists() and lin.exists()
    assert out["quality"]["corpus"]["ok"]
    assert corpus_quality(settings.corpus_dir)["docs"] > 0
