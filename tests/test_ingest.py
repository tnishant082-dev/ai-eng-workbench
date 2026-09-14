from app.core.config import get_settings
from app.dataeng.etl import run_etl
from app.dataeng.validate import validate_corpus_frames
from app.rag.index import build_index


def test_build_index():
    settings = get_settings()
    idx = build_index(settings.corpus_dir)
    assert idx.ready
    assert idx.doc_count >= 20
    assert len(idx.chunks) >= 20


def test_etl_and_validate(tmp_path):
    settings = get_settings()
    catalog = tmp_path / "catalog.json"
    lineage = tmp_path / "lineage.json"
    out = run_etl(settings.corpus_dir, catalog_path=catalog, lineage_path=lineage)
    assert out["chunks"] > 0
    assert catalog.exists() and lineage.exists()
    v = validate_corpus_frames(settings.corpus_dir)
    assert v["ok"] or v["corpus_docs"] > 0
