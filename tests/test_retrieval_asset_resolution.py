from pathlib import Path

from kgproweight.retrieval.bootstrap import (
    resolve_bm25_index_path,
    resolve_corpus_path,
    resolve_dense_index_path,
)


def test_default_retrieval_assets_resolve_to_existing_full_wiki18(monkeypatch):
    for name in ("KGPW_CORPUS_PATH", "KGPW_DENSE_INDEX_PATH", "KGPW_BM25_INDEX_PATH"):
        monkeypatch.delenv(name, raising=False)
    paths = [
        Path(resolve_corpus_path()),
        Path(resolve_dense_index_path()),
        Path(resolve_bm25_index_path()),
    ]
    assert all(path.exists() for path in paths)
    assert all("indexes_wiki18" in str(path) for path in paths)
