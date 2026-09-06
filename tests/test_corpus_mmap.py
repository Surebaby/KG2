"""Full wiki18 corpus must be accessible without materialising 21M rows."""

from __future__ import annotations

import json


def test_load_corpus_mmap_decodes_rows_lazily(tmp_path, monkeypatch):
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"id": "0", "contents": "Title A\nText A"},
        {"id": "1", "contents": "Title B\nText B"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("KGPW_CORPUS_MMAP", "1")

    from flashrag.retriever.utils import load_corpus

    corpus = load_corpus(str(path))
    assert len(corpus) == 2
    assert corpus[0] == rows[0]
    assert corpus[1] == rows[1]
    assert (tmp_path / "corpus.mmindex.json").exists()
