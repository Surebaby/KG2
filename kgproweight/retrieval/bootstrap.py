"""Path resolution helpers for the corpus, dense, and sparse indices."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from kgproweight.utils.paths import index_dir, project_root


def resolve_corpus_path(explicit: Optional[str] = None) -> str:
    """Resolution order: argument → env ``KGPW_CORPUS_PATH`` → ``$INDEX_DIR/corpus_flashrag.jsonl``."""
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get("KGPW_CORPUS_PATH")
    if env:
        return str(Path(env).expanduser())
    return str(index_dir() / "corpus_flashrag.jsonl")


def resolve_dense_index_path(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get("KGPW_DENSE_INDEX_PATH")
    if env:
        return str(Path(env).expanduser())
    return str(index_dir() / "e5_Flat.index")


def resolve_bm25_index_path(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get("KGPW_BM25_INDEX_PATH")
    if env:
        return str(Path(env).expanduser())
    return str(index_dir() / "bm25")


def resolve_kg_cache_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get("KGPW_KG_CACHE_DIR")
    if env:
        return str(Path(env).expanduser())
    return str(index_dir() / "kg_cache")


def resolve_entity_cache_path(explicit: Optional[str] = None) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    env = os.environ.get("KGPW_ENTITY_CACHE_PATH")
    if env:
        return str(Path(env).expanduser())
    return str(index_dir() / "entity_cache.jsonl")


def project_relative(path: str) -> str:
    """Make ``path`` relative to the project root if possible."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(project_root()))
    except ValueError:
        return str(p)
