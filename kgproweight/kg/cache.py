"""Disk-backed caches for entities and Wikidata subgraphs.

Both caches are append-only JSONL: each line is one cache entry. Loading is
O(n) on disk size; lookups after load are O(1) in memory. Writers append in
a thread-safe manner using a global lock.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Entity → QID cache
# ---------------------------------------------------------------------------

class EntityCache:
    """In-memory mapping ``surface_label_lower -> QID`` with append-only persistence."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path).expanduser() if path else None
        self._cache: Dict[str, str] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    label = str(obj["label"]).strip().lower()
                    qid = str(obj["qid"]).strip()
                    if label and qid:
                        self._cache[label] = qid
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, label: str) -> Optional[str]:
        return self._cache.get(label.strip().lower())

    def set(self, label: str, qid: str, persist: bool = True) -> None:
        key = label.strip().lower()
        if not key or not qid:
            return
        self._cache[key] = qid
        if persist and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"label": label, "qid": qid}, ensure_ascii=False) + "\n")

    def items(self) -> Iterable[Tuple[str, str]]:
        return self._cache.items()

    def __contains__(self, label: str) -> bool:
        return label.strip().lower() in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# QID → triples cache
# ---------------------------------------------------------------------------

class SubgraphCache:
    """In-memory mapping ``cache_key -> list[(h, r, t)]`` with disk persistence.

    ``cache_key`` is usually ``f"{qid}_{max_hops}"`` so that 1-hop and 2-hop
    fetches share storage but distinct entries.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path).expanduser() if path else None
        self._cache: Dict[str, List[Tuple[str, str, str]]] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = obj["key"]
                    triples = [tuple(t) for t in obj["triples"] if len(t) == 3]
                    self._cache[key] = triples
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, key: str) -> Optional[List[Tuple[str, str, str]]]:
        return self._cache.get(key)

    def set(self, key: str, triples: List[Tuple[str, str, str]], persist: bool = True) -> None:
        self._cache[key] = triples
        if persist and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "triples": triples}, ensure_ascii=False) + "\n")

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)
