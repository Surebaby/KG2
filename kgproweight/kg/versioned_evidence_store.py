"""Read-only entity and property store derived from a frozen training partition.

The store is deliberately small and dataset-version specific.  It is not a
replacement for a complete Wikidata dump: aliases and edges are present only
when their QIDs occurred in the allowed source partition recorded by the store
manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from kgproweight.kg.entity_linker import LinkCandidate, LinkResult


STORE_SCHEMA_VERSION = "versioned-2wiki-evidence-store-1"


def normalize_alias(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text.strip().casefold())


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class VersionedEvidenceStore:
    """Resolve exact aliases and retrieve exact ``(QID, PID)`` edges."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "store_manifest.json"
        aliases_path = self.root / "aliases.jsonl"
        edges_path = self.root / "edges.jsonl"
        for path in (manifest_path, aliases_path, edges_path):
            if not path.is_file():
                raise FileNotFoundError(f"versioned evidence store asset missing: {path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported store schema: {self.manifest.get('schema_version')!r}"
            )
        self._aliases: Dict[str, List[Dict[str, Any]]] = {
            str(row["normalized_alias"]): list(row.get("candidates") or [])
            for row in _read_jsonl(aliases_path)
        }
        self._edges: Dict[str, List[Dict[str, Any]]] = {
            str(row["key"]): list(row.get("edges") or [])
            for row in _read_jsonl(edges_path)
        }
        label_votes: Dict[str, Dict[str, int]] = {}
        for candidates in self._aliases.values():
            for candidate in candidates:
                qid = str(candidate.get("qid") or "")
                label = str(candidate.get("label") or "")
                if qid and label:
                    label_votes.setdefault(qid, {})[label] = (
                        label_votes.setdefault(qid, {}).get(label, 0)
                        + int(candidate.get("evidence_count") or 0)
                    )
        self._qid_labels = {
            qid: sorted(votes.items(), key=lambda value: (-value[1], value[0]))[0][0]
            for qid, votes in label_votes.items()
        }

    @staticmethod
    def edge_key(qid: str, pid: str) -> str:
        return f"{str(qid).strip().upper()}::{str(pid).strip().upper()}"

    def resolve(self, surface: str) -> LinkResult:
        candidates = self._aliases.get(normalize_alias(surface), [])
        by_qid: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            qid = str(candidate.get("qid") or "")
            if qid:
                by_qid[qid] = candidate
        rendered = [
            LinkCandidate(
                qid=qid,
                label=str(candidate.get("label") or surface),
                description="2Wiki official-ID training-partition alias",
                score=1.0,
            )
            for qid, candidate in sorted(by_qid.items())
        ]
        if len(rendered) != 1:
            reason = "versioned store alias miss" if not rendered else "versioned store alias ambiguous"
            return LinkResult(
                mention=surface,
                abstained=True,
                abstain_reason=reason,
                candidates=rendered,
            )
        selected = rendered[0]
        return LinkResult(
            mention=surface,
            selected_qid=selected.qid,
            selected_label=selected.label,
            description=selected.description,
            score=1.0,
            margin=1.0,
            candidates=rendered,
        )

    def fetch_edges(self, qid: str, pids: Sequence[str]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for pid in pids:
            result.extend(self._edges.get(self.edge_key(qid, pid), []))
        return result

    def label_for_qid(self, qid: str) -> str | None:
        return self._qid_labels.get(str(qid).strip().upper())
