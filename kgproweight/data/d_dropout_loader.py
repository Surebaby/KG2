"""Dropout dataset (D_dropout).

Bug-fix #5 from :doc:`docs/refactor_notes`. Each item carries
``metadata.dropout.modified_kg``: a *severed* version of its original
2-hop subgraph (answer-path bridge triples replaced with random noise).
The inference pipeline reads this field instead of calling Wikidata, so
the gate's graceful-fallback behaviour is actually exercised.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass
class DropoutItem:
    qid: str
    question: str
    gold_answer: str
    original_kg: List[Tuple[str, str, str]] = field(default_factory=list)
    modified_kg: List[Tuple[str, str, str]] = field(default_factory=list)
    dropout_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DropoutItem":
        meta = d.get("metadata") or {}
        dropout = meta.get("dropout") or {}
        original = [tuple(t) for t in (dropout.get("original_kg") or [])]
        modified = [tuple(t) for t in (dropout.get("modified_kg") or [])]
        info = dict(dropout)
        info.pop("original_kg", None)
        info.pop("modified_kg", None)
        extras = {k: v for k, v in d.items() if k not in {"qid", "id", "question", "answer", "golden_answers", "metadata"}}
        return cls(
            qid=str(d.get("qid") or d.get("id") or ""),
            question=str(d.get("question", "")),
            gold_answer=str(d.get("answer") or (d.get("golden_answers") or [""])[0]),
            original_kg=original,
            modified_kg=modified,
            dropout_info=info,
            metadata=meta,
            extra=extras,
        )

    @property
    def effective_kg(self) -> List[Tuple[str, str, str]]:
        """Use ``modified_kg`` if present, else ``original_kg``."""
        return self.modified_kg or self.original_kg


@dataclass
class DropoutDataset:
    path: Path
    items: List[DropoutItem]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DropoutItem:
        return self.items[idx]

    def __iter__(self) -> Iterator[DropoutItem]:
        return iter(self.items)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "DropoutDataset":
        p = Path(path)
        items: List[DropoutItem] = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(DropoutItem.from_dict(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return cls(path=p, items=items)

    def to_flashrag_dataset(self) -> List[Dict[str, Any]]:
        """Render the items into the FlashRAG ``Dataset.data`` list-of-dicts shape.

        Each output dict already preserves the ``metadata.dropout`` block,
        which the KG-ProWeight pipeline then reads.
        """
        out: List[Dict[str, Any]] = []
        for it in self.items:
            d = copy.deepcopy(it.extra)
            d["id"] = it.qid
            d["question"] = it.question
            d["golden_answers"] = [it.gold_answer] if it.gold_answer else []
            metadata = copy.deepcopy(it.metadata) or {}
            dropout_block = dict(it.dropout_info)
            dropout_block["original_kg"] = [list(t) for t in it.original_kg]
            dropout_block["modified_kg"] = [list(t) for t in it.modified_kg]
            metadata["dropout"] = dropout_block
            d["metadata"] = metadata
            out.append(d)
        return out


def load_dropout_dataset(path: str | Path) -> DropoutDataset:
    return DropoutDataset.from_jsonl(path)
