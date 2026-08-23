#!/usr/bin/env python
"""Build a compact offline `label → [(qid, description)]` index for entity linking.

Wikidata's search API is unreachable in this environment (``*.wikidata.org`` is
blocked), so ``EntityLinker._search_candidates`` returns ``[]`` offline and
disambiguation degrades to a first-fuzzy-cache-match. This script reconstructs a
*description* for every QID we already have in the local KG subgraph cache
(``kg_subgraph_cache.jsonl``), so the linker can score candidates against
retrieved-passage context without any network.

Output ``indexes/entity_desc_index.json``:
    { "<label lower>": [{"qid": ..., "label": ..., "description": ...}, ...] }

Description is built from the entity's 1-hop triples restricted to a small set
of QA-relevant relations (``instance of``, ``occupation``, ``country``, ...),
joined as ``"relation: tail; relation: tail"``. It is a *surface* description
(no statements), sufficient for the linker's lexical passage-support scoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from kgproweight.retrieval.bootstrap import resolve_entity_cache_path, resolve_kg_cache_dir
from kgproweight.utils.logging import configure_logging, get_logger
from kgproweight.utils.paths import index_dir

configure_logging("INFO")
logger = get_logger(__name__)

# Relations whose tail is informative enough to appear in a surface description.
_DESC_RELATIONS = {
    "instance of", "subclass of", "occupation", "country of citizenship",
    "country", "genre", "position held", "field of work", "developer",
    "publisher", "creator", "director", "screenwriter", "author", "cast member",
    "located in the administrative territorial entity", "headquarters location",
    "founded by", "owned by", "manufacturer", "member of sports team",
    "member of political party", "award received", "educated at", "employer",
}
_MAX_DESC_ITEMS = 12


def _load_entity_cache(path: str):
    """Return (label_lower -> set(qid), qid -> primary label)."""
    label_to_qids: Dict[str, set] = defaultdict(set)
    qid_to_label: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = (obj.get("label") or "").strip()
            qid = (obj.get("qid") or "").strip()
            if not label or not qid:
                continue
            # Normalise identically to entity_linker._clean so lookups match.
            label_to_qids[" ".join(label.lower().split())].add(qid)
            qid_to_label.setdefault(qid, label)
    return label_to_qids, qid_to_label


def _build_description(triples, primary_label: str) -> str:
    label_l = primary_label.strip().lower()
    parts: List[str] = []
    seen = set()
    for h, r, t in triples:
        if not isinstance(t, str) or not isinstance(r, str):
            continue
        # 1-hop only: the triple's head must be the entity itself.
        if str(h).strip().lower() != label_l:
            continue
        rl = r.strip().lower()
        if rl not in _DESC_RELATIONS:
            continue
        if t.strip().lower() == label_l:
            continue
        item = f"{rl}: {t.strip()}"
        if item in seen:
            continue
        seen.add(item)
        parts.append(item)
        if len(parts) >= _MAX_DESC_ITEMS:
            break
    return "; ".join(parts)


def main() -> None:
    entity_cache_path = resolve_entity_cache_path()
    subgraph_cache_path = Path(resolve_kg_cache_dir()) / "kg_subgraph_cache.jsonl"
    out_path = Path(index_dir()) / "entity_desc_index.json"

    logger.info("Loading entity cache: %s", entity_cache_path)
    label_to_qids, qid_to_label = _load_entity_cache(entity_cache_path)
    logger.info("  %d unique labels, %d QIDs", len(label_to_qids), len(qid_to_label))

    logger.info("Loading KG subgraph cache: %s", subgraph_cache_path)
    qid_desc: Dict[str, str] = {}
    if subgraph_cache_path.exists():
        with open(subgraph_cache_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = obj.get("key", "")
                qid = key.split("_", 1)[0]
                if not qid.startswith("Q"):
                    continue
                label = qid_to_label.get(qid)
                if not label:
                    continue
                desc = _build_description(obj.get("triples", []), label)
                if desc:
                    qid_desc[qid] = desc
    else:
        logger.warning("subgraph cache missing: %s", subgraph_cache_path)

    logger.info("  %d QIDs with a description", len(qid_desc))

    # label -> candidates (deduped by qid), with descriptions where available.
    index: Dict[str, List[dict]] = {}
    n_ambiguous_with_desc = 0
    for label_l, qids in label_to_qids.items():
        cands: List[dict] = []
        for qid in sorted(qids):
            cands.append({
                "qid": qid,
                "label": qid_to_label.get(qid, label_l),
                "description": qid_desc.get(qid, ""),
            })
        index[label_l] = cands
        if len(qids) >= 2 and any(c["description"] for c in cands):
            n_ambiguous_with_desc += 1

    out_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Wrote %d labels (incl. %d ambiguous with descriptions) → %s (%.1f MB)",
        len(index), n_ambiguous_with_desc, out_path,
        out_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
