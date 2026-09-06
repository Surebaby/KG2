#!/usr/bin/env python
"""Hotpot relation-graph v2 offline anchor-resolution diagnostic.

Resolves each planner anchor with the passage-title supplement (no property
fetch, no gold).  Reports resolved count, passage-title fallback count, abstain
count, and the source distribution.  nonempty/complete remain "waiting for
prefetch" — this audit only measures the linking-coverage ceiling.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.kg.anchor_resolver import resolve_anchor
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.kg.wikipedia_title_resolver import WikipediaTitleResolver
from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--proof_input", required=True)
    parser.add_argument("--entity_index", required=True)
    parser.add_argument("--entity_cache", required=True)
    parser.add_argument("--title_cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plans = _read_jsonl(Path(args.plans))
    pi_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(args.proof_input))}
    linker = EntityLinker(cache_path=args.entity_cache, offline=True, entity_index_path=args.entity_index)
    title_resolver = WikipediaTitleResolver(cache_path=args.title_cache, offline=True)

    rows = []
    source_counts = Counter()
    resolved = fallback = abstain = 0
    for r in plans:
        qid = str(r["qid"])
        pi = pi_by_qid[qid]
        passages = pi.get("retrieved_passages") or []
        predicted = r.get("predicted_target") or {}
        anchors = predicted.get("anchors") or []
        for anchor in anchors:
            result, source = resolve_anchor(
                anchor, pi["question"], passages, title_resolver, linker,
            )
            source_counts[source] += 1
            if source == "planner_anchor":
                resolved += 1
            elif source == "passage_title_fallback":
                fallback += 1
                resolved += 1
            else:
                abstain += 1
            rows.append({"qid": qid, "anchor": anchor, "source": source,
                         "qid_resolved": result.selected_qid if result else None})

    total = resolved + abstain
    report = {
        "schema_version": "hotpot-anchor-resolution-audit-1",
        "n_anchors": total,
        "anchor_qid_resolved": resolved,
        "passage_title_new_resolutions": fallback,
        "abstain": abstain,
        "source_distribution": dict(source_counts),
        "rate": {"resolved": resolved / total if total else 0.0, "passage_fallback": fallback / total if total else 0.0},
        "nonempty_complete": "WAITING_FOR_PREFETCH (this audit is linking-coverage only)",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "anchor_resolution_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "anchor_resolution_details.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    dump_manifest(out, extra={"phase": "audit_hotpot_anchor_resolution", **report}, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
