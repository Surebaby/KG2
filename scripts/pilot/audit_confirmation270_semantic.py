#!/usr/bin/env python
"""Mechanical semantic audit of the frozen confirmation270 audit sample.

Checks (all gold-free, computed from the frozen ProofKG + question + gold label):
- gold_answer_visible_in_kg_tail: gold answer string appears as a KG tail;
- head_entity_in_question: each KG head surface appears in the question;
- chain_connected: the triples form a contiguous head/tail chain (tail_i == head_{i+1}).

These are proxies, NOT a proof of semantic correctness (PID intent, direction and
name collision still need LLM/manual judgment). Gold is used only to EVALUATE.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = ROOT / "outputs/audits/2wiki_confirmation270_semantic_audit/audit_sample.jsonl"
DEFAULT_HIST = sorted((ROOT / "outputs/sft_quota70_baseline_eval/2wikimultihopqa/seed_42").glob("*_kg_proweight"))[0] / "intermediate_data.json"


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--hist", type=Path, default=DEFAULT_HIST)
    args = parser.parse_args()

    audit = [json.loads(l) for l in args.audit.read_text(encoding="utf-8").splitlines() if l.strip()]
    hist = json.loads(args.hist.read_text(encoding="utf-8"))
    gold_by_qid = {str(r["id"]): r.get("golden_answers") or [] for r in hist}

    rows: List[Dict[str, Any]] = []
    for a in audit:
        qid = str(a["qid"])
        kg = a.get("kg_subgraph") or []
        golds = gold_by_qid.get(qid, [])
        question = a["question"]
        tails = [str(t[2]) for t in kg if len(t) == 3]
        heads = [str(t[0]) for t in kg if len(t) == 3]
        # gold visibility: any gold answer normalised appears in any tail
        gold_visible = any(_norm(g) and _norm(g) in _norm(t) for g in golds for t in tails)
        # head in question
        head_in_q = all(_norm(h) and _norm(h) in _norm(question) for h in heads) if heads else False
        # chain connected: consecutive triples share head/tail
        connected = True
        for i in range(len(kg) - 1):
            if _norm(kg[i][2]) != _norm(kg[i + 1][0]):
                connected = False
                break
        rows.append({
            "qid": qid,
            "question": question,
            "gold_answers": golds,
            "kg_subgraph": kg,
            "gold_visible": gold_visible,
            "head_in_question": head_in_q,
            "chain_connected": connected,
            "n_triples": len(kg),
        })

    n = len(rows)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MECHANICAL_SEMANTIC_AUDIT",
        "n": n,
        "rates": {
            "gold_answer_visible_in_kg_tail": sum(r["gold_visible"] for r in rows) / n,
            "head_entity_in_question": sum(r["head_in_question"] for r in rows) / n,
            "chain_connected": sum(r["chain_connected"] for r in rows) / n,
        },
        "n_triples_histogram": dict(Counter(r["n_triples"] for r in rows)),
        "examples": rows,
        "note": "mechanical proxies only; PID intent, relation direction, and entity name collision require LLM/manual judgment.",
    }
    out = ROOT / "outputs/audits/2wiki_confirmation270_semantic_audit/mechanical_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["rates"], ensure_ascii=False, indent=2))
    print("n_triples_histogram:", report["n_triples_histogram"])


if __name__ == "__main__":
    main()
