#!/usr/bin/env python
"""Audit the retrieval/KG solvability of the 'gold hidden' val questions.

For every replay-validation item whose gold is NOT verbatim in the passages,
look up the silver trajectory and report whether the gold (or any of its
surface forms) appears in the KG subgraph or in the retrieved passages, plus
the KG / passage counts.  This distinguishes "retrieval miss" from "KG coverage
gap" before anyone touches the reward.

    python scripts/pilot/audit_hidden_retrieval.py \
        --replay <ppo_smoke_*.jsonl> --silver <silver_with_logprobs.jsonl> \
        --out <audit.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--silver", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    replay = [json.loads(l) for l in Path(args.replay).read_text(encoding="utf-8").splitlines() if l.strip()]
    silver = {}
    for l in Path(args.silver).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        silver[r["qid"]] = r

    hidden = [r for r in replay if not r.get("gold_in_passages")]
    print(f"replay items: {len(replay)}, hidden (gold not in passages): {len(hidden)}", flush=True)

    rows = []
    for r in hidden:
        qid = r["qid"]
        s = silver.get(qid)
        golds = r.get("gold") or []
        gold = golds[0] if golds else (s or {}).get("metadata", {}).get("gold_answer", "")
        gn = _norm(gold)
        kg = [(tuple(t) if isinstance(t, list) and len(t) == 3 else t) for t in (s or {}).get("kg_subgraph", [])]
        passages = (s or {}).get("retrieved_passages", []) or []
        ptexts = [str(p.get("contents") or p.get("text") or "") for p in passages]

        gold_in_kg = any(gn and gn in _norm(" ".join(map(str, t))) for t in kg if isinstance(t, tuple))
        gold_in_passages = any(gn and gn in _norm(t) for t in ptexts)
        # also check per-triple-part (answer might be a tail of a triple)
        triple_parts_hit = [str(t) for t in kg if isinstance(t, tuple) and any(gn and gn == _norm(x) for x in t)]

        rows.append({
            "qid": qid,
            "question": r["question"],
            "gold": gold,
            "gold_in_kg": gold_in_kg,
            "gold_in_passages": gold_in_passages,
            "kg_nonempty": bool(kg),
            "n_kg_triples": len(kg),
            "n_passages": len(passages),
            "triple_parts_hit_gold": triple_parts_hit[:6],
            "em": r.get("em"),
            "passage_titles": [str(p.get("id") or p.get("source") or "")[:60] for p in passages[:5]],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"\n=== summary (n={n} hidden questions) ===")
    print(f"gold in KG        : {sum(1 for r in rows if r['gold_in_kg'])}/{n}")
    print(f"gold in passages  : {sum(1 for r in rows if r['gold_in_passages'])}/{n}")
    print(f"KG empty          : {sum(1 for r in rows if not r['kg_nonempty'])}/{n}")
    print(f"KG nonempty       : {sum(1 for r in rows if r['kg_nonempty'])}/{n}")
    print(f"KG part matches   : {sum(1 for r in rows if r['triple_parts_hit_gold'])}/{n}")
    print(f"EM among hidden   : {sum(1 for r in rows if r['em'] > 0)}/{n}")
    print(f"\nwrote {n} rows to {out}")


if __name__ == "__main__":
    main()
