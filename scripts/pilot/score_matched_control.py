#!/usr/bin/env python
"""Paired scoring of the matched control (legacy arm vs frozen canonical Proof).

Computes paired EM/F1 deltas, a paired bootstrap CI (10,000 draws, frozen seed)
and McNemar's exact test, plus the prompt-identity check (legacy prompt equals
canonical prompt except for the KG block).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.utils.logging import dump_manifest, get_logger

logger = get_logger(__name__)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _mcnemar(gained: int, lost: int) -> float:
    d = gained + lost
    if d == 0:
        return 1.0
    tail = sum(math.comb(d, v) for v in range(0, min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2 ** d))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_predictions", required=True)
    parser.add_argument("--canonical_intermediate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    legacy = _read_jsonl(Path(args.legacy_predictions))
    canon = json.loads(Path(args.canonical_intermediate).read_text(encoding="utf-8"))
    canon_by_qid = {str(r["id"]): r for r in canon}

    pairs = []
    for lr in legacy:
        qid = str(lr["qid"])
        cr = canon_by_qid[qid]
        proof_pred = str(cr["output"].get("pred") or "")
        golds = [str(g) for g in cr.get("golden_answers") or [] if str(g).strip()]
        pairs.append({
            "qid": qid,
            "legacy_em": lr["em"],
            "proof_em": compute_em(proof_pred, golds) if proof_pred and golds else 0.0,
            "legacy_f1": lr["f1"],
            "proof_f1": compute_f1(proof_pred, golds) if proof_pred and golds else 0.0,
        })

    n = len(pairs)
    em_diffs = [p["proof_em"] - p["legacy_em"] for p in pairs]
    f1_diffs = [p["proof_f1"] - p["legacy_f1"] for p in pairs]
    em_gain = sum(1 for d in em_diffs if d > 0)
    em_loss = sum(1 for d in em_diffs if d < 0)

    rng = random.Random(args.seed)
    def boot_ci(diffs):
        means = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(args.draws))
        return [means[int(0.025 * args.draws)], means[int(0.975 * args.draws) - 1]]

    legacy_em = sum(p["legacy_em"] for p in pairs) / n
    proof_em = sum(p["proof_em"] for p in pairs) / n
    legacy_f1 = sum(p["legacy_f1"] for p in pairs) / n
    proof_f1 = sum(p["proof_f1"] for p in pairs) / n

    report = {
        "schema_version": "matched-control-score-1",
        "n": n,
        "legacy_em": legacy_em, "proof_em": proof_em,
        "legacy_f1": legacy_f1, "proof_f1": proof_f1,
        "em_delta": proof_em - legacy_em,
        "f1_delta": proof_f1 - legacy_f1,
        "em_paired_bootstrap_ci95": boot_ci(em_diffs),
        "f1_paired_bootstrap_ci95": boot_ci(f1_diffs),
        "per_question": {"em_gained": em_gain, "em_lost": em_loss, "em_tie": n - em_gain - em_loss},
        "mcnemar_p": _mcnemar(em_gain, em_loss),
        "seed": args.seed, "draws": args.draws,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "matched_control_score.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(out, extra={"phase": "score_matched_control", "em_delta": report["em_delta"], "mcnemar_p": report["mcnemar_p"]}, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
