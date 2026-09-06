#!/usr/bin/env python
"""Train the L0 scalar verifier head on frozen v2.1 features.

Frozen encoder (strong SFT) is never updated.  A 7->32->1 MLP is trained with a
pairwise logistic loss over correct/wrong pairs, 3 head seeds, final = average
logit.  Dev is used ONLY for early-stop + checkpoint selection; confirmation is
opened exactly once at the end.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from kgproweight.eval.metrics import compute_em
from kgproweight.reward.proofkg_process_v2 import build_execution_trace, score_proofkg_v2
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir, get_logger

logger = get_logger(__name__)

FEATURES = [
    "P_precise_citation", "H_hop_coverage", "O_dependency_order",
    "G_conclusion_grounding", "A_answer_consistency", "m_A_deterministic", "C",
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _rescore(cands: List[Dict[str, Any]], kg_by_qid, detail_by_qid, gold_by_qid) -> List[Dict[str, Any]]:
    trace = {}
    planned = {}
    for qid, kg in kg_by_qid.items():
        plan = kg.get("query_plan") or {}
        planned[qid] = len(plan.get("hops") or [])
        trace[qid] = build_execution_trace(plan, detail_by_qid.get(qid, {}).get("execution") or {})
    out = []
    for c in cands:
        qid = str(c["qid"])
        kg = kg_by_qid[qid]
        proc = score_proofkg_v2(
            question=str(kg["question"]), generation=str(c["generation"]),
            kg_triples=kg.get("kg_subgraph") or [], execution_trace=trace[qid], planned_hops=planned[qid],
        )
        golds = [str(g) for g in gold_by_qid.get(qid, []) if str(g).strip()]
        c["process"] = proc
        c["em"] = compute_em(proc["prediction"], golds) if proc["prediction"] and golds else 0.0
        out.append(c)
    return out


def _feats(c) -> List[float]:
    comp = c["process"].get("components") or {}
    return [float(comp.get(f, 0.0)) for f in FEATURES]


class ScalarHead(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(len(FEATURES), 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _pairs(cands: List[Dict[str, Any]]) -> List[Tuple[List[float], List[float]]]:
    by_qid = defaultdict(list)
    for c in cands:
        if c["candidate_type"] == "sampled":
            by_qid[str(c["qid"])].append(c)
    pairs = []
    for rows in by_qid.values():
        correct = [c for c in rows if c["em"] > 0.5]
        wrong = [c for c in rows if c["em"] <= 0.5]
        for a in correct:
            for b in wrong:
                pairs.append((_feats(a), _feats(b)))
    return pairs


def _eval_pool(cands, head: ScalarHead) -> Tuple[float, float, float]:
    by_qid = defaultdict(list)
    for c in cands:
        if c["candidate_type"] == "sampled":
            by_qid[str(c["qid"])].append(c)
    greedy = {str(c["qid"]): c for c in cands if c["candidate_type"] == "greedy"}
    # assign verifier scores
    scored = []
    for rows in by_qid.values():
        for c in rows:
            c["_vscore"] = float(head(torch.tensor([_feats(c)], dtype=torch.float32)).item())
    # pairwise accuracy
    wins = ties = comps = 0
    for rows in by_qid.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["em"] == b["em"]:
                    continue
                corr, wrong = (a, b) if a["em"] > b["em"] else (b, a)
                comps += 1
                if corr["_vscore"] > wrong["_vscore"]:
                    wins += 1
                elif corr["_vscore"] == wrong["_vscore"]:
                    ties += 1
    pairwise = (wins + 0.5 * ties) / comps if comps else 0.0
    # top1 EM vs greedy EM
    top1 = [max(rows, key=lambda c: c["_vscore"]) for rows in by_qid.values()]
    top1_em = sum(c["em"] for c in top1) / len(top1)
    greedy_em = sum(greedy[q]["em"] for q in greedy) / len(greedy)
    return pairwise, top1_em, greedy_em


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--question_kg", nargs="+", required=True)
    parser.add_argument("--runtime_details", nargs="+", required=True)
    parser.add_argument("--proof_input", nargs="+", required=True)
    parser.add_argument("--train_qids", required=True)
    parser.add_argument("--dev_qids", required=True)
    parser.add_argument("--confirmation_qids", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--head_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    # load + rescore all pools with reward v2.1
    all_cands = []
    for cp, kgp, rdp, pp in zip(args.candidates, args.question_kg, args.runtime_details, args.proof_input):
        cands = _read_jsonl(Path(cp))
        kg_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(kgp))}
        detail_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(rdp))}
        gold_by_qid = {str(r["qid"]): r.get("gold_answers") or [] for r in _read_jsonl(Path(pp))}
        all_cands.extend(_rescore(cands, kg_by_qid, detail_by_qid, gold_by_qid))

    train_qids = {str(r["qid"]) for r in _read_jsonl(Path(args.train_qids))}
    dev_qids = {str(r["qid"]) for r in _read_jsonl(Path(args.dev_qids))}
    conf_qids = {str(r["qid"]) for r in _read_jsonl(Path(args.confirmation_qids))}

    train_cands = [c for c in all_cands if str(c["qid"]) in train_qids]
    dev_cands = [c for c in all_cands if str(c["qid"]) in dev_qids]
    conf_cands = [c for c in all_cands if str(c["qid"]) in conf_qids]

    run_dir, experiment_id = prepare_new_run_dir(args.run_dir, experiment_id=args.experiment_id,
                                                 extra={"phase": "train_l0_verifier"})

    train_pairs = _pairs(train_cands)
    logger.info("train pairs=%d, dev qids=%d, conf qids=%d", len(train_pairs), len(dev_cands) // 5, len(conf_cands) // 5)

    heads = []
    results = []
    for seed in args.head_seeds:
        head = ScalarHead(seed)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        best = {"dev_pairwise": -1.0, "state": None, "epoch": -1}
        for epoch in range(args.epochs):
            idx = list(range(len(train_pairs)))
            torch.manual_seed(seed + epoch * 1000)
            idx = torch.randperm(len(train_pairs)).tolist()
            head.train()
            for s in range(0, len(idx), args.batch_size):
                batch = [train_pairs[i] for i in idx[s:s + args.batch_size]]
                Xc = torch.tensor([p[0] for p in batch], dtype=torch.float32)
                Xw = torch.tensor([p[1] for p in batch], dtype=torch.float32)
                sc = head(Xc)
                sw = head(Xw)
                loss = -torch.log(torch.sigmoid(sc - sw) + 1e-9).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            head.eval()
            dp, dt, dg = _eval_pool(dev_cands, head)
            if dp > best["dev_pairwise"]:
                best = {"dev_pairwise": dp, "state": {k: v.clone() for k, v in head.state_dict().items()}, "epoch": epoch}
                import copy
                best["state"] = copy.deepcopy(head.state_dict())
            # early stop
            if epoch - best["epoch"] >= args.patience:
                break
        head.load_state_dict(best["state"])
        head.eval()
        dp, dt, dg = _eval_pool(dev_cands, head)
        heads.append(head)
        results.append({"seed": seed, "dev_pairwise": dp, "dev_top1": dt, "dev_greedy": dg, "best_epoch": best["epoch"]})
        print(f"seed {seed}: dev_pairwise={dp:.3f} dev_top1={dt:.3f} dev_greedy={dg:.3f} best_epoch={best['epoch']}", flush=True)

    # final = average logit over seeds
    def ensemble_score(feats):
        return float(sum(h(torch.tensor([feats], dtype=torch.float32)).item() for h in heads) / len(heads))
    # eval dev + confirmation with ensemble
    for cands, label in ((dev_cands, "dev"), (conf_cands, "confirmation")):
        by_qid = defaultdict(list)
        for c in cands:
            if c["candidate_type"] == "sampled":
                by_qid[str(c["qid"])].append(c)
        greedy = {str(c["qid"]): c for c in cands if c["candidate_type"] == "greedy"}
        for rows in by_qid.values():
            for c in rows:
                c["_vscore"] = ensemble_score(_feats(c))
        wins = ties = comps = 0
        for rows in by_qid.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a, b = rows[i], rows[j]
                    if a["em"] == b["em"]:
                        continue
                    corr, wrong = (a, b) if a["em"] > b["em"] else (b, a)
                    comps += 1
                    if corr["_vscore"] > wrong["_vscore"]:
                        wins += 1
                    elif corr["_vscore"] == wrong["_vscore"]:
                        ties += 1
        pairwise = (wins + 0.5 * ties) / comps if comps else 0.0
        top1 = [max(rows, key=lambda c: c["_vscore"]) for rows in by_qid.values()]
        top1_em = sum(c["em"] for c in top1) / len(top1)
        greedy_em = sum(greedy[q]["em"] for q in greedy) / len(greedy)
        print(f"{label}: pairwise={pairwise:.3f} top1={top1_em:.3f} greedy={greedy_em:.3f} (top1-greedy={top1_em-greedy_em:+.3f})", flush=True)

    report = {"experiment_id": experiment_id, "head_seeds": args.head_seeds, "results": results}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(run_dir, extra={"experiment_id": experiment_id, "phase": "train_l0_verifier"}, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
