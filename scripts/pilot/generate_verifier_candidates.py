#!/usr/bin/env python
"""Generate verifier-training candidates: greedy + K=4 sampled, scored with reward v2.1.

For the L0 learned verifier data pool.  Gold answers are used only to label EM
(never to score the reward).  Sampling uses the frozen K=4 protocol (temperature
1.0, top_p 1.0, max_new_tokens 512, seed 42).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.reward.proofkg_process_v2 import build_execution_trace, score_proofkg_v2
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir, get_logger

logger = get_logger(__name__)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _build_prompt(tokenizer, row) -> str:
    from kgproweight.data.prompts import build_inference_messages
    messages = build_inference_messages(
        question=str(row["question"]), retrieved_passages=list(row["retrieved_passages"]),
        kg_triples=list(row["kg_subgraph"]), top_k=10, max_kg_triples=12,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof_input", required=True)
    parser.add_argument("--question_kg_records", required=True)
    parser.add_argument("--runtime_details", required=True)
    parser.add_argument("--greedy_predictions", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--rollouts_per_qid", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    proof = _read_jsonl(Path(args.proof_input))
    kg_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(args.question_kg_records))}
    detail_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(args.runtime_details))}
    gold_by_qid = {str(r["qid"]): r.get("gold_answers") or [] for r in proof}
    greedy_by_qid = {str(r["qid"]): r for r in _read_jsonl(Path(args.greedy_predictions))}

    trace_by_qid: Dict[str, list] = {}
    planned_by_qid: Dict[str, int] = {}
    for qid, kg in kg_by_qid.items():
        plan = kg.get("query_plan") or {}
        planned_by_qid[qid] = len(plan.get("hops") or [])
        trace_by_qid[qid] = build_execution_trace(plan, detail_by_qid.get(qid, {}).get("execution") or {})

    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir, experiment_id=args.experiment_id, extra={"phase": "generate_verifier_candidates", "n": len(proof)}
    )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = {str(r["qid"]): _build_prompt(tokenizer, r) for r in proof}
    candidates: List[Dict[str, Any]] = []

    def score_one(qid, generation, ctype, cidx):
        kg = kg_by_qid[qid]
        proc = score_proofkg_v2(
            question=str(kg["question"]), generation=generation, kg_triples=kg.get("kg_subgraph") or [],
            execution_trace=trace_by_qid[qid], planned_hops=planned_by_qid[qid],
        )
        golds = [str(g) for g in gold_by_qid.get(qid, []) if str(g).strip()]
        return {
            "qid": qid, "question": kg["question"], "candidate_type": ctype,
            "candidate_index": cidx, "generation": generation,
            "process": proc,
            "em": compute_em(proc["prediction"], golds) if proc["prediction"] and golds else 0.0,
            "f1": compute_f1(proc["prediction"], golds) if proc["prediction"] and golds else 0.0,
        }

    # greedy candidates (from frozen greedy predictions)
    for r in proof:
        qid = str(r["qid"])
        g = greedy_by_qid[qid]
        candidates.append(score_one(qid, g["generation"], "greedy", 0))

    # K sampled candidates
    expanded = [(r, i) for r in proof for i in range(args.rollouts_per_qid)]
    batch_size = 2
    done = 0
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start:start + batch_size]
        encoded = tokenizer([prompts[str(r["qid"])] for r, _ in batch], return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
        plen = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p, top_k=0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, plen:], skip_special_tokens=True)
        for (r, cidx), gen in zip(batch, texts):
            candidates.append(score_one(str(r["qid"]), gen, "sampled", cidx))
        done += len(batch)
        if done % 20 == 0 or done == len(expanded):
            print(f"sampled {done}/{len(expanded)}", flush=True)

    out = run_dir / "candidates.jsonl"
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates), encoding="utf-8")
    n_sampled = sum(1 for c in candidates if c["candidate_type"] == "sampled")
    report = {"experiment_id": experiment_id, "n_qids": len(proof), "n_candidates": len(candidates), "n_sampled": n_sampled}
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(run_dir, extra={"experiment_id": experiment_id, "phase": "generate_verifier_candidates", **report}, status="COMPLETE")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
