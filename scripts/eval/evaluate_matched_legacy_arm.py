#!/usr/bin/env python
"""Greedy single-arm (legacy KG) inference on the frozen matched-control inputs.

Reuses the canonical run's FRESH passages and the strong SFT checkpoint, but
swaps ProofKG -> legacy KG, so the result can be paired against the already
frozen canonical Proof output (EM 0.64) with the same prompt/scorer (except the
KG block).  No re-retrieval, no Proof arm re-generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from kgproweight.data.prompts import build_inference_messages
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir, get_logger

logger = get_logger(__name__)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arm", default="legacy", help="Arm label written into predictions.")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input))
    assert len(rows) >= 1
    run_dir, experiment_id = prepare_new_run_dir(
        args.run_dir, experiment_id=args.experiment_id,
        extra={"phase": "matched_control_legacy_arm", "n": len(rows)},
    )

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    predictions: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        messages = build_inference_messages(
            question=str(row["question"]),
            retrieved_passages=list(row["retrieved_passages"]),
            kg_triples=list(row["kg_subgraph"]),
            top_k=10,
            max_kg_triples=12,
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            generation = model.generate(
                **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generation[:, encoded["input_ids"].shape[1]:]
        text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        answer = extract_kg_proweight_answer(text)
        golds = [str(g) for g in row.get("gold_answers") or [] if str(g).strip()]
        predictions.append({
            "row_id": row.get("row_id", row["qid"]), "qid": row["qid"], "question": row["question"],
            "gold_answers": golds, "arm": args.arm, "prediction": answer,
            "em": compute_em(answer, golds) if answer and golds else 0.0,
            "f1": compute_f1(answer, golds) if answer and golds else 0.0,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "generation": text,
        })
        print(f"legacy arm {index}/{len(rows)}", flush=True)

    pred_path = run_dir / "predictions.jsonl"
    pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions), encoding="utf-8")
    em = sum(r["em"] for r in predictions) / len(predictions)
    f1 = sum(r["f1"] for r in predictions) / len(predictions)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "status": "MATCHED_CONTROL_LEGACY_ARM",
        "n": len(predictions),
        "em": em, "f1": f1,
        "input_sha256": _sha256(Path(args.input)),
        "predictions_sha256": _sha256(pred_path),
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_manifest(run_dir, extra={"experiment_id": experiment_id, "phase": "matched_control_legacy_arm", "em": em, "f1": f1}, status="COMPLETE")
    print(json.dumps({"em": em, "f1": f1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
