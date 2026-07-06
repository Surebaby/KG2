#!/usr/bin/env python
import os as _os; _os.environ.pop("OMP_NUM_THREADS", None); _os.environ["OMP_NUM_THREADS"] = "8"
import os as _os; _os.environ.pop("OMP_NUM_THREADS", None); _os.environ["OMP_NUM_THREADS"] = "8"
from __future__ import annotations

# --- Fix AutoDL OMP_NUM_THREADS libgomp crash (before any c-extension import) ---
import os as _os
if "OMP_NUM_THREADS" in _os.environ:
    try: int(_os.environ["OMP_NUM_THREADS"])
    except (ValueError, TypeError): _os.environ["OMP_NUM_THREADS"] = "8"

"""Build hard-example training set: SFT inference → keep EM=0 trajectories.

Runs SFT checkpoint inference over the complete silver dataset (9,839 accepted
trajectories). For each trajectory, generates an answer using the SFT model and
compares it against the gold answer (EM). Trajectories where SFT's answer is
WRONG (EM=0) are written to a new JSONL file — these are the "hard examples"
that PPO can improve on.

Output: ``checkpoints/prm_alpha_gate/silver_hard_examples.jsonl`` (~6,000
trajectories, ~800 MB).

Usage (on remote GPU server):
    /root/autodl-tmp/kgpw_env/bin/python -u scripts/prepare/build_hard_examples.py \
        --silver checkpoints/prm_alpha_gate/silver_with_logprobs.jsonl \
        --sft_checkpoint checkpoints/sft_student/final \
        --output checkpoints/prm_alpha_gate/silver_hard_examples.jsonl

Estimated: 9,839 generations × ~4s = ~11 hours on a single GPU.
For a faster first pass, use --max_samples 2000 to test the pipeline.
"""

import argparse
import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Force early flush so we can see where it crashes
print("build_hard_examples: starting imports...", flush=True)

import torch

# Package imports
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kgproweight.data.prompts import build_rl_messages
from kgproweight.data.parsers import extract_final_answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _em(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def _load_model(sft_checkpoint: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_id = os.environ.get("KGPW_LLAMA3_PATH", "")
    if not base_id:
        base_id = str(Path(sft_checkpoint).parent.parent / "models" / "llama3-8b")
    if not os.path.exists(base_id):
        # AutoDL default
        base_id = "/root/autodl-tmp/models/llama3-8b"

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_id, quantization_config=bnb_config, device_map="cuda:0")

    # Load LoRA adapter
    adapter_path = sft_checkpoint
    if os.path.isdir(adapter_path) and os.path.exists(
        os.path.join(adapter_path, "adapter_config.json")
    ):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded SFT adapter from {adapter_path}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silver", required=True, help="Path to silver_with_logprobs.jsonl")
    ap.add_argument("--sft_checkpoint", required=True, help="SFT LoRA adapter dir")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap number of trajectories (for testing)")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for generation (keep 1 for memory)")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (deterministic) for SFT error mining")
    return ap.parse_args()


def run(args):
    # --- Load data ---
    print(f"Loading silver trajectories from {args.silver}...")
    trajectories = []
    with open(args.silver, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("accepted", False):
                trajectories.append(obj)

    total = len(trajectories)
    print(f"  Accepted trajectories: {total}")
    if args.max_samples and args.max_samples < total:
        trajectories = trajectories[: args.max_samples]
        total = len(trajectories)
        print(f"  Capped to: {total}")

    # --- Load model ---
    print(f"Loading SFT model from {args.sft_checkpoint}...")
    model, tokenizer = _load_model(args.sft_checkpoint)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Model on: {device}")

    # --- Inference ---
    hard_examples: List[dict] = []
    correct = 0
    wrong = 0
    skipped = 0
    t0 = time.time()

    for idx, traj in enumerate(trajectories):
        question = traj.get("question", "")
        gold = str(traj.get("metadata", {}).get("gold_answer", "") or "").strip()
        if not gold:
            skipped += 1
            continue

        retrieved = traj.get("retrieved_passages") or []
        kg = traj.get("kg_subgraph") or []

        msgs = build_rl_messages(
            question=question,
            retrieved_passages=retrieved if isinstance(retrieved, list) else [],
            kg_triples=kg if isinstance(kg, list) else [],
            top_k=15,
            max_kg_triples=30,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n\n".join(m["content"] for m in msgs)

        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=4096).to(device)

        with torch.no_grad():
            if args.temperature == 0.0:
                gen = model.generate(
                    input_ids=enc["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )[0]
            else:
                gen = model.generate(
                    input_ids=enc["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=1.0,
                    top_k=0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )[0]

        rids = gen[enc["input_ids"].size(1):]
        resp = tokenizer.decode(rids, skip_special_tokens=True)
        pred = extract_final_answer(resp) or resp.strip().split()[-1] if resp.strip() else ""

        is_correct = _em(pred, gold)
        if is_correct:
            correct += 1
        else:
            wrong += 1
            hard_examples.append(traj)

        if (idx + 1) % 500 == 0:
            n = idx + 1
            elapsed = time.time() - t0
            eta = elapsed / n * (total - n) if n > 0 else 0
            acc = correct / n
            print(f"  [{n:5d}/{total}]  SFT EM={acc:.3f}  "
                  f"correct={correct}  wrong={wrong}  hard={len(hard_examples)}  "
                  f"eta={eta/3600:.1f}h")

    # --- Summary ---
    elapsed = (time.time() - t0) / 60
    total_valid = correct + wrong
    acc = correct / max(total_valid, 1)
    print(f"\n{'='*55}")
    print(f"  SFT Error Mining — DONE  ({elapsed:.0f} min)")
    print(f"  Total:     {total_valid}  (skipped no-gold: {skipped})")
    print(f"  SFT EM:    {acc:.3f}  (correct={correct})")
    print(f"  Hard (EM=0): {len(hard_examples)}  ({len(hard_examples)/max(total_valid,1)*100:.0f}%)")
    print(f"{'='*55}")

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for traj in hard_examples:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(hard_examples)} hard examples → {out_path} ({size_mb:.0f} MB)")

    return {
        "total": total_valid,
        "sft_em": acc,
        "correct": correct,
        "hard_examples": len(hard_examples),
        "output": str(out_path),
    }


if __name__ == "__main__":
    main()
