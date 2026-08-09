#!/usr/bin/env python
"""Validate the Phase 3a SFT adapter on the held-out VAL fold.

What this answers: did SFT actually teach the ``[Step N] ... [Final Answer]``
schema, and does the student answer correctly on trajectories it never trained
on? Both are prerequisites for Phase 3b — PPO cannot shape a reward it cannot
parse, and every prior PPO run's ``valid_rate`` swinging 0.125-1.0 traces back to
unparseable rollouts.

    python scripts/eval/validate_sft.py --n 200
    python scripts/eval/validate_sft.py --n 200 --base_only   # untuned baseline

VAL, never TEST. The test fold is spent once, at the end, on the configuration
chosen on val — selecting anything by looking at test is tuning on test.

No retrieval index needed: the silver file already stores ``retrieved_passages``
per trajectory, so this replays the exact prompt SFT was trained against. That is
also the limitation — it measures the student's reasoning given fixed passages,
NOT end-to-end pipeline quality, which needs the wiki18 corpus and E5/BM25 index
(both absent locally: indexes/e5_Flat.index and indexes/bm25 are 0-byte
placeholders).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from kgproweight.data.prompts import build_rl_messages
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
    SplitSpec,
)
from kgproweight.eval.metrics import compute_em, compute_f1
from kgproweight.retrieval.hybrid import DEFAULT_TOPK
from kgproweight.utils.paths import model_path

def _norm(s: str) -> str:
    """Same normalisation compute_em uses, for the answer-visibility check."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


STEP_RE = re.compile(r"\[Step\s+(\d+)\]", re.IGNORECASE)
FINAL_RE = re.compile(r"\[Final Answer\]\s*(.*?)\s*(?:\[Step|\Z)", re.IGNORECASE | re.DOTALL)


def parse_trace(text: str):
    """Extract (n_steps, answer, well_formed) from a generation.

    ``well_formed`` requires at least one step AND a final answer, which is
    exactly what composite_reward needs to score a rollout. Steps must also be
    numbered from 1 without gaps — a trace that jumps [Step 1] -> [Step 3] means
    the schema was only partially learned.
    """
    nums = [int(m.group(1)) for m in STEP_RE.finditer(text)]
    fm = FINAL_RE.search(text)
    answer = (fm.group(1).strip() if fm else "").split("\n")[0].strip()
    contiguous = nums == list(range(1, len(nums) + 1))
    return len(nums), answer, bool(nums and fm and answer), contiguous


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter",
                    default="checkpoints/sft_student_split/final",
                    help="LoRA adapter dir. Ignored with --base_only.")
    ap.add_argument("--silver",
                    default="data/silver_data/silver_v1_reannotated.jsonl")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="Held-out fold. Use test ONLY for the final report.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--base_only", action="store_true",
                    help="Skip the adapter: measures what the untuned base model "
                         "does, which is the only way to tell whether SFT helped.")
    ap.add_argument("--out", default=None, help="Write per-example JSONL here.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    ap.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = ap.parse_args()

    if args.split == "test" and not args.base_only:
        print("!! You asked for the TEST fold. It should be touched once, after\n"
              "   val has already chosen the configuration. Ctrl-C now if this\n"
              "   is not that final run.\n", file=sys.stderr)

    spec = SplitSpec(val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                     seed=args.split_seed)
    reader = SilverDatasetReader(args.silver, split=args.split, split_spec=spec)
    trajs = reader.accepted()[: args.n]
    print("fold=%s  %d accepted in fold, evaluating %d"
          % (args.split, len(reader.accepted()), len(trajs)), flush=True)
    if not trajs:
        print("no trajectories in fold", file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = model_path("llama3-8B-instruct")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto")

    tag = "base"
    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = "sft"
        print("loaded adapter:", args.adapter, flush=True)
    model.eval()

    rows = []
    n_wf = n_contig = 0
    step_hist = Counter()
    ems, f1s = [], []
    # 73.5% of val golds appear verbatim in the top-15 passages, so a headline EM
    # conflates "extracted a visible string" with "reasoned to an answer".
    # Splitting the metric by visibility is what makes the number interpretable:
    # the not-in-passage subset is the one that needs multi-hop reasoning.
    ems_vis, ems_hid = [], []

    for i, t in enumerate(trajs):
        # build_rl_messages, not build_sft_messages: this is the PPO-time prompt
        # (no teacher trace), so the numbers describe what PPO will roll out from.
        msgs = build_rl_messages(
            question=t.question,
            retrieved_passages=list(t.retrieved_passages)[:DEFAULT_TOPK],
            kg_triples=t.kg_subgraph,
            top_k=DEFAULT_TOPK,
        )
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy: this is a measurement, not a sample
                # llama3-8b's generation_config ships temperature=0.6 / top_p=0.9.
                # They are ignored under do_sample=False but transformers warns on
                # every call; setting them to None keeps the log readable without
                # changing the decode.
                temperature=None,
                top_p=None,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        n_steps, answer, well_formed, contiguous = parse_trace(gen)
        gold = t.metadata.get("gold_answer") if isinstance(t.metadata, dict) else None
        golds = [g for g in [gold, t.answer] if g]
        em = compute_em(answer, golds) if golds and answer else 0.0
        f1 = compute_f1(answer, golds) if golds and answer else 0.0

        # Is the gold string literally in the passages the model was shown?
        blob = _norm(" ".join(
            (p.get("contents") or p.get("text") or "") if isinstance(p, dict) else str(p)
            for p in list(t.retrieved_passages)[:DEFAULT_TOPK]))
        visible = bool(golds) and any(_norm(g) and _norm(g) in blob for g in golds)
        (ems_vis if visible else ems_hid).append(em)

        n_wf += well_formed
        n_contig += contiguous
        step_hist[n_steps] += 1
        ems.append(em)
        f1s.append(f1)
        rows.append({"qid": t.qid, "question": t.question, "gold": golds,
                     "pred": answer, "n_steps": n_steps,
                     "well_formed": well_formed, "contiguous": contiguous,
                     "em": em, "f1": f1, "gold_in_passages": visible,
                     "generation": gen})

        if (i + 1) % 20 == 0:
            print("  %d/%d  parse_rate=%.3f  EM=%.3f  F1=%.3f"
                  % (i + 1, len(trajs), n_wf / (i + 1),
                     sum(ems) / len(ems), sum(f1s) / len(f1s)), flush=True)

    n = len(trajs)
    print("\n" + "=" * 62)
    print("Phase 3a SFT validation — %s model, fold=%s, n=%d" % (tag, args.split, n))
    print("=" * 62)
    # parse_rate is the gate on Phase 3b: composite_reward can only score a
    # rollout it can parse, so this bounds how much of PPO's batch carries signal.
    print("parse_rate (>=1 step + final answer) : %.3f  (%d/%d)" % (n_wf / n, n_wf, n))
    print("step numbering contiguous            : %.3f" % (n_contig / n))
    print("EM                                   : %.3f" % (sum(ems) / n))
    print("F1                                   : %.3f" % (sum(f1s) / n))
    # Breakdown, because a single EM hides which capability produced it.
    if ems_vis:
        print("  EM | gold visible in passages       : %.3f  (n=%d, %.1f%% of fold)"
              % (sum(ems_vis) / len(ems_vis), len(ems_vis), 100.0 * len(ems_vis) / n))
    if ems_hid:
        print("  EM | gold NOT in passages          : %.3f  (n=%d) <- needs reasoning"
              % (sum(ems_hid) / len(ems_hid), len(ems_hid)))
    print("steps per trace: %s"
          % ", ".join("%d:%d" % (k, step_hist[k]) for k in sorted(step_hist)))

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("\nwrote %d rows to %s" % (len(rows), p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
