#!/usr/bin/env python
"""Phase 3a SFT (try variant) — QLoRA 4-bit so 8B fits a 24GB card.

The package's ``scripts/train/phase3_sft.py`` loads the base in bf16 (~16GB of
weights) and uses max_length=4096, which OOMs on a 24GB 4090 during the forward
pass. This standalone entry-point reuses the package's data pipeline
(``_render_assistant_trace`` / ``_build_dataset`` / ``_tokenise``) unchanged and
only changes the model-loading + Trainer side:

* 4-bit NF4 base (QLoRA) + ``prepare_model_for_kbit_training``,
* gradient checkpointing,
* a CLI-tunable ``--max_length`` (default 1024, vs the package's 4096).

The package files are left untouched. Run as a script (not -m).

    python scripts/train/try/phase3_sft_try.py \
        --silver scripts/train/try/outputs/silver_try_50b.jsonl \
        --output_dir checkpoints/sft_student_try [--max_length 1024]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make sibling try-modules importable regardless of CWD/subdir layout.
_TRY_ROOT = Path(__file__).resolve().parent.parent
for _d in (_TRY_ROOT, _TRY_ROOT / "shared", _TRY_ROOT / "phase1_distill",
           _TRY_ROOT / "phase2_prm", _TRY_ROOT / "phase3_sft", _TRY_ROOT / "phase3_ppo"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# reused, unchanged, from the package SFT module
import kgproweight.training.phase3_sft as p3sft
from kgproweight.training.phase3_sft import _build_dataset, _tokenise
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.utils.logging import configure_logging, dump_manifest, get_logger
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed

configure_logging("INFO")
logger = get_logger(__name__)


def _render_assistant_trace_fixed(traj) -> str:
    """Override of the package's _render_assistant_trace with two fixes:

    #3  Final answer is rendered from ``metadata['gold_answer']`` (falling back
        to ``traj.answer``), matching the target PPO's EM reward scores against.
        The package used ``traj.answer`` (the teacher's answer), so for the ~18%
        of accepted trajectories where teacher≠gold, SFT taught one answer and
        PPO rewarded a different one.
    #7  Steps are renumbered contiguously (1,2,3,…) while dropping label==-1
        steps, instead of keeping the original gappy indices (1,3,4,…) which
        teach the student a broken numbering scheme.
    """
    lines = []
    n = 0
    for step in traj.steps:
        if step.label == -1:
            continue  # drop hallucinated steps from SFT
        n += 1
        lines.append(f"[Step {n}]\n{step.text.strip()}")
    gold = (traj.metadata.get("gold_answer") if isinstance(traj.metadata, dict) else None)
    final = (gold or traj.answer or "").strip()
    if final:
        lines.append(f"[Final Answer]\n{final}")
    return "\n\n".join(lines)


# Patch the package module so the reused _build_dataset picks up the fixed render.
p3sft._render_assistant_trace = _render_assistant_trace_fixed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silver", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="llama3-8B-instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_4bit", action="store_true", help="load bf16 instead of 4-bit (needs >24GB)")
    p.add_argument("--merge_output", action="store_true",
                   help="after training, merge LoRA into a full bf16 model (out_dir/merged) for PPO")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    base_id = model_path(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False  # required with gradient checkpointing

    # --- data: reuse the package pipeline verbatim ---
    reader = SilverDatasetReader(args.silver)
    ds_raw = _build_dataset(reader, tokenizer, args.max_length)
    if len(ds_raw) == 0:
        raise ValueError(f"No accepted trajectories with a renderable trace in {args.silver}")
    ds_tok = _tokenise(ds_raw, tokenizer, args.max_length)
    logger.info("SFT (try) on %d accepted trajectories, max_length=%d, 4bit=%s",
                len(ds_raw), args.max_length, not args.no_4bit)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=args.seed,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tok, data_collator=collator)
    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    dump_manifest(out_dir, extra={
        "phase": "phase3_sft_try", "silver": args.silver, "epochs": args.epochs,
        "lr": args.lr, "max_length": args.max_length, "use_4bit": not args.no_4bit, "seed": args.seed,
    })
    logger.info("Phase 3a SFT (try) done. Adapter checkpoint at %s", final_dir)

    # --- optional merge into a full bf16 model so PPO can load it directly ---
    # The PPO entry-point feeds --sft_checkpoint straight to from_pretrained,
    # which needs a full model, not a bare LoRA adapter dir. Merging produces a
    # self-contained checkpoint. We do NOT merge on the 4-bit model (lossy);
    # instead reload the base in bf16, attach the just-trained adapter, merge.
    if args.merge_output:
        from peft import PeftModel

        del model, trainer
        torch.cuda.empty_cache()
        merged_dir = out_dir / "merged"
        logger.info("Merging LoRA into a full bf16 model at %s (reloading base in bf16)", merged_dir)
        base_bf16 = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        merged = PeftModel.from_pretrained(base_bf16, str(final_dir))
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        logger.info("Merged full model saved to %s — feed THIS to PPO --sft_checkpoint", merged_dir)


if __name__ == "__main__":
    main()

