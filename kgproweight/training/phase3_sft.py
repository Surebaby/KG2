"""Phase 3a — Supervised Fine-Tuning of the Student.

Bug-fix #7. Before PPO can train the Student to follow the
``[Step N] ... [Final Answer]`` schema, we first SFT it on the accepted
silver trajectories. Without this step the PPO rollout almost never
produces a parseable trace, so reward shaping cannot kick in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from kgproweight.data.prompts import (
    SFT_SYSTEM_PROMPT,
    build_sft_messages,
)
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class Phase3SFTConfig:
    silver_path: str
    output_dir: str
    base_model: str = "llama3-8B-instruct"
    dtype: str = "bf16"
    seed: int = 42
    epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 2.0e-5
    max_length: int = 4096
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    extra: Dict[str, Any] = field(default_factory=dict)


def _render_assistant_trace(traj) -> str:
    """Render the accepted Teacher trajectory in the unified schema."""
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


def _build_dataset(reader: SilverDatasetReader, tokenizer, max_length: int):
    """Tokenise each trajectory once, masking PROMPT tokens out of the loss.

    BUGFIX (2026-06-22): the previous version emitted only ``text`` and let
    ``DataCollatorForLanguageModeling`` build labels from the full sequence — so
    the loss covered the system prompt AND the retrieved-passages block. The
    Student learned to *reproduce passages* ("Retrieved Passage: ...") instead of
    reasoning, and at inference it echoed passages and rarely reached
    ``[Final Answer]``. We now build ``labels`` with the prompt region set to
    -100 so only the assistant trace is supervised.

    Also fixes a double-BOS bug: the chat template already prepends
    ``<|begin_of_text|>``; tokenising with the default ``add_special_tokens=True``
    added a second one. We tokenise with ``add_special_tokens=False``.

    Truncation strategy: the prompt (15 passages ≈ 3.7k tokens median) can still
    overflow ``max_length``. Rather than clipping the sequence — which would cut
    the answer off the END and destroy the only supervised tokens — we DROP the
    lowest-ranked passages until ``question + remaining passages + KG + answer``
    fits. The answer is always retained in full.
    """
    import datasets

    def _encode(traj, n_passages):
        asst = _render_assistant_trace(traj)
        if not asst.strip():
            return None
        msgs = build_sft_messages(
            question=traj.question,
            retrieved_passages=list(traj.retrieved_passages)[:n_passages],
            kg_triples=traj.kg_subgraph,
            answer_trace=asst,
            top_k=n_passages,
        )
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        p_len = len(p_ids) if f_ids[: len(p_ids)] == p_ids else 0
        return f_ids, p_len

    from kgproweight.retrieval.hybrid import DEFAULT_TOPK

    rows = []
    n_dropped_passages = 0
    n_skipped = 0
    for i_traj, traj in enumerate(reader.accepted()):
        if not hasattr(tokenizer, "apply_chat_template"):
            continue
        # PERF: start from the target passage budget (DEFAULT_TOPK=15), NOT the
        # full ~50 stored passages. Starting at 50 made the shrink loop re-tokenise
        # an ~11k-token prompt ~35 times per trajectory (≈700k tokenisations over
        # 9839 rows → tens of minutes). At 15 the prompt usually fits 4096 already,
        # so the loop runs 0-2 times.
        n_passages = min(len(traj.retrieved_passages), DEFAULT_TOPK)
        enc = _encode(traj, n_passages)
        if enc is None:
            continue
        # Shrink the passage set until the full sequence fits, keeping the answer.
        while enc[0] and len(enc[0]) > max_length and n_passages > 0:
            n_passages -= 1
            n_dropped_passages += 1
            enc = _encode(traj, n_passages)
        if (i_traj + 1) % 2000 == 0:
            logger.info("SFT data prep: %d trajectories processed...", i_traj + 1)
        full_ids, prompt_len = enc
        if not full_ids or len(full_ids) > max_length or prompt_len >= len(full_ids):
            # Even with 0 passages it doesn't fit, or nothing left to supervise.
            n_skipped += 1
            continue
        labels = list(full_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100  # mask prompt + passages out of the loss
        rows.append({"input_ids": full_ids, "labels": labels})

    if n_dropped_passages or n_skipped:
        logger.warning(
            "SFT data: dropped %d passages across rows to fit max_length=%d; "
            "skipped %d trajectories that couldn't fit even with 0 passages.",
            n_dropped_passages, max_length, n_skipped,
        )
    return datasets.Dataset.from_list(rows)


def _tokenise(ds, tokenizer, max_length: int):
    # No-op: _build_dataset now emits pre-tokenised input_ids + masked labels.
    # Kept for backward compatibility with any external callers.
    return ds


def run_phase3_sft(cfg: Phase3SFTConfig) -> Dict[str, Any]:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_id = model_path(cfg.base_model)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype_map.get(cfg.dtype, torch.bfloat16), device_map="auto"
    )

    if cfg.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            lora = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora)
            model.print_trainable_parameters()
            # Gradient checkpointing + LoRA: the frozen base inputs need grads so
            # backprop reaches the LoRA adapters through the checkpointed graph.
            model.enable_input_require_grads()
        except Exception as exc:
            logger.warning("PEFT unavailable (%s); full-parameter SFT.", exc)

    reader = SilverDatasetReader(cfg.silver_path)
    ds_raw = _build_dataset(reader, tokenizer, cfg.max_length)
    ds_tok = _tokenise(ds_raw, tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        bf16=cfg.dtype == "bf16",
        fp16=cfg.dtype == "fp16",
        logging_steps=20,
        save_strategy="epoch",
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        seed=cfg.seed,
    )

    from transformers import DataCollatorForSeq2Seq

    # Pads input_ids AND labels (labels padded with -100 so pad tokens are
    # ignored in the loss). Replaces DataCollatorForLanguageModeling, which would
    # overwrite our prompt-masked labels by rebuilding them from input_ids.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding="longest", label_pad_token_id=-100
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tok, data_collator=collator)
    trainer.train()

    # Record the loss curve to a clean sft_loss.jsonl (one row per logged step),
    # mirroring PPO's history.jsonl. trainer_state.json also has the raw
    # log_history, but this is the tidy, easy-to-plot version.
    import json as _json

    loss_path = out_dir / "sft_loss.jsonl"
    with open(loss_path, "w", encoding="utf-8") as fh:
        for rec in trainer.state.log_history:
            if "loss" in rec:  # training-loss rows (skip the final summary row)
                fh.write(_json.dumps({
                    "step": rec.get("step"),
                    "epoch": rec.get("epoch"),
                    "loss": rec.get("loss"),
                    "grad_norm": rec.get("grad_norm"),
                    "learning_rate": rec.get("learning_rate"),
                }) + "\n")
    logger.info("Wrote SFT loss curve (%d points) to %s",
                sum(1 for r in trainer.state.log_history if "loss" in r), loss_path)

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    dump_manifest(
        out_dir,
        extra={
            "phase": "phase3_sft",
            "silver_path": str(cfg.silver_path),
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "base_model": base_id,
        },
    )
    logger.info("Phase 3a SFT done. Checkpoint at %s", final_dir)
    return {"output_dir": str(out_dir), "final_checkpoint": str(final_dir)}
