"""Phase 3a — Supervised Fine-Tuning of the Student.

Bug-fix #7. Before PPO can train the Student to follow the
``[Step N] ... [Final Answer]`` schema, we first SFT it on the accepted
silver trajectories. Without this step the PPO rollout almost never
produces a parseable trace, so reward shaping cannot kick in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch

from kgproweight.data.prompts import (
    SFT_SYSTEM_PROMPT,
    build_saeg_sft_messages,
    build_sft_messages,
)
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
)
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.utils.logging import (
    artifact_identity,
    dump_manifest,
    get_logger,
    prepare_new_run_dir,
)
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
    # Continue training an existing LoRA adapter instead of reinitialising LoRA
    # on the base model.  This is required for the Proof-KG curriculum smoke.
    init_adapter_path: Optional[str] = None
    # Identity-safe per-question prompt/reward KG override.
    question_kg_records_path: Optional[str] = None
    min_question_kg_record_coverage: float = 1.0
    require_nonempty_question_kg_records: bool = False
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    # Keep intermediate model states for pre-registered checkpoint selection.
    # This is deliberately a persistence/evaluation control, not an extra SFT
    # regularizer: the optimiser objective and LoRA trainability are unchanged.
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    save_only_model: bool = False
    log_with: Optional[str] = None
    logging_dir: Optional[str] = None
    # Fold to train on. Must match the fold Phase 2 used, or the SFT model has
    # seen the trajectories the PRM head is later evaluated against. ``None``
    # reproduces the pre-split behaviour of training on the whole file.
    split: Optional[str] = None
    split_allow_none: bool = False
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    split_seed: Optional[int] = DEFAULT_SPLIT_SEED
    extra: Dict[str, Any] = field(default_factory=dict)

    def build_split_spec(self):
        from kgproweight.data.silver_split import SplitSpec

        return SplitSpec(
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed if self.split_seed is None else self.split_seed,
        )


def _render_assistant_trace(traj) -> str:
    """Render the accepted Teacher trajectory in the unified schema."""
    lines = []
    n = 0
    for step in traj.steps:
        # Labels are continuous r_kg values, so `== -1` missed anything strictly
        # between -1 and 0. Drop any step the KG net-contradicts.
        if float(step.label) <= -0.5:
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
        if traj.evidence_mode is not None or traj.passage_evidence:
            msgs = build_saeg_sft_messages(
                question=traj.question,
                retrieved_passages=list(traj.retrieved_passages)[:n_passages],
                kg_triples=traj.kg_subgraph,
                passage_evidence=traj.passage_evidence,
                answer_trace=asst,
                top_k=n_passages,
            )
        else:
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
    if cfg.save_strategy == "steps" and cfg.save_steps <= 0:
        raise ValueError("save_steps must be > 0 when save_strategy='steps'")
    if cfg.save_total_limit is not None and cfg.save_total_limit <= 0:
        raise ValueError("save_total_limit must be > 0 when provided")
    if cfg.split is None and not cfg.split_allow_none:
        raise ValueError(
            "Phase 3a split is None: this would train on the whole silver file, "
            "including val/test. Set split='train' (normal runs), or set "
            "split_allow_none=True only to reproduce a historical whole-file run."
        )
    if cfg.init_adapter_path and not Path(cfg.init_adapter_path).is_dir():
        raise FileNotFoundError(
            f"init_adapter_path must be an existing adapter directory: {cfg.init_adapter_path}"
        )
    if cfg.question_kg_records_path and not Path(cfg.question_kg_records_path).is_file():
        raise FileNotFoundError(
            "question_kg_records_path does not exist: "
            f"{cfg.question_kg_records_path}"
        )
    # Validate the fold and every dataset::qid/question-hash join before loading
    # the model. Bad curriculum data must fail in CPU preflight, not after a GPU
    # has been reserved.
    reader = SilverDatasetReader(
        cfg.silver_path,
        split=cfg.split,
        split_spec=cfg.build_split_spec() if cfg.split else None,
    )
    if cfg.split is None:
        logger.warning(
            "Phase 3a split: NONE — SFT trains on the whole file (%d trajectories, "
            "%d accepted). Nothing is held back.",
            len(reader.trajectories), len(reader.accepted()),
        )
    else:
        logger.info(
            "Phase 3a split: fold=%s -> %d/%d trajectories in file, %d accepted "
            "(val=%.3f test=%.3f split_seed=%d)",
            cfg.split, len(reader.trajectories), reader.n_total_in_file,
            len(reader.accepted()), cfg.val_ratio, cfg.test_ratio,
            cfg.build_split_spec().seed,
        )
    question_kg_stats = None
    if cfg.question_kg_records_path:
        records = read_question_kg_records(cfg.question_kg_records_path)
        question_kg_stats = apply_training_question_kg(
            reader.accepted(),
            records,
            min_coverage=cfg.min_question_kg_record_coverage,
            require_nonempty=cfg.require_nonempty_question_kg_records,
        ).to_dict()
        logger.info("SFT question-KG preflight: %s", question_kg_stats)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 3a SFT requires CUDA, but torch.cuda.is_available() is False. "
            "Fix the NVIDIA driver/container runtime before reserving an Experiment ID."
        )
    if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 3a dtype=bf16 but the active GPU does not support bf16")
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    set_seed(cfg.seed)
    base_id = model_path(cfg.base_model)
    out_dir, experiment_id = prepare_new_run_dir(
        cfg.output_dir,
        extra={
            "phase": "phase3_sft",
            "config": asdict(cfg),
            "input_artifacts": {
                "silver": artifact_identity(cfg.silver_path),
                "base_model": artifact_identity(base_id),
                "init_adapter": (
                    artifact_identity(cfg.init_adapter_path)
                    if cfg.init_adapter_path else None
                ),
                "question_kg_records": (
                    artifact_identity(cfg.question_kg_records_path)
                    if cfg.question_kg_records_path else None
                ),
            },
        },
    )
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype_map.get(cfg.dtype, torch.bfloat16), device_map="auto"
    )

    if cfg.init_adapter_path and not cfg.use_lora:
        raise ValueError("init_adapter_path requires use_lora=True")
    if cfg.init_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, cfg.init_adapter_path, is_trainable=True
        )
        peft_cfg = next(iter(model.peft_config.values()))
        actual_targets = set(peft_cfg.target_modules or [])
        expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
        if (
            int(peft_cfg.r) != cfg.lora_r
            or int(peft_cfg.lora_alpha) != cfg.lora_alpha
            or actual_targets != expected_targets
        ):
            raise ValueError(
                "init adapter LoRA config does not match requested training config: "
                f"r={peft_cfg.r}, alpha={peft_cfg.lora_alpha}, "
                f"targets={sorted(actual_targets)}"
            )
        model.print_trainable_parameters()
        model.enable_input_require_grads()
        logger.info("Continuing SFT from adapter %s", cfg.init_adapter_path)
    elif cfg.use_lora:
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
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_only_model=cfg.save_only_model,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        report_to=[cfg.log_with] if cfg.log_with else [],
        logging_dir=cfg.logging_dir,
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
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    trainer.train()

    # nvidia-smi reports the caching allocator's RESERVED bytes, which it does not
    # return to the driver once freed — so a run whose active peak is ~46 GB can
    # still show ~96 GB and look like it is about to OOM. Logging both numbers
    # makes the difference visible, which is what tells you whether headroom for a
    # larger batch/seq actually exists.
    if torch.cuda.is_available():
        logger.info(
            "SFT peak GPU memory: allocated %.2f GB | reserved %.2f GB "
            "(nvidia-smi shows the reserved figure)",
            torch.cuda.max_memory_allocated() / 1024 ** 3,
            torch.cuda.max_memory_reserved() / 1024 ** 3,
        )

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
    intermediate_checkpoints = sorted(
        (path for path in out_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )

    dump_manifest(
        out_dir,
        extra={
            "phase": "phase3_sft",
            "experiment_id": experiment_id,
            "config": asdict(cfg),
            "silver_path": str(cfg.silver_path),
            "input_artifacts": {
                "silver": artifact_identity(cfg.silver_path),
                "base_model": artifact_identity(base_id),
                "init_adapter": (
                    artifact_identity(cfg.init_adapter_path)
                    if cfg.init_adapter_path else None
                ),
                "question_kg_records": (
                    artifact_identity(cfg.question_kg_records_path)
                    if cfg.question_kg_records_path else None
                ),
            },
            "output_artifacts": {
                "final_checkpoint": artifact_identity(final_dir),
                "loss_history": artifact_identity(loss_path),
                "intermediate_checkpoints": [
                    artifact_identity(path) for path in intermediate_checkpoints
                ],
            },
            "global_step": trainer.state.global_step,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "base_model": base_id,
            "max_length": cfg.max_length,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
            "question_kg_override": question_kg_stats,
            "peak_gpu_gb": (
                {"allocated": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                 "reserved": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)}
                if torch.cuda.is_available() else None
            ),
            # Same reason as Phase 2: without this there is no way to tell later
            # whether this checkpoint saw the held-out trajectories.
            "split_info": {
                "split": cfg.split,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "split_seed": cfg.build_split_spec().seed,
                "n_trajectories": len(reader.trajectories),
                "n_trajectories_in_file": reader.n_total_in_file,
                "n_accepted": len(reader.accepted()),
            },
        },
    )
    logger.info("Phase 3a SFT done. Checkpoint at %s", final_dir)
    return {"output_dir": str(out_dir), "final_checkpoint": str(final_dir)}
