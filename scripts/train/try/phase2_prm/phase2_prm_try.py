#!/usr/bin/env python
"""Phase 2 PRM + α-Gate (try variant) — 4-bit + correctness fixes.

Two concerns are handled here, neither by touching the package files:

1. **4-bit fit**: the package ``phase2_prm`` loads the base twice in bf16
   (logprob pre-pass with ``AutoModelForCausalLM`` + PRM-head training with
   ``AutoModel``), which OOMs on a 24GB 4090. We monkeypatch the two model
   loaders to 4-bit NF4 at runtime.

2. **Correctness fixes** (a code review found these in the package's
   ``run_phase2``; we reimplement the loop as ``run_phase2_fixed`` here, reusing
   every package helper — PRMHead / AlphaGate / _StepDataset / _collate — so only
   the buggy lines change):
     - #1  Phase 2 trained on REJECTED trajectories too (the try Phase 1 writes
           accepted+rejected to one file). Fixed: build samples from
           ``reader.accepted()`` only.
     - #2  α-gate calibration was degenerate — the same thresholded ``coverage``
           was both an input feature AND the BCE target, so the gate learned to
           copy one feature and nothing about density/entropy. Fixed: target is
           "does the KG render a verdict on this step" (label ≠ 0/NEUTRAL),
           which is INDEPENDENT of the three gate inputs; link-confidence feature
           is the continuous coverage, not the thresholded copy of the target.
     - #4  PRM head read ``last_hidden_state[:, -1, :]`` = the PAD position for
           right-padded short rows. Fixed: gather the last NON-pad token via the
           attention mask.
     - #5  the logprob write-back desynced from ``samples`` under
           ``--binary_labels_only`` (it skipped only empty-text, not label==0).
           Fixed: provenance-indexed write-back that can't drift.

   Use ``--legacy`` to fall back to the original monkeypatched ``run_phase2``
   (kept for comparison / reproducing the old behaviour).

    python scripts/train/try/phase2_prm/phase2_prm_try.py \
        --silver scripts/train/try/outputs/silver_try_80.jsonl \
        --output_dir checkpoints/prm_alpha_gate_try [--max_length 1024] [--legacy]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

# Make sibling try-modules importable regardless of CWD/subdir layout.
_TRY_ROOT = Path(__file__).resolve().parent.parent
for _d in (_TRY_ROOT, _TRY_ROOT / "shared", _TRY_ROOT / "phase1_distill",
           _TRY_ROOT / "phase2_prm", _TRY_ROOT / "phase3_sft", _TRY_ROOT / "phase3_ppo"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import kgproweight.training.phase2_prm as p2
from kgproweight.training.phase2_prm import (
    Phase2Config,
    run_phase2,
    _StepSample,
    _label_to_class,
    PRMHead,
    _StepDataset,
    _collate,
)
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.parsers import parsed_step_from_silver_dict
from kgproweight.kg.entity_linker import EntityLinker
from entity_filter_try import clean_entities
from kgproweight.kg.coverage import graph_density
from kgproweight.reward.alpha_gate import AlphaGate, AlphaCalibrationLoss, entropy_from_logprobs, compute_link_confidence
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed
from kgproweight.utils.logging import dump_manifest, configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


def _bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _compute_step_logprobs_4bit(
    samples: Sequence[_StepSample],
    base_model_id: str,
    device: str = "cuda",
    dtype: str = "bf16",
    batch_size: int = 1,
    max_length: int = 1024,
) -> List[float]:
    """4-bit drop-in for the package's logprob pre-pass (same return contract)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=_bnb_config(), device_map={"": 0}
    )
    model.eval()
    out: List[float] = []
    for s in samples:
        with torch.no_grad():
            enc = tokenizer(s.text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            input_ids = enc["input_ids"]
            outputs = model(input_ids=input_ids, labels=input_ids)
            out.append(-outputs.loss.item())
    del model
    torch.cuda.empty_cache()
    return out


def _build_base_model_4bit(cfg: Phase2Config):
    """4-bit drop-in for the package's PRM base builder (returns base, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    from kgproweight.utils.paths import model_path

    base_id = model_path(cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(base_id, quantization_config=_bnb_config(), device_map={"": 0})
    if cfg.use_lora:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type=TaskType.FEATURE_EXTRACTION,
        )
        base = get_peft_model(base, lora_cfg)
        base.print_trainable_parameters()
    return base, tokenizer


# ===========================================================================
# Fixed Phase 2 (reuses package helpers; only the buggy lines change)
# ===========================================================================

@dataclass
class _SampleWithProvenance:
    """A flattened step sample plus its (traj_idx, step_idx) origin, so the
    logprob write-back can never desync from the sample list (#5)."""
    sample: _StepSample
    traj_idx: int
    step_idx: int


def _build_samples_accepted_only(
    reader: SilverDatasetReader, *, binary_labels_only: bool, entity_linker: EntityLinker
) -> List[_SampleWithProvenance]:
    """#1: iterate ONLY accepted trajectories. #5: record provenance.

    Mirrors the package's ``_step_samples_from_silver`` skip logic exactly
    (empty text, and label==0 under binary mode) but ties every produced sample
    to its source (traj_idx into ``reader.accepted()``, step_idx into that
    trajectory's steps), so write-back indexes by provenance, not a parallel
    counter that can drift.

    Finding 2 (align with PPO): the ``_StepSample.coverage`` field now carries the
    **step-level link_confidence** — ``compute_link_confidence(step_entities, …)``
    over the entities parsed from the step text — computed EXACTLY the way PPO's
    composite_reward does at inference (same parser, same EntityLinker, same fn).
    Previously this field held the trajectory-level ``metadata['coverage']``
    (a per-trajectory constant), so the α-gate was trained on a different feature
    distribution than it sees at PPO time. The α-gate's middle input (init weight
    1.5, the largest) is this channel, so the mismatch was a silent miscalibration.
    """
    accepted = reader.accepted()
    out: List[_SampleWithProvenance] = []
    for t_idx, traj in enumerate(accepted):
        quality = 1 if traj.accepted else -1  # always +1 here (accepted-only)
        for s_idx, step in enumerate(traj.steps):
            text = step.text or ""
            if not text.strip():
                continue
            label = int(step.label)
            if binary_labels_only and label == 0:
                continue
            # Step entities via the SAME parser PPO uses (parse_steps → ParsedStep),
            # then the SAME scaffold filter, then the SAME link-confidence fn. All
            # three must match PPO or the train/inference contract breaks.
            parsed = parsed_step_from_silver_dict(step.to_dict(), fallback_index=s_idx)
            step_entities = clean_entities(parsed.mentioned_entities)
            link_conf = compute_link_confidence(
                step_entities=step_entities,
                entity_linker=entity_linker,
            )
            out.append(
                _SampleWithProvenance(
                    sample=_StepSample(
                        text=text,
                        label=label,
                        label_class=_label_to_class(label),
                        kg_subgraph=list(traj.kg_subgraph),
                        coverage=link_conf,  # Finding 2: step-level link_confidence
                        binary_quality=quality,
                        semantic_entropy=0.0,
                    ),
                    traj_idx=t_idx,
                    step_idx=s_idx,
                )
            )
    return out


def _last_nonpad_hidden(last_hidden_state: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """#4: gather the last NON-pad token's hidden state per row, instead of
    ``[:, -1, :]`` which is a PAD position for right-padded short sequences."""
    lengths = attention.long().sum(dim=1) - 1            # index of last real token
    lengths = lengths.clamp(min=0)
    batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, lengths].float()


def run_phase2_fixed(cfg: Phase2Config) -> Dict[str, Any]:
    """Reimplemented Phase 2 loop with fixes #1/#2/#4/#5. Loaders are the 4-bit
    monkeypatches set in ``main`` (compute_step_logprobs / _build_base_model)."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    logger.info("Loading silver data from %s", cfg.silver_path)
    reader = SilverDatasetReader(cfg.silver_path)
    accepted = reader.accepted()
    logger.info(
        "Phase2-fixed: %d/%d trajectories accepted (training on accepted only).",
        len(accepted), len(reader.trajectories),
    )
    # Finding 2: same EntityLinker PPO uses, loaded with the SAME on-disk entity
    # cache (resolve_entity_cache_path). Without the cache, link_confidence is a
    # constant 0 — aligned but dead. With it, the feature is live (e.g. 0.77/0.80/1.0)
    # and identical to PPO inference.
    from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
    entity_linker = EntityLinker(cache_path=resolve_entity_cache_path())
    logger.info("Phase2 link_confidence: EntityLinker cache=%s (%d entries)",
                resolve_entity_cache_path(), len(list(entity_linker.cache.items())))
    prov = _build_samples_accepted_only(
        reader, binary_labels_only=cfg.binary_labels_only, entity_linker=entity_linker
    )
    if not prov:
        raise ValueError(f"No step samples found in accepted trajectories of {cfg.silver_path}")
    samples = [p.sample for p in prov]

    # ---- Logprob pre-pass (4-bit loader via monkeypatch) -----------------
    logger.info("Logprob pre-pass over %d steps using %s", len(samples), model_path(cfg.base_model))
    logprob_means = p2.compute_step_logprobs(
        samples,
        base_model_id=model_path(cfg.base_model),
        device=cfg.device,
        dtype=cfg.logprob_dtype,
        max_length=cfg.max_length,
    )
    # #5: provenance-indexed write-back — cannot desync from `samples`.
    for flat_idx, p in enumerate(prov):
        lp = [float(logprob_means[flat_idx])]
        accepted[p.traj_idx].steps[p.step_idx].token_logprobs = lp
        samples[flat_idx].semantic_entropy = entropy_from_logprobs(lp)

    enriched_path = out_dir / "silver_with_logprobs.jsonl"
    SilverDatasetReader.write_jsonl(enriched_path, reader.trajectories)
    logger.info("Wrote enriched silver data to %s", enriched_path)

    # ---- Model assembly (4-bit loader via monkeypatch) -------------------
    base, tokenizer = p2._build_base_model(cfg)
    hidden_size = getattr(base.config, "hidden_size", None) or base.config.to_dict().get("hidden_size", 4096)
    prm_head = PRMHead(hidden_size=hidden_size, n_classes=3).to(device=cfg.device, dtype=torch.float32)
    alpha_gate = AlphaGate().to(device=cfg.device, dtype=torch.float32)
    text_reward_head = None
    if cfg.train_text_reward_head:
        text_reward_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh()).to(
            device=cfg.device, dtype=torch.float32
        )

    trainable = list(filter(lambda p: p.requires_grad, base.parameters()))
    params = trainable + list(prm_head.parameters()) + list(alpha_gate.parameters())
    if text_reward_head is not None:
        params += list(text_reward_head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    calibration = AlphaCalibrationLoss(weight=cfg.calibration_weight)
    text_mse = nn.MSELoss()

    ds = _StepDataset(samples, tokenizer=tokenizer, max_length=cfg.max_length)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: _collate(b, pad_token_id=tokenizer.pad_token_id),
    )

    base.train(); prm_head.train(); alpha_gate.train()
    if text_reward_head is not None:
        text_reward_head.train()
    step_count = 0
    history: List[Dict[str, float]] = []
    for epoch in range(cfg.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention = batch["attention_mask"].to(cfg.device)
            labels_class = batch["label_class"].to(cfg.device)
            density = batch["graph_density"].to(cfg.device)
            # Finding 2: batch["coverage"] now carries the STEP-LEVEL link_confidence
            # (see _build_samples_accepted_only) — the same quantity PPO feeds the
            # gate at inference. Kept the batch key "coverage" to avoid touching the
            # package _StepDataset/_collate; the variable is the link_confidence feat.
            link_conf_feat = batch["coverage"].to(cfg.device)
            entropy_real = batch["semantic_entropy"].to(cfg.device)

            outputs = base(input_ids=input_ids, attention_mask=attention)
            last_hidden = _last_nonpad_hidden(outputs.last_hidden_state, attention)  # #4
            logits = prm_head(last_hidden)
            loss_prm = ce(logits, labels_class)

            # #2: non-degenerate α-gate calibration.
            #   - link_confidence feature: the STEP-LEVEL entity-linker confidence
            #     (Finding 2 — matches PPO inference), not a thresholded copy of
            #     the target and not the trajectory-level coverage constant.
            #   - target: "did the KG render a verdict on this step?" i.e. the step
            #     is NOT neutral (label_class != 1 → +1/-1 carry KG signal). This is
            #     INDEPENDENT of the 3 gate inputs (density, link_conf, entropy), so
            #     the gate must actually combine them rather than copy one feature.
            link_confidence = link_conf_feat.clamp(0.0, 1.0)  # already [0,1]; defensive
            alpha = alpha_gate(density, link_confidence, entropy_real)
            kg_has_verdict = (labels_class != 1).float()
            loss_cal = calibration(alpha, kg_has_verdict)

            loss = loss_prm + loss_cal
            if text_reward_head is not None:
                tr = text_reward_head(last_hidden).squeeze(-1)
                tr_target = torch.where(
                    labels_class == 2, torch.ones_like(tr),
                    torch.where(labels_class == 0, -torch.ones_like(tr), torch.zeros_like(tr)),
                )
                loss_text = text_mse(tr, tr_target)
                loss = loss + cfg.text_reward_lr * loss_text
            loss = loss / cfg.grad_accum
            loss.backward()
            if (step_count + 1) % cfg.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
            step_count += 1

            if step_count % 10 == 0:
                total = float(loss.item()) * cfg.grad_accum
                rec = {"epoch": float(epoch), "step": float(step_count), "loss": total,
                       "prm": float(loss_prm.item()), "cal": float(loss_cal.item())}
                history.append(rec)
                logger.info("epoch=%d step=%d loss=%.4f (prm=%.4f cal=%.4f)",
                            epoch, step_count, total, loss_prm.item(), loss_cal.item())

    # ---- Save -------------------------------------------------------------
    base.eval(); prm_head.eval(); alpha_gate.eval()
    if text_reward_head is not None:
        text_reward_head.eval()
    prm_dir = out_dir / "prm_head"
    prm_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(base, "save_pretrained"):
        base.save_pretrained(prm_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(prm_dir)
    torch.save(prm_head.state_dict(), prm_dir / "prm_head.pt")
    torch.save(alpha_gate.state_dict(), out_dir / "alpha_gate.pt")
    if text_reward_head is not None:
        torch.save(text_reward_head.state_dict(), out_dir / "text_reward_head.pt")

    history_path = out_dir / "history.jsonl"
    with open(history_path, "w", encoding="utf-8") as fh:
        for row in history:
            fh.write(json.dumps(row) + "\n")

    dump_manifest(out_dir, extra={
        "phase": "phase2_prm_fixed", "silver_path": str(cfg.silver_path),
        "enriched_silver": str(enriched_path),
        "accepted_trajectories": len(accepted), "total_trajectories": len(reader.trajectories),
        "n_samples": len(samples), "epochs": cfg.epochs, "lr": cfg.lr, "seed": cfg.seed,
        "alpha_W": alpha_gate.W.data.cpu().tolist(),
        "alpha_b": float(alpha_gate.b.data.cpu().item()),
        "alpha_tau": float(alpha_gate.tau.cpu().item()),
        "fixes": ["#1 accepted-only", "#2 non-degenerate calibration",
                  "#4 last-nonpad hidden", "#5 provenance write-back"],
    })
    logger.info("Phase 2 (fixed) complete. α-gate W=%s b=%.4f tau=%.4f",
                alpha_gate.W.data.cpu().tolist(), float(alpha_gate.b.item()), float(alpha_gate.tau.item()))
    return {"output_dir": str(out_dir), "alpha_gate_path": str(out_dir / "alpha_gate.pt")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silver", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="llama3-8B-instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--binary_labels_only", action="store_true")
    p.add_argument("--legacy", action="store_true",
                   help="Use the original (buggy) package run_phase2 instead of the fixed loop.")
    return p.parse_args()



def main():
    args = parse_args()
    # Patch the two loaders to 4-bit; both run_phase2 and run_phase2_fixed call
    # them via the p2 module namespace, so the patch takes either way.
    p2.compute_step_logprobs = _compute_step_logprobs_4bit
    p2._build_base_model = _build_base_model_4bit
    logger.info("Phase 2 (try): patched model loaders to 4-bit NF4.")

    cfg = Phase2Config(
        silver_path=args.silver,
        output_dir=args.output_dir,
        base_model=args.base_model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        binary_labels_only=args.binary_labels_only,
    )
    if args.legacy:
        logger.warning("Phase 2 (try): --legacy → original package run_phase2 (known bugs #1/#2/#4/#5).")
        result = run_phase2(cfg)
    else:
        logger.info("Phase 2 (try): fixed loop (#1 accepted-only, #2 calibration, #4 hidden, #5 write-back).")
        result = run_phase2_fixed(cfg)
    logger.info("Phase 2 (try) result: %s", result)


if __name__ == "__main__":
    main()
