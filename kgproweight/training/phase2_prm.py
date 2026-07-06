"""Phase 2 — PRM head + α-Gate joint training.

Bug-fix #3: the legacy code hardcoded ``semantic_entropy = 0.5`` for every
step. We now run a *logprob pre-pass* over silver data to compute the
real token logprobs, persist them to ``silver_with_logprobs.jsonl``, and
feed them into both the PRM cross-entropy and the α-Gate calibration loss.

Outputs
-------
- ``<output_dir>/prm_head/`` — LoRA adapter on the base LM + PRM linear head.
- ``<output_dir>/alpha_gate.pt`` — trained α-Gate state dict.
- ``<output_dir>/manifest.json`` — reproducibility manifest.
- ``<output_dir>/silver_with_logprobs.jsonl`` — silver data with per-step logprobs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory
from kgproweight.data.parsers import parsed_step_from_silver_dict
from kgproweight.data.entity_filter import clean_entities
from kgproweight.kg.coverage import graph_density
from kgproweight.kg.entity_linker import EntityLinker
from kgproweight.reward.alpha_gate import (
    AlphaCalibrationLoss,
    AlphaGate,
    compute_link_confidence,
    entropy_from_logprobs,
)
from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
from kgproweight.utils.logging import dump_manifest, get_logger
from kgproweight.utils.paths import model_path
from kgproweight.utils.seed import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step-level dataset (already token-aligned)
# ---------------------------------------------------------------------------

@dataclass
class _StepSample:
    text: str
    label: int                       # +1 / 0 / -1
    label_class: int                 # 0 / 1 / 2 — index for cross-entropy
    kg_subgraph: List[tuple]
    coverage: float                  # holds step-level link_confidence (Finding 2)
    binary_quality: int              # +1 if accepted, -1 otherwise
    semantic_entropy: float          # populated after the logprob pre-pass


@dataclass
class _SampleWithProvenance:
    """A step sample plus where it came from, so the logprob pre-pass can write
    results back to the exact (trajectory, step) without a fragile parallel
    counter (fix #5 — the old flat_idx desynced under binary_labels_only)."""
    sample: _StepSample
    traj_idx: int
    step_idx: int


def _label_to_class(label: int) -> int:
    return {-1: 0, 0: 1, 1: 2}.get(label, 1)


def _build_samples_accepted_only(
    reader: SilverDatasetReader,
    *,
    binary_labels_only: bool = False,
    entity_linker: EntityLinker,
) -> List[_SampleWithProvenance]:
    """Build step samples from ACCEPTED trajectories only (fix #1).

    ``coverage`` carries the STEP-LEVEL link_confidence computed with the same
    parser + scaffold filter + fn the PPO reward uses (Finding 2), so the α-gate
    sees the same feature distribution at training and inference time. Provenance
    (traj_idx into ``reader.accepted()``, step_idx into ``traj.steps``) is recorded
    for the logprob write-back (fix #5).
    """
    accepted = reader.accepted()
    out: List[_SampleWithProvenance] = []
    for t_idx, traj in enumerate(accepted):
        quality = 1 if traj.accepted else -1
        for s_idx, step in enumerate(traj.steps):
            text = step.text or ""
            if not text.strip():
                continue
            label = int(step.label)
            if binary_labels_only and label == 0:
                continue
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
                        coverage=float(link_conf),
                        binary_quality=quality,
                        semantic_entropy=0.0,
                    ),
                    traj_idx=t_idx,
                    step_idx=s_idx,
                )
            )
    return out


def _step_samples_from_silver(reader: SilverDatasetReader, *, binary_labels_only: bool = False) -> List[_StepSample]:
    """Legacy builder kept for back-compat. Trains on ALL trajectories and fills
    ``coverage`` with the trajectory-level constant. Prefer
    ``_build_samples_accepted_only`` (used by run_phase2)."""
    out: List[_StepSample] = []
    for traj in reader:
        coverage = float(traj.metadata.get("coverage", 0.0))
        quality = 1 if traj.accepted else -1
        for step in traj.steps:
            text = step.text or ""
            if not text.strip():
                continue
            label = int(step.label)
            if binary_labels_only and label == 0:
                continue
            out.append(
                _StepSample(
                    text=text,
                    label=label,
                    label_class=_label_to_class(label),
                    kg_subgraph=list(traj.kg_subgraph),
                    coverage=coverage,
                    binary_quality=quality,
                    semantic_entropy=0.0,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Logprob pre-pass
# ---------------------------------------------------------------------------

def compute_step_logprobs(
    samples: Sequence[_StepSample],
    base_model_id: str,
    device: str = "cuda",
    dtype: str = "bf16",
    batch_size: int = 16,
    max_length: int = 1024,
) -> List[float]:
    """Mean token logprob per step. Batched (scale fix): the old version ran one
    forward per step and forced a GPU→CPU sync each iteration, costing hours at
    ~15k steps. We now pad a batch, run a single forward, and compute each row's
    mean logprob from a manual shifted cross-entropy over the attention mask
    (``outputs.loss`` averages over pad and cannot be used with padding)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch_dtype, device_map=device)
    model.eval()

    out: List[float] = []
    texts = [s.text for s in samples]
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        with torch.no_grad():
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            # shift for next-token prediction
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attn[:, 1:].float()
            logprobs = torch.log_softmax(shift_logits, dim=-1)
            tok_lp = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
            tok_lp = tok_lp * shift_mask
            denom = shift_mask.sum(dim=1).clamp(min=1.0)
            mean_lp = (tok_lp.sum(dim=1) / denom)  # (B,) — signed mean logprob
        out.extend(mean_lp.detach().cpu().tolist())
    return out


# ---------------------------------------------------------------------------
# PRM model: a base LM + a 3-way classification head over the last hidden state.
# ---------------------------------------------------------------------------

class PRMHead(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int = 3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        return self.proj(last_hidden_state)


def _last_nonpad_hidden(last_hidden_state: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """Hidden state at each row's LAST REAL token (fix #4).

    The old code took ``[:, -1, :]`` which, for right-padded short rows, is a PAD
    position — feeding the PRM head garbage. We index the last non-pad token per
    row using the attention mask."""
    lengths = attention.long().sum(dim=1) - 1   # index of last real token
    lengths = lengths.clamp(min=0)
    batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, lengths].float()


class _StepDataset(Dataset):
    def __init__(self, samples: Sequence[_StepSample], tokenizer, max_length: int = 1024) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        enc = self.tokenizer(
            s.text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_class": torch.tensor(s.label_class, dtype=torch.long),
            "graph_density": torch.tensor(graph_density(s.kg_subgraph), dtype=torch.float32),
            "coverage": torch.tensor(s.coverage, dtype=torch.float32),
            "semantic_entropy": torch.tensor(s.semantic_entropy, dtype=torch.float32),
        }


def _collate(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        attention[i, :L] = b["attention_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "label_class": torch.stack([b["label_class"] for b in batch]),
        "graph_density": torch.stack([b["graph_density"] for b in batch]),
        "coverage": torch.stack([b["coverage"] for b in batch]),
        "semantic_entropy": torch.stack([b["semantic_entropy"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Phase 2 config + main loop
# ---------------------------------------------------------------------------

@dataclass
class Phase2Config:
    silver_path: str
    output_dir: str
    base_model: str = "llama3-8B-instruct"
    dtype: str = "bf16"
    device: str = "cuda"
    seed: int = 42
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 5.0e-5
    max_length: int = 2048
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    calibration_weight: float = 0.1
    train_text_reward_head: bool = True
    text_reward_lr: float = 1.0e-4
    text_reward_path: Optional[str] = None  # output path for the head
    logprob_dtype: str = "bf16"
    binary_labels_only: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def _build_base_model(cfg: Phase2Config):
    from transformers import AutoModel, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    base_id = model_path(cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(base_id, torch_dtype=torch_dtype, device_map=cfg.device)

    if cfg.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            base = get_peft_model(base, lora_cfg)
            base.print_trainable_parameters()
        except Exception as exc:
            logger.warning("PEFT unavailable (%s); falling back to full-parameter training.", exc)
    return base, tokenizer


def run_phase2(cfg: Phase2Config) -> Dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    logger.info("Loading silver data from %s", cfg.silver_path)
    reader = SilverDatasetReader(cfg.silver_path)
    accepted = reader.accepted()
    logger.info(
        "Phase2: %d/%d trajectories accepted (training on accepted only).",
        len(accepted), len(reader.trajectories),
    )
    entity_linker = EntityLinker(cache_path=resolve_entity_cache_path())
    logger.info("Phase2 link_confidence: EntityLinker cache=%s", resolve_entity_cache_path())
    prov = _build_samples_accepted_only(
        reader,
        binary_labels_only=cfg.binary_labels_only,
        entity_linker=entity_linker,
    )
    if not prov:
        raise ValueError(f"No step samples found in accepted trajectories of {cfg.silver_path}")
    samples = [p.sample for p in prov]

    # ---- Logprob pre-pass ------------------------------------------------
    logger.info("Logprob pre-pass over %d steps using %s", len(samples), model_path(cfg.base_model))
    logprob_means = compute_step_logprobs(
        samples,
        base_model_id=model_path(cfg.base_model),
        device=cfg.device,
        dtype=cfg.logprob_dtype,
        max_length=cfg.max_length,
    )
    # Persist logprobs back into the exact (trajectory, step) via provenance
    # (fix #5 — no fragile parallel counter that desyncs under binary_labels_only).
    for flat_idx, p in enumerate(prov):
        lp = [float(logprob_means[flat_idx])]
        accepted[p.traj_idx].steps[p.step_idx].token_logprobs = lp
        samples[flat_idx].semantic_entropy = entropy_from_logprobs(lp)

    enriched_path = out_dir / "silver_with_logprobs.jsonl"
    SilverDatasetReader.write_jsonl(enriched_path, reader.trajectories)
    logger.info("Wrote enriched silver data to %s", enriched_path)

    # ---- Model assembly --------------------------------------------------
    base, tokenizer = _build_base_model(cfg)
    hidden_size = getattr(base.config, "hidden_size", None) or base.config.to_dict().get("hidden_size", 4096)
    prm_head = PRMHead(hidden_size=hidden_size, n_classes=3).to(device=cfg.device, dtype=torch.float32)
    alpha_gate = AlphaGate().to(device=cfg.device, dtype=torch.float32)
    text_reward_head: Optional[nn.Sequential] = None
    if cfg.train_text_reward_head:
        text_reward_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh()).to(
            device=cfg.device, dtype=torch.float32
        )

    # ---- Optimiser -------------------------------------------------------
    trainable = list(filter(lambda p: p.requires_grad, base.parameters()))
    params: List[torch.nn.Parameter] = trainable + list(prm_head.parameters()) + list(alpha_gate.parameters())
    if text_reward_head is not None:
        params += list(text_reward_head.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    calibration = AlphaCalibrationLoss(weight=cfg.calibration_weight)
    text_mse = nn.MSELoss()

    # ---- DataLoader ------------------------------------------------------
    ds = _StepDataset(samples, tokenizer=tokenizer, max_length=cfg.max_length)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, pad_token_id=tokenizer.pad_token_id),
    )

    base.train()
    prm_head.train()
    alpha_gate.train()
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
            coverage = batch["coverage"].to(cfg.device)
            entropy_real = batch["semantic_entropy"].to(cfg.device)

            outputs = base(input_ids=input_ids, attention_mask=attention)
            last_hidden = _last_nonpad_hidden(outputs.last_hidden_state, attention)  # fix #4
            logits = prm_head(last_hidden)
            loss_prm = ce(logits, labels_class)

            # α-gate uses real semantic_entropy + step-level link_confidence.
            # ``coverage`` now carries the continuous per-step link_confidence
            # (Finding 2), NOT a thresholded copy of the calibration target.
            link_confidence = coverage.clamp(0.0, 1.0)
            alpha = alpha_gate(density, link_confidence, entropy_real)
            # Non-degenerate target (fix #2): calibrate α toward "the KG renders a
            # verdict on this step" = label is not NEUTRAL. This is independent of
            # the three gate inputs, so the gate can no longer trivially copy a
            # feature into the target.
            kg_has_verdict = (labels_class != 1).float()
            loss_cal = calibration(alpha, kg_has_verdict)

            loss = loss_prm + loss_cal
            if text_reward_head is not None:
                tr = text_reward_head(last_hidden).squeeze(-1)
                # Target: binary_quality ∈ {-1, +1} for this trajectory; broadcast per-step.
                # Approximate via labels_class — positive (2) or negative (0) drives ±1.
                tr_target = torch.where(
                    labels_class == 2,
                    torch.ones_like(tr),
                    torch.where(labels_class == 0, -torch.ones_like(tr), torch.zeros_like(tr)),
                )
                loss_text = text_mse(tr, tr_target)
                loss = loss + cfg.text_reward_lr * loss_text  # tiny multiplier; head is auxiliary
            loss = loss / cfg.grad_accum
            loss.backward()
            if (step_count + 1) % cfg.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
            step_count += 1

            if step_count % 50 == 0:
                total_loss = float(loss.item()) * cfg.grad_accum
                record: Dict[str, float] = {
                    "epoch": float(epoch),
                    "step": float(step_count),
                    "loss": total_loss,
                    "prm": float(loss_prm.item()),
                    "cal": float(loss_cal.item()),
                }
                if text_reward_head is not None:
                    record["text"] = float(loss_text.item())
                history.append(record)
                if text_reward_head is not None:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f (prm=%.4f, cal=%.4f, text=%.4f)",
                        epoch,
                        step_count,
                        total_loss,
                        loss_prm.item(),
                        loss_cal.item(),
                        loss_text.item(),
                    )
                else:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f (prm=%.4f, cal=%.4f)",
                        epoch,
                        step_count,
                        total_loss,
                        loss_prm.item(),
                        loss_cal.item(),
                    )

    # ---- Save -------------------------------------------------------------
    base.eval()
    prm_head.eval()
    alpha_gate.eval()
    if text_reward_head is not None:
        text_reward_head.eval()

    prm_dir = out_dir / "prm_head"
    prm_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(base, "save_pretrained"):
        base.save_pretrained(prm_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(prm_dir)
    torch.save(prm_head.state_dict(), out_dir / "prm_head" / "prm_head.pt")
    torch.save(alpha_gate.state_dict(), out_dir / "alpha_gate.pt")
    if text_reward_head is not None:
        head_path = Path(cfg.text_reward_path or (out_dir / "text_reward_head.pt"))
        head_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(text_reward_head.state_dict(), head_path)

    history_path = out_dir / "history.jsonl"
    with open(history_path, "w", encoding="utf-8") as fh:
        for row in history:
            fh.write(json.dumps(row) + "\n")
    logger.info("Wrote training history (%d points) to %s", len(history), history_path)

    dump_manifest(
        out_dir,
        extra={
            "phase": "phase2_prm",
            "silver_path": str(cfg.silver_path),
            "enriched_silver": str(enriched_path),
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "seed": cfg.seed,
            "alpha_W": alpha_gate.W.data.cpu().tolist(),
            "alpha_b": float(alpha_gate.b.data.cpu().item()),
            "alpha_tau": float(alpha_gate.tau.cpu().item()),
            "history_tail": history[-5:],
            "history_path": str(history_path),
            "history_points": len(history),
        },
    )
    logger.info("Phase 2 complete. Outputs under %s", out_dir)
    return {
        "output_dir": str(out_dir),
        "alpha_gate_path": str(out_dir / "alpha_gate.pt"),
        "prm_head_dir": str(prm_dir),
        "enriched_silver": str(enriched_path),
        "history_path": str(history_path),
        "text_reward_path": str(Path(cfg.text_reward_path or (out_dir / "text_reward_head.pt")))
        if text_reward_head is not None
        else None,
    }
