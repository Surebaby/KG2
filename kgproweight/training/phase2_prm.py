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
from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
)
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
    label: float                     # r_kg in [-1, 1] (continuous, not just ±1/0)
    label_class: int                 # 0 / 1 / 2 — index for cross-entropy
    kg_subgraph: List[tuple]
    coverage: float                  # holds step-level link_confidence (Finding 2)
    binary_quality: int              # +1 if accepted, -1 otherwise
    semantic_entropy: float          # populated after the logprob pre-pass
    # Conclusions of the PRECEDING steps, oldest first (fix #7). NEGATIVE is
    # assigned by ``PRMAnnotator._is_contradiction(conclusion, prev_conclusions)``,
    # so without them the label is not a function of the input: the same step
    # text is NEG when an earlier step said something incompatible and NEU when
    # it did not. Training on the step alone made NEG unlearnable — held-out NEG
    # recall was 0.152 vs 0.802 on seen data, with precision collapsing 1.000 →
    # 0.490. We prepend them to the encoder input so the contradiction signal is
    # actually present.
    prev_conclusions: List[str] = field(default_factory=list)


@dataclass
class _SampleWithProvenance:
    """A step sample plus where it came from, so the logprob pre-pass can write
    results back to the exact (trajectory, step) without a fragile parallel
    counter (fix #5 — the old flat_idx desynced under binary_labels_only)."""
    sample: _StepSample
    traj_idx: int
    step_idx: int


def _label_to_class(label: float) -> int:
    """Map a step label to the 3-way CE target.

    ``PRMAnnotator.label`` returns a CONTINUOUS r_kg = precision × relevance in
    (0, 1) for partially-verified citations, not just {-1, 0, +1}. The previous
    ``int(step.label)`` truncation sent every such value to 0 → NEUTRAL
    (int(0.5)=0, int(0.75)=0), silently discarding the PRM's entire partial-credit
    signal. Bucket instead: > +0.5 is positive evidence, < -0.5 is a
    contradiction, and the ambiguous middle stays NEUTRAL.
    """
    if label >= _POSITIVE_THRESHOLD:
        return 2
    if label <= -_POSITIVE_THRESHOLD:
        return 0
    return 1


# A step needs a majority of its citations verified AND relevant to count as
# positive evidence; below that the KG has not really rendered a verdict.
_POSITIVE_THRESHOLD = 0.5


def _build_samples_accepted_only(
    reader: SilverDatasetReader,
    *,
    binary_labels_only: bool = False,
    entity_linker: EntityLinker,
    accepted: Optional[List[SilverTrajectory]] = None,
) -> List[_SampleWithProvenance]:
    """Build step samples from ACCEPTED trajectories only (fix #1).

    ``coverage`` carries the STEP-LEVEL link_confidence computed with the same
    parser + scaffold filter + fn the PPO reward uses (Finding 2), so the α-gate
    sees the same feature distribution at training and inference time. Provenance
    (traj_idx into the returned ``accepted`` list, step_idx into ``traj.steps``)
    is recorded for the logprob write-back (fix #5).

    ``accepted`` lets the caller supply an already-narrowed list (e.g. the train
    fold) while ``reader`` still holds the whole file. The provenance indices are
    relative to whichever list is used, so the caller MUST write logprobs back
    through that same list — passing the fold here and indexing the full list
    there would scatter logprobs onto the wrong steps.
    """
    if accepted is None:
        accepted = reader.accepted()
    out: List[_SampleWithProvenance] = []
    for t_idx, traj in enumerate(accepted):
        quality = 1 if traj.accepted else -1
        # Walk steps in order so each sample can carry the conclusions that
        # precede it — the annotator's contradiction test reads exactly this.
        prev_conclusions: List[str] = []
        for s_idx, step in enumerate(traj.steps):
            text = step.text or ""
            parsed = parsed_step_from_silver_dict(step.to_dict(), fallback_index=s_idx)
            if not text.strip():
                # Still thread the conclusion through: a blank step is skipped as
                # a sample but must not break the chain for later steps.
                if parsed.intermediate_conclusion:
                    prev_conclusions.append(parsed.intermediate_conclusion)
                continue
            label = float(step.label)
            label_class = _label_to_class(label)
            skip = binary_labels_only and label_class == 1
            if not skip:
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
                            label_class=label_class,
                            kg_subgraph=list(traj.kg_subgraph),
                            coverage=float(link_conf),
                            binary_quality=quality,
                            semantic_entropy=0.0,
                            prev_conclusions=list(prev_conclusions),
                        ),
                        traj_idx=t_idx,
                        step_idx=s_idx,
                    )
                )
            if parsed.intermediate_conclusion:
                prev_conclusions.append(parsed.intermediate_conclusion)
    return out


def _step_samples_from_silver(reader: SilverDatasetReader, *, binary_labels_only: bool = False) -> List[_StepSample]:
    """Legacy builder kept for back-compat. Trains on ALL trajectories and fills
    ``coverage`` with the trajectory-level constant. Prefer
    ``_build_samples_accepted_only`` (used by run_phase2)."""
    out: List[_StepSample] = []
    for traj in reader:
        coverage = float(traj.metadata.get("coverage", 0.0))
        quality = 1 if traj.accepted else -1
        prev_conclusions: List[str] = []
        for s_idx, step in enumerate(traj.steps):
            text = step.text or ""
            parsed = parsed_step_from_silver_dict(step.to_dict(), fallback_index=s_idx)
            if not text.strip():
                if parsed.intermediate_conclusion:
                    prev_conclusions.append(parsed.intermediate_conclusion)
                continue
            label = float(step.label)
            label_class = _label_to_class(label)
            if not (binary_labels_only and label_class == 1):
                out.append(
                    _StepSample(
                        text=text,
                        label=label,
                        label_class=label_class,
                        kg_subgraph=list(traj.kg_subgraph),
                        coverage=coverage,
                        binary_quality=quality,
                        semantic_entropy=0.0,
                        prev_conclusions=list(prev_conclusions),
                    )
                )
            if parsed.intermediate_conclusion:
                prev_conclusions.append(parsed.intermediate_conclusion)
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
    # Deliberately ``s.text``, NOT ``build_prm_input(s)``: this feeds the α-Gate's
    # semantic-entropy feature, which must measure the model's uncertainty over
    # the step it generated. At PPO time ``compute_features`` receives the
    # rollout's own logprobs with no prior-conclusion prefix, so conditioning on
    # one here would shift the feature distribution between training and
    # inference. The prefix belongs in the PRM classifier input only.
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

    # Free the pre-pass model BEFORE the trainer loads its own copy. Without
    # this, run_phase2 holds two full bf16 Llama-3-8B (~32 GB) at once and OOMs
    # on a 24 GB card.
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


# How many preceding conclusions to prepend. The annotator compares against
# ALL prior conclusions, but trajectories are capped at 7 steps and the last few
# carry the contradictions that matter; 6 covers every step of a max-length
# trajectory while keeping the prefix short.
_MAX_PREV_CONCLUSIONS = 6


def build_prm_input(sample: _StepSample, max_prev: int = _MAX_PREV_CONCLUSIONS) -> str:
    """Render the encoder input for one step, including prior conclusions.

    The NEGATIVE class is defined relationally — ``_is_contradiction`` checks the
    step's conclusion against earlier ones — so a step in isolation does not
    determine its own label. Feeding only ``sample.text`` made NEG unlearnable
    (held-out recall 0.152). Prefixing the prior conclusions puts the deciding
    evidence in the input.

    Kept as a module-level function, not a Dataset method, so evaluation and
    inference build the string exactly the same way.
    """
    prev = [c.strip() for c in sample.prev_conclusions if c and c.strip()]
    if not prev:
        return "[No prior conclusions]\n\n[Current Step]\n" + sample.text
    if max_prev > 0:
        prev = prev[-max_prev:]
    lines = ["[Prior Conclusions]"]
    lines.extend("- %s" % c for c in prev)
    lines.append("")
    lines.append("[Current Step]")
    lines.append(sample.text)
    return "\n".join(lines)


class _StepDataset(Dataset):
    def __init__(self, samples: Sequence[_StepSample], tokenizer, max_length: int = 1024) -> None:
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Truncate from the LEFT so an over-long input loses the oldest prior
        # conclusions rather than the current step. Right truncation would cut
        # the step being classified — and ``_last_nonpad_hidden`` reads the final
        # token, so the head would be pooling over a severed prefix.
        try:
            self.tokenizer.truncation_side = "left"
        except Exception:  # pragma: no cover — exotic tokenizers
            logger.warning("Could not set truncation_side='left' on tokenizer.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        enc = self.tokenizer(
            build_prm_input(s),
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
    # One reasoning step plus its prior conclusions per sample, NOT a prompt.
    # Measured with the prefix over 13,374 accepted steps: p99=242, p99.9=701,
    # max=1386. Batches pad to their longest member, so an oversized cap lets one
    # long step blow up a whole batch's activation footprint — but at batch_size=4
    # the >512 tail is only 0.19% of samples, so raising the cap to 1024 leaves
    # the mean batch-max effectively unchanged (130 -> 133 tokens) while sparing
    # the 26 samples whose median length is 749 from losing a third of their text.
    max_length: int = 1024
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    calibration_weight: float = 0.1
    # Re-weight the 3-way CE by inverse class frequency. NEG is ~4% of accepted
    # steps, so an unweighted loss lets the head trade NEG away for 4 points of
    # accuracy — which it did (held-out NEG recall 0.152). Weights are computed
    # from the actual label histogram, normalised to mean 1.0 so the loss stays
    # on its previous scale and ``calibration_weight`` keeps its meaning.
    class_weighted_loss: bool = True
    # Cap on any single class weight after normalisation. Without it a rare class
    # in a small subset (e.g. a 200-sample smoke run) can get a weight of 50+ and
    # dominate the gradient.
    max_class_weight: float = 10.0
    train_text_reward_head: bool = True
    text_reward_lr: float = 1.0e-4
    text_reward_path: Optional[str] = None  # output path for the head
    logprob_dtype: str = "bf16"
    # Batch for the logprob pre-pass, which is the memory-heaviest part of Phase 2
    # per token: it materialises (B, L, 128256) logits, then a float32 copy, then
    # log_softmax's own copy. At B=4 that is ~2.6 GB transient at L=512 and
    # ~5.3 GB at L=1024, on top of the 16 GB of bf16 weights. Fine on the 96 GB
    # box (measured peak 46.5 GB for the whole phase); keep B small on 24 GB.
    logprob_batch_size: int = 4
    gradient_checkpointing: bool = True
    binary_labels_only: bool = False
    # Fold to train on. ``None`` reproduces the historical whole-file behaviour
    # of every run before the split existed; set "train" to hold val/test back.
    # The split is a pure function of question text + accepted flag + seed (see
    # kgproweight.data.silver_split), so it is recomputed identically here, in
    # Phase 3, and in evaluation without materialising separate files.
    split: Optional[str] = None
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    # Defaults to ``seed`` when None so the split moves with the run's seed only
    # if the caller intends it to. Pinning it separately means a seed sweep over
    # training randomness does not also reshuffle the held-out set, which would
    # make the sweep's variance uninterpretable.
    split_seed: Optional[int] = DEFAULT_SPLIT_SEED
    extra: Dict[str, Any] = field(default_factory=dict)

    def build_split_spec(self):
        from kgproweight.data.silver_split import SplitSpec

        return SplitSpec(
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed if self.split_seed is None else self.split_seed,
        )


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

    # A 24 GB card cannot hold bf16 Llama-3-8B (16 GB) plus full activations for
    # 32 layers at seq-len 2048. Recomputing activations in the backward pass
    # trades ~30% step time for several GB, which is what makes the run fit.
    if cfg.gradient_checkpointing:
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        # LoRA freezes the base weights, so without this the checkpointed
        # segments have no grad-requiring input and autograd silently skips
        # recomputation, leaving the adapters without gradients.
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        inner = getattr(base, "config", None)
        if inner is not None and hasattr(inner, "use_cache"):
            inner.use_cache = False
        logger.info("Gradient checkpointing enabled (use_reentrant=False).")
    return base, tokenizer


def run_phase2(cfg: Phase2Config) -> Dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    logger.info("Loading silver data from %s", cfg.silver_path)
    # Read the WHOLE file regardless of cfg.split. The enriched
    # silver_with_logprobs.jsonl written below is the input to Phase 3, so
    # filtering at load time would hand Phase 3 a file containing only Phase 2's
    # train fold — quietly deleting the val/test trajectories it needs. The fold
    # is applied to the training samples only, a few lines down.
    reader = SilverDatasetReader(cfg.silver_path)
    split_spec = cfg.build_split_spec()

    # Samples are built and the logprob pre-pass runs over ALL accepted
    # trajectories, not just the train fold. token_logprobs is a feature read off
    # the FROZEN base model — it depends on neither the labels nor the PRM head,
    # so computing it for val/test leaks nothing. Skipping them would instead
    # leave those trajectories with token_logprobs=None in the enriched file,
    # which is Phase 3's input, and Phase 3 would silently fall back to a
    # constant semantic_entropy for exactly the held-out data.
    accepted_all = reader.accepted()
    if not accepted_all:
        raise ValueError(f"No accepted trajectories in {cfg.silver_path}")
    entity_linker = EntityLinker(cache_path=resolve_entity_cache_path())
    logger.info("Phase2 link_confidence: EntityLinker cache=%s", resolve_entity_cache_path())
    prov = _build_samples_accepted_only(
        reader,
        binary_labels_only=cfg.binary_labels_only,
        entity_linker=entity_linker,
        accepted=accepted_all,
    )
    if not prov:
        raise ValueError(f"No step samples found in accepted trajectories of {cfg.silver_path}")
    all_samples = [p.sample for p in prov]

    # ---- Logprob pre-pass ------------------------------------------------
    logger.info("Logprob pre-pass over %d steps using %s", len(all_samples), model_path(cfg.base_model))
    logprob_means = compute_step_logprobs(
        all_samples,
        base_model_id=model_path(cfg.base_model),
        device=cfg.device,
        dtype=cfg.logprob_dtype,
        batch_size=cfg.logprob_batch_size,
        max_length=cfg.max_length,
    )
    # Persist logprobs back into the exact (trajectory, step) via provenance
    # (fix #5 — no fragile parallel counter that desyncs under binary_labels_only).
    # traj_idx indexes accepted_all, the same list passed to the builder above.
    for flat_idx, p in enumerate(prov):
        lp = [float(logprob_means[flat_idx])]
        accepted_all[p.traj_idx].steps[p.step_idx].token_logprobs = lp
        all_samples[flat_idx].semantic_entropy = entropy_from_logprobs(lp)

    enriched_path = out_dir / "silver_with_logprobs.jsonl"
    SilverDatasetReader.write_jsonl(enriched_path, reader.trajectories)
    logger.info("Wrote enriched silver data to %s", enriched_path)

    # ---- Fold selection (AFTER feature extraction, BEFORE any gradient) ----
    split_info: Dict[str, Any] = {"split": cfg.split}
    if cfg.split is None:
        samples = all_samples
        logger.info(
            "Phase2: %d/%d trajectories accepted, %d step samples. NO split — "
            "val/test are NOT held back, so any accuracy measured on this data is "
            "in-sample.",
            len(accepted_all), len(reader.trajectories), len(samples),
        )
    else:
        from kgproweight.data.silver_split import (
            SPLIT_NAMES,
            assign_split,
            summarize_split,
        )

        if cfg.split not in SPLIT_NAMES:
            raise ValueError(f"cfg.split must be one of {SPLIT_NAMES}, got {cfg.split!r}")
        # Named split_counts, not counts: the class-weight block further down
        # binds `counts` to the label histogram.
        split_counts = summarize_split(reader.splits(split_spec), split_spec)
        # Fold per accepted trajectory, then keep the samples whose trajectory is
        # in the requested fold. Going through provenance rather than re-deriving
        # from the sample keeps this exact under binary_labels_only, which drops
        # NEU samples and so breaks any positional correspondence.
        traj_fold = [assign_split(t, split_spec) for t in accepted_all]
        keep = [i for i, p in enumerate(prov) if traj_fold[p.traj_idx] == cfg.split]
        samples = [all_samples[i] for i in keep]
        n_traj_fold = sum(1 for f in traj_fold if f == cfg.split)
        logger.info(
            "Phase2 split seed=%d val=%.3f test=%.3f -> trajectories train %d / val %d / "
            "test %d (accepted %d / %d / %d)",
            split_spec.seed, split_spec.val_ratio, split_spec.test_ratio,
            split_counts.n["train"], split_counts.n["val"], split_counts.n["test"],
            split_counts.n_accepted["train"], split_counts.n_accepted["val"],
            split_counts.n_accepted["test"],
        )
        logger.info(
            "Phase2: training on fold %r — %d/%d accepted trajectories, %d/%d step samples.",
            cfg.split, n_traj_fold, len(accepted_all), len(samples), len(all_samples),
        )
        if not samples:
            raise ValueError(
                f"No step samples in split={cfg.split!r} of {cfg.silver_path}"
            )
        split_info.update(
            seed=split_spec.seed,
            val_ratio=split_spec.val_ratio,
            test_ratio=split_spec.test_ratio,
            counts=split_counts.as_dict(),
            n_samples_in_fold=len(samples),
            n_samples_total=len(all_samples),
        )

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

    class_weight: Optional[torch.Tensor] = None
    counts = [0, 0, 0]
    for s in samples:
        counts[s.label_class] += 1
    logger.info(
        "Label histogram: NEG=%d (%.2f%%) NEU=%d (%.2f%%) POS=%d (%.2f%%)",
        counts[0], 100.0 * counts[0] / len(samples),
        counts[1], 100.0 * counts[1] / len(samples),
        counts[2], 100.0 * counts[2] / len(samples),
    )
    if cfg.class_weighted_loss:
        # Inverse frequency, mean-normalised to 1.0, then clamped. Absent classes
        # get weight 1.0 (they contribute no loss terms anyway).
        raw = [len(samples) / (3.0 * c) if c > 0 else 1.0 for c in counts]
        scale = sum(raw) / 3.0
        w = [min(r / scale, cfg.max_class_weight) for r in raw]
        class_weight = torch.tensor(w, dtype=torch.float32, device=cfg.device)
        logger.info("Class-weighted CE: NEG=%.3f NEU=%.3f POS=%.3f", w[0], w[1], w[2])
    ce = nn.CrossEntropyLoss(weight=class_weight)
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
            # Which fold this checkpoint saw. Without it there is no way to tell,
            # months later, whether a number came from a run that held val/test
            # back — the difference between a held-out result and an in-sample one.
            "split_info": split_info,
            "max_length": cfg.max_length,
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
        "split_info": split_info,
        "text_reward_path": str(Path(cfg.text_reward_path or (out_dir / "text_reward_head.pt")))
        if text_reward_head is not None
        else None,
    }
