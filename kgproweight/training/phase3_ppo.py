"""Phase 3b — PPO + GAE + Critic + Reference Model (default on Pro 6000 96 GB).

Fixes bugs #8 and #9: the legacy script used a single scalar reward per
trajectory and never invoked the critic head. Here we run a full TRL
``PPOTrainer`` with:

- a frozen reference model (the Phase 3a SFT checkpoint),
- a policy model (also initialised from SFT, then PEFT-LoRA-tuned),
- a value head attached to the policy (TRL provides it in
  ``AutoModelForCausalLMWithValueHead``),
- a per-step composite reward function via
  :class:`kgproweight.training.reward_function.KGProWeightRewardFunction`,
- per-token reward shaping that places the step reward on the last token
  of the corresponding ``[Step N]`` span.

We also support an ``alpha_override`` for the alpha-ablations
(``α=0`` / ``α=0.5`` / ``α=1``) by passing the value straight to the
reward function.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from kgproweight.data.parsers import parse_steps
from kgproweight.data.prompts import build_rl_messages, build_sft_messages
from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
)
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.kg.training_question_kg import (
    apply_training_question_kg,
    read_question_kg_records,
)
from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.kg.wikidata_retriever import _QA_RELATION_FILTER, WikidataSubgraphRetriever
from kgproweight.reward.alpha_gate import AlphaGate
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.reward.proofkg_process import (
    is_automatic_proofkg,
    is_identity_safe_automatic_proofkg,
)
from kgproweight.reward.text_reward_model import build_text_reward_model
from kgproweight.training.reward_function import (
    KGProWeightRewardFunction,
    RewardSpec,
    source_gate_format_contract_version,
    step_spans_over_ids,
    validate_source_gate_format_contract,
    validate_source_gate_source_integrity,
    validate_source_gate_credit_config,
    load_source_gate_for_runtime,
)
from kgproweight.training.step_reward_ppo_trainer import StepRewardPPOTrainer
from kgproweight.training.ppo_tensorboard import log_ppo_batch, log_run_metadata
from kgproweight.training.ppo_tensorboard_runtime import create_ppo_writer, log_runtime
from kgproweight.utils.logging import (
    artifact_identity,
    dump_manifest,
    get_logger,
    prepare_new_run_dir,
)
from kgproweight.utils.paths import index_dir, model_path
from kgproweight.utils.seed import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Phase3PPOConfig:
    silver_path: str
    output_dir: str
    base_model: str = "llama3-8B-instruct"
    sft_checkpoint: Optional[str] = None
    sft_selection_report_path: Optional[str] = None
    sft_replay_silver_path: Optional[str] = None
    sft_replay_split: Optional[str] = None
    alpha_gate_path: Optional[str] = None
    text_reward_backend: str = "auto"  # rearag | llama_head | auto | dummy
    text_reward_fallback_path: Optional[str] = None
    dtype: str = "bf16"
    seed: int = 42
    # Explicit successor contract; historical configurations retain their
    # original generation, EOS trimming and permissive reward alignment.
    runtime_contract_version: str = "legacy"  # legacy | v2

    # PPO
    learning_rate: float = 1.0e-5
    batch_size: int = 64
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2  # value-function clip range (TRL default 0.2)
    # KL anchor: 0.1 is the midpoint. 0.2 locked the policy to SFT
    # (only rephrasing, not changing answers); 0.05 was too loose (policy
    # dropped [Step N] structure to chase reward). Combined with
    # step_format_bonus, 0.1 gives room to change answers safely.
    kl_coef: float = 0.1
    gamma: float = 0.95
    lam: float = 0.95
    max_grad_norm: float = 1.0
    total_steps: int = 5000
    vf_coef: float = 0.5
    # TRL attaches a newly-created critic to an SFT-only checkpoint.  Historical
    # runs used nn.Linear's random initialisation and ValueHead's implicit 0.1
    # dropout.  ``zero`` starts from the neutral baseline V(s)=0 while remaining
    # fully trainable; it is an explicit experiment setting, not a silent change
    # to the historical default.
    value_head_init: str = "default"  # default | zero
    value_head_dropout: float = 0.1
    # Smoke-only rolling cost guard. Disabled when after_steps == 0. This is an
    # operational stop rule, not an evaluation protocol or a success criterion.
    health_guard_after_steps: int = 0
    health_guard_window: int = 15
    health_guard_min_valid_rate: float = 0.0
    health_guard_max_length_capped_frac: float = 1.0
    health_guard_max_mean_kl: float = 1.0e9
    # Adaptive-controller target KL (TRL's `target`), NOT the early-stop knob.
    target_kl: float = 8.0
    # Horizon for the adaptive KL controller.
    kl_horizon: float = 2000.0
    early_stopping: bool = False
    # EM bonus weight: when the predicted answer matches the gold AND the
    # trajectory is valid, the final step gets +outcome_weight.
    # R7: outcome is now CONDITIONAL on trajectory validity — no more
    # unconditional answer reward (see problem_and_solutions.md).
    outcome_weight: float = 8.0  # R9 v6: EM-dominant reward (aligned with YAML)
    text_reward_scale: float = 0.3  # R6: scale down ReaRAG text reward
    pure_em_reward: bool = False
    # Gold-free automatic-ProofKG branch.  All controls default off so existing
    # experiments retain their old Reward/Loss and validity protocol.
    proofkg_process_reward: bool = False
    proofkg_outcome_only_reward: bool = False
    proofkg_process_version: str = "v1"
    proofkg_process_weight: float = 1.0
    proofkg_f1_weight: float = 0.0
    proofkg_dynamic_validity: bool = False
    mixed_outcome_reward: bool = False
    mixed_text_reward: bool = False
    source_gated_reward_version: str = "disabled"
    source_gate_format_version: str = "v1"
    answer_format_reward_version: str = "legacy"
    source_gate_credit_version: str = "disabled"  # Explicit disabled/v1/v2; default preserves old artifacts.
    source_gate_mode: str = "learned"
    source_gate_calibration_path: Optional[str] = None
    proofkg_require_all_eligible: bool = False
    # Number of independently sampled responses per selected question.
    rollouts_per_prompt: int = 1
    # R7: minimum number of parsed [Step N] blocks for trajectory validity.
    # Trajectories with fewer steps cannot receive the outcome reward.
    min_valid_steps: int = 3
    # retraining_plan §9.4-3 / R-1b: step-shortfall penalty, anchored to
    # outcome_weight. 0.0 reproduces every pre-2026-08-22 run.
    shortfall_coef: float = 0.0
    # Measured against silver's min_steps (3), not min_valid_steps.
    target_steps: int = 3
    # ------------------------------------------------------------------
    # retraining_plan §9.4-1 (量纲/D2): R_Text DC removal.
    # ------------------------------------------------------------------
    # MEASURED r_text mean 0.6284 sd 0.1353 against r_kg mean 0.0896 sd 0.1166 --
    # a 7x gap in means on a nominally shared [-1,1] scale. The offset is
    # invisible to the policy gradient (advantage whitening removes it) but NOT
    # to the alpha channel, where it makes d r_total / d alpha = -0.148, i.e. the
    # reward pays the policy to LOWER alpha, hence to make the KG subgraph look
    # sparser. Centering removes the offset and neutralises that gradient.
    # False = every pre-2026-08-23 run, bit-for-bit. See CompositeRewardModel.
    center_text_reward: bool = False
    # EMA momentum for the baseline. 0.99 => effective window ~100 step
    # observations, se ~0.014 against the signal's own sd 0.135 (10%). A literal
    # within-batch mean (~12 samples at batch_size=4) would inject se 0.039, i.e.
    # 29% of the signal, and would make the reward batch-dependent in a way the
    # critic cannot represent.
    text_baseline_momentum: float = 0.99
    # R8: minimum characters of actual reasoning content per step. Empty
    # "Reasoning:\n" blocks are treated as invalid (content-aware gate).
    min_reasoning_chars: int = 20
    # R9: scale up per-step composite reward to cover KL token cost.
    # With KG online, max R_step ≈ 0.8; KL cost ≈ 5 per 100 tokens.
    # Scale ×5 brings max R_step to ~4.0 so multi-step reasoning is net positive.
    step_reward_scale: float = 0.5  # R9 v6: down-weighted (aligned with YAML)
    # Fraction of PPO *samples seen* replayed through a separate supervised CE
    # update on the matched full silver trajectory. Replay data never enters the
    # PPO rollout batch: putting a gold trace into a rollout prompt leaks the
    # answer, and the historical implementation paired that prompt with a random
    # question's RewardSpec. A fractional-credit scheduler below makes values
    # such as 0.10 exact in the long run even when batch_size=4.
    sft_replay_ratio: float = 0.10
    # Weight for the supervised replay CE update.
    sft_anchor_weight: float = 0.02  # λ: small, for format prior only
    # Legacy cadence, retained only to reproduce old runs when
    # sft_replay_ratio == 0. New formal runs use the ratio scheduler.
    sft_anchor_interval: int = 0
    log_with: Optional[str] = None
    # Save a recoverable adapter checkpoint every N trajectories seen (0 = only
    # at the end). Lets a collapsed run roll back to the last healthy step.
    save_every_steps: int = 256

    # Generation
    max_new_tokens: int = 256
    # PPO ROLLOUT SAMPLING MUST MATCH TRL's logprob recomputation distribution.
    # TRL's batched_forward_pass scores responses from RAW logits — i.e.
    # temperature=1.0, no top_p, no top_k. If we sample at temperature=0.7 /
    # top_p=0.9 (a sharper, truncated distribution) the recomputed logp_old is
    # not the distribution the tokens were actually drawn from, so the PPO ratio
    # baseline is wrong and the KL estimate drifts NEGATIVE (the 2026-06-23
    # symptom: KL swung 60→0.34→-20). Keep these at the no-op values and disable
    # top_k in the generate() call so rollout == scoring distribution.
    temperature: float = 1.0
    top_p: float = 1.0
    # R10 speed: how many prompts to decode in ONE generate() call. 1 reproduces
    # the old prompt-at-a-time loop exactly; higher batches the decode.
    #
    # Why this is the main speed lever: at batch_size=8 / max_new_tokens=256 the
    # serial loop runs ~8x256 = 2048 single-sample decode steps per optimiser
    # update, against 16 fwd+bwd passes in the PPO stage. Batching turns those
    # 2048 into ~256 batched steps.
    #
    # Memory: output_scores keeps one (chunk, vocab) fp32 logit tensor per
    # generated token. At vocab=128256 that is chunk x 128256 x 4 B = 0.5 MB per
    # prompt per token, so a chunk of 8 over 256 tokens is ~1 GB held until we
    # collapse it. That fits the 5.5 GB free, but the chunk knob exists so it can
    # be lowered without touching batch_size if a longer max_new_tokens or a
    # bigger batch ever makes it tight.
    rollout_chunk_size: int = 8
    max_input_length: int = 4096
    # R9 v3: cap parsed [Step N] blocks per rollout (reduced to save VRAM).
    max_steps: int = 5
    # B3: the prompt template orders Question → Passages → KG, and the tokenizer
    # right-truncates. With the package defaults (50 passages) the KG block — the
    # whole point of KG-grounding — gets truncated away before the model sees it
    # during rollouts. Cap passages so the KG always fits.
    # Unified with SFT/eval (DEFAULT_TOPK=15) so PPO rolls out in the SAME
    # passage context the policy is later evaluated in (no train/inference
    # mismatch). 15 passages + KG fit max_input_length=6144; on a 96GB card the
    # policy (8B) + frozen SFT ref + ReaRAG-9B reward co-reside (~50GB) with room
    # for activations at batch_size=8 / mini_batch_size=1 + gradient checkpointing.
    # If OOM: first lower batch/rollout chunk size, then passages. Keep the
    # cross-stage KG budget at 12 unless running a separately tracked ablation.
    ppo_max_passages: int = 15
    # The formal cross-stage budget is 12 and the CLI forwards it explicitly.
    # Prompt preparation now drops low-ranked passages when needed and refuses
    # any residual overflow, so this trailing KG block is never right-truncated.
    ppo_max_kg_triples: int = 12
    # Minimum retained after score thresholding. Hard-delete and predicate
    # quotas are never relaxed; this matches Phase 1, Phase 2, inference and the
    # question-KG index builder.
    ppo_min_kg_triples: int = 5
    prm_min_subgraph_for_verify: int = 3

    # LoRA on policy
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Ablation hooks
    alpha_override: Optional[float] = None
    binary_labels_only: bool = False
    # P1-1: use real per-step token logprobs for the α-gate's entropy feature.
    use_real_logprobs: bool = True

    # Fold to roll out on. Must match the fold Phase 2 / 3a used: PPO generates
    # from these questions, so a question in the val or test fold here has been
    # optimised against and is no longer held out for anything downstream.
    # ``None`` reproduces the pre-split behaviour of using the whole file, which
    # trains on the val/test folds. Since 2026-08-22 (retraining_plan §13-1) that
    # is a hard error unless ``split_allow_none`` is set explicitly -- a data leak
    # must not be reachable by omitting a flag. See :func:`_resolve_split`.
    split: Optional[str] = None
    # Opt-in escape hatch for deliberately reproducing a pre-split run.
    split_allow_none: bool = False

    # Pre-built question -> KG-triples index used for PROMPT-side KG injection.
    # 2026-08-22 (retraining_plan §10.3, R-2): made configurable. It used to be
    # hardcoded to question_kg_index_v2.json with a fallback to
    # question_kg_index.json; both were built over the DEV split, while PPO rolls
    # out over the silver TRAIN fold, so every question missed and all 9839
    # prompts silently took the fallback path. The miss also degraded r_kg,
    # because the same subgraph is the verification reference in RewardSpec.
    # None => keep the legacy two-name probe (for reproducing old runs).
    question_kg_index_path: Optional[str] = None
    # Refuse to train when the index misses more than this fraction of prompts.
    # 1.0 disables the check (legacy behaviour: warn only).
    max_kg_index_miss_rate: float = 1.0
    # Formal stored-silver runs require every indexed triple list to equal the
    # trajectory's stored KG, not merely share the same question set.
    require_exact_kg_index_alignment: bool = False
    # New identity-safe replacement for the legacy raw-question-text index.
    # Exactly one of this and question_kg_index_path is required for new runs.
    question_kg_records_path: Optional[str] = None
    min_question_kg_record_coverage: float = 1.0
    require_nonempty_question_kg_records: bool = False
    # Optional versioned passage input for a pre-frozen smoke rollout schedule.
    # Both paths are required together: partial overrides would silently mix
    # old and new retrieval distributions inside one PPO run.
    passage_overrides_path: Optional[str] = None
    rollout_schedule_path: Optional[str] = None
    rollout_sampling_weights_path: Optional[str] = None
    fixed_rollout_schedule_path: Optional[str] = None
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_rollout_indices(
    population_size: int,
    batch_size: int,
    rollouts_per_prompt: int,
    generator: torch.Generator,
    sampling_weights: Optional[Sequence[float]] = None,
) -> List[int]:
    """Sample independent responses, optionally grouped by identical prompt."""
    if population_size <= 0 or batch_size <= 0:
        raise ValueError("population_size and batch_size must be positive")
    k = int(rollouts_per_prompt)
    if k <= 0 or batch_size % k:
        raise ValueError(
            "rollouts_per_prompt must be positive and divide batch_size exactly; "
            f"got batch_size={batch_size}, rollouts_per_prompt={k}"
        )
    # Preserve the historical RNG call exactly when both features are disabled.
    if k == 1 and sampling_weights is None:
        return torch.randint(
            0, population_size, (batch_size,), generator=generator,
        ).tolist()
    if sampling_weights is None:
        anchors = torch.randint(
            0, population_size, (batch_size // k,), generator=generator,
        ).tolist()
    else:
        weights = torch.as_tensor(list(sampling_weights), dtype=torch.float64)
        if weights.numel() != population_size:
            raise ValueError(
                "sampling_weights length must equal population_size; "
                f"got {weights.numel()} and {population_size}"
            )
        if not torch.isfinite(weights).all() or bool((weights < 0).any()):
            raise ValueError("sampling_weights must be finite and non-negative")
        if float(weights.sum().item()) <= 0:
            raise ValueError("sampling_weights must have positive total mass")
        anchors = torch.multinomial(
            weights, batch_size // k, replacement=True, generator=generator,
        ).tolist()
    return anchors if k == 1 else [index for index in anchors for _ in range(k)]


def _select_rollout_batch_indices(
    *,
    population_size: int,
    batch_size: int,
    rollouts_per_prompt: int,
    generator: torch.Generator,
    sampling_weights: Optional[Sequence[float]] = None,
    fixed_indices: Optional[Sequence[int]] = None,
    offset: int = 0,
) -> List[int]:
    """Select one rollout batch, preferring an explicitly frozen schedule."""

    if fixed_indices:
        batch = [int(value) for value in fixed_indices[offset : offset + batch_size]]
        if len(batch) != batch_size:
            raise RuntimeError(
                "fixed rollout schedule ended inside a PPO batch at "
                f"trajectory {offset}"
            )
        return batch
    return _sample_rollout_indices(
        population_size,
        batch_size,
        rollouts_per_prompt,
        generator,
        sampling_weights=sampling_weights,
    )


def _load_rollout_sampling_weights(
    path: str | Path,
    trajectories: Sequence[Any],
) -> Tuple[List[float], Dict[str, Dict[str, Any]]]:
    """Load identity-safe prompt weights aligned to ``trajectories``.

    The file is deliberately separate from silver metadata so PPO-O and PPO-K
    can bind the exact same sampling distribution by hash.  Missing, duplicate,
    stale-question, or extra rows are fatal rather than silently renormalised.
    """
    weight_path = Path(path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"rollout_sampling_weights_path does not exist: {weight_path}")
    records: Dict[str, Dict[str, Any]] = {}
    with weight_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = question_key(str(row.get("dataset") or ""), str(row.get("qid") or ""))
            if key in records:
                raise ValueError(f"duplicate rollout sampling weight key at line {line_number}: {key}")
            weight = float(row.get("sampling_probability", -1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"invalid rollout sampling probability for {key}: {weight}")
            records[key] = dict(row)

    aligned: List[float] = []
    expected: set[str] = set()
    for trajectory in trajectories:
        dataset = str(
            trajectory.get("spec").metadata.get("dataset")
            if isinstance(trajectory, dict)
            else trajectory.dataset
        )
        qid = str(
            trajectory.get("spec").metadata.get("qid")
            if isinstance(trajectory, dict)
            else trajectory.qid
        )
        question = str(
            trajectory.get("spec").query
            if isinstance(trajectory, dict)
            else trajectory.question
        )
        key = question_key(dataset, qid)
        if key in expected:
            raise ValueError(f"PPO sampling population contains duplicate identity: {key}")
        expected.add(key)
        row = records.get(key)
        if row is None:
            raise ValueError(f"rollout sampling weights missing key: {key}")
        declared_hash = str(row.get("question_sha256") or "")
        if declared_hash and declared_hash != question_sha256(question):
            raise ValueError(f"rollout sampling question hash mismatch for {key}")
        aligned.append(float(row["sampling_probability"]))
    extras = sorted(set(records) - expected)
    if extras:
        raise ValueError(
            f"rollout sampling weights contain {len(extras)} identities outside the PPO population"
        )
    total = sum(aligned)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"rollout sampling probabilities must sum to 1.0, got {total:.12f}")
    return aligned, records


def _load_fixed_rollout_schedule(
    path: str | Path,
    trajectories: Sequence[Any],
    *,
    total_steps: int,
    rollouts_per_prompt: int,
    sampling_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Resolve a frozen qid schedule to population indices, fail-closed."""
    schedule_path = Path(path)
    if not schedule_path.is_file():
        raise FileNotFoundError(f"fixed_rollout_schedule_path does not exist: {schedule_path}")
    rows: List[Dict[str, Any]] = []
    with schedule_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != int(total_steps):
        raise ValueError(
            f"fixed rollout schedule has {len(rows)} rows, expected {total_steps}"
        )
    if [int(row.get("rollout_index", -1)) for row in rows] != list(range(1, total_steps + 1)):
        raise ValueError("fixed rollout schedule indices must be contiguous from 1")

    k = int(rollouts_per_prompt)
    if k <= 0 or int(total_steps) % k:
        raise ValueError(
            "fixed rollout schedule length must be divisible by "
            f"rollouts_per_prompt={k}; got total_steps={total_steps}"
        )

    population: Dict[str, int] = {}
    population_questions: Dict[str, str] = {}
    for index, trajectory in enumerate(trajectories):
        if isinstance(trajectory, dict):
            dataset = str(trajectory["spec"].metadata.get("dataset") or "")
            qid = str(trajectory["spec"].metadata.get("qid") or "")
        else:
            dataset, qid = str(trajectory.dataset), str(trajectory.qid)
        key = question_key(dataset, qid)
        if key in population:
            raise ValueError(f"PPO population contains duplicate identity: {key}")
        population[key] = index
        population_questions[key] = str(
            trajectory["spec"].query if isinstance(trajectory, dict) else trajectory.question
        )

    indices: List[int] = []
    for row in rows:
        key = question_key(str(row.get("dataset") or ""), str(row.get("qid") or ""))
        if key not in population:
            raise ValueError(f"fixed rollout schedule qid is outside PPO population: {key}")
        declared_hash = str(row.get("question_sha256") or "")
        if declared_hash and declared_hash != question_sha256(population_questions[key]):
            raise ValueError(f"fixed rollout schedule question hash mismatch: {key}")
        if sampling_records is not None:
            weight_row = sampling_records.get(key)
            if weight_row is None:
                raise ValueError(f"fixed rollout qid has no sampling record: {key}")
            if str(row.get("stratum") or "") != str(weight_row.get("stratum") or ""):
                raise ValueError(f"fixed rollout stratum mismatch: {key}")
        indices.append(population[key])
    for start in range(0, len(indices), k):
        if len(set(indices[start:start + k])) != 1:
            raise ValueError(f"fixed rollout schedule breaks K={k} grouping at row {start + 1}")
    return indices, rows


def _validate_mixed_reward_config(cfg: Phase3PPOConfig) -> None:
    """Fail closed on the frozen mixed-dataset outcome/process definition."""

    if cfg.source_gated_reward_version not in {"disabled", "v1"}:
        raise ValueError("source_gated_reward_version must be disabled or v1")
    source_gate_format_contract_version(cfg.source_gate_format_version)
    if cfg.answer_format_reward_version not in {"legacy", "v2"}:
        raise ValueError("answer_format_reward_version must be legacy or v2")
    if cfg.answer_format_reward_version == "v2" and (
        cfg.source_gated_reward_version != "v1"
        or cfg.source_gate_format_version != "v2"
        or cfg.source_gate_credit_version != "v2"
    ):
        raise ValueError("answer format reward v2 requires source-gated v1, format v2 and source credit v2")
    validate_source_gate_credit_config(cfg.source_gate_credit_version,
                                      cfg.source_gated_reward_version,
                                      cfg.source_gate_format_version)
    source_v1 = cfg.source_gated_reward_version == "v1"
    if source_v1:
        if cfg.source_gate_mode not in {"text", "fixed", "learned"}:
            raise ValueError("source_gate_mode must be text, fixed or learned")
        if not cfg.source_gate_calibration_path:
            raise ValueError("source-gated v1 requires source_gate_calibration_path")
        if not (cfg.mixed_outcome_reward and cfg.mixed_text_reward):
            raise ValueError("source-gated v1 requires mixed outcome and text rewards")
        if cfg.runtime_contract_version != "v2":
            raise ValueError("source-gated v1 requires runtime_contract_version=v2")
        if not cfg.proofkg_process_reward or cfg.proofkg_process_version != "v2_3":
            raise ValueError("source-gated v1 requires ProofKG process v2_3 in every arm")
        if cfg.center_text_reward:
            raise ValueError("source-gated v1 requires center_text_reward=false (frozen stats)")
        if cfg.proofkg_outcome_only_reward:
            raise ValueError("source-gated v1 cannot combine proofkg_outcome_only_reward")
    if cfg.mixed_text_reward and not cfg.mixed_outcome_reward:
        raise ValueError("mixed_text_reward requires mixed_outcome_reward=true")
    if not cfg.mixed_outcome_reward:
        return
    if cfg.alpha_gate_path is not None:
        raise ValueError(
            "mixed_outcome_reward requires alpha_gate_path=null; the historical "
            "alpha gate is not part of PPO-T/PPO-TK"
        )
    if cfg.alpha_override is not None:
        raise ValueError(
            "mixed_outcome_reward requires alpha_override=null; PPO-T/PPO-TK use "
            "the frozen additive reward without alpha mixing"
        )
    if not cfg.question_kg_records_path:
        raise ValueError(
            "mixed_outcome_reward requires identity-safe question_kg_records_path"
        )
    if cfg.pure_em_reward:
        raise ValueError("mixed_outcome_reward cannot be combined with pure_em_reward")
    if not math.isclose(float(cfg.outcome_weight), 4.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("mixed_outcome_reward requires outcome_weight=4.0")
    if not math.isclose(float(cfg.proofkg_f1_weight), 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("mixed_outcome_reward requires proofkg_f1_weight=0.10")
    if not cfg.proofkg_dynamic_validity:
        raise ValueError("mixed_outcome_reward requires proofkg_dynamic_validity=true")
    if cfg.mixed_text_reward:
        if str(cfg.text_reward_backend).lower() != "rearag":
            raise ValueError(
                "mixed_text_reward requires text_reward_backend=rearag (fail-hard); "
                "auto/llama_head/dummy fallback is forbidden"
            )
        if not math.isclose(
            float(cfg.text_reward_scale), 0.30, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("mixed_text_reward requires text_reward_scale=0.30")
        if not source_v1 and not cfg.center_text_reward:
            raise ValueError("mixed_text_reward requires center_text_reward=true")
        if not source_v1 and not math.isclose(
            float(cfg.text_baseline_momentum), 0.99, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("mixed_text_reward requires text_baseline_momentum=0.99")
    if cfg.proofkg_require_all_eligible:
        raise ValueError(
            "mixed_outcome_reward is designed for mixed eligibility and cannot "
            "set proofkg_require_all_eligible=true"
        )
    if cfg.proofkg_process_reward and cfg.proofkg_process_version not in {"v2_1", "v2_2", "v2_3"}:
        raise ValueError(
            "mixed PPO-K requires proofkg_process_version=v2_1, v2_2 or v2_3; legacy process "
            "scorers are forbidden on the mixed route"
        )
    if cfg.proofkg_process_reward and not math.isclose(
        float(cfg.proofkg_process_weight), 0.20, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("mixed PPO-TK requires proofkg_process_weight=0.20")


def _assert_mixed_text_backend(cfg: Phase3PPOConfig, text_reward: Any) -> None:
    """Fail closed if a mixed-text run did not build frozen ReaRAG.

    Config validation rejects ``auto``/``dummy`` before model allocation.  This
    second check catches a faulty builder or future fallback implementation that
    returns a different backend despite an explicit ``rearag`` request.
    """

    if not cfg.mixed_text_reward:
        return
    if (
        str(getattr(text_reward, "name", "")).lower() != "rearag"
        or bool(getattr(text_reward, "is_dummy", False))
    ):
        raise RuntimeError(
            "mixed_text_reward requires the frozen ReaRAG backend; fallback is forbidden"
        )
    if cfg.source_gated_reward_version == "v1":
        from kgproweight.reward.text_reward_model import RearagPromptScorer
        backend = getattr(text_reward, "backend", None)
        if not isinstance(backend, RearagPromptScorer):
            raise RuntimeError("source-gated v1 requires the actual RearagPromptScorer")
        if backend.model.training or any(p.requires_grad for p in backend.model.parameters()):
            raise RuntimeError("source-gated v1 requires frozen eval-mode ReaRAG")


def _mixed_reward_dataset_diagnostics(
    reward_infos: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate the mixed reward decomposition separately by dataset."""

    sums: Dict[str, Dict[str, Any]] = {}
    for info in reward_infos:
        row = info.get("mixed_reward")
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset") or "unknown")
        group = sums.setdefault(
            dataset,
            {
                "count": 0.0,
                "valid_count": 0.0,
                "proofkg_eligible_count": 0.0,
                "process_applied_count": 0.0,
                "outcome": 0.0,
                "text": 0.0,
                "process": 0.0,
                "total": 0.0,
                "text_step_count": 0.0,
                "text_raw_sum": 0.0,
                "text_baseline_preupdate_sum": 0.0,
                "text_centered_unclipped_sum": 0.0,
                "text_centered_sum": 0.0,
                "text_centered_abs_sum": 0.0,
                "text_clip_count": 0.0,
                "text_ema_baseline": 0.0,
                "text_ema_n_obs": 0.0,
                "em_matched_nonprimary_count": 0.0,
                "f1_matched_nonprimary_count": 0.0,
            },
        )
        group["count"] += 1.0
        group["valid_count"] += float(bool(info.get("trajectory_valid")))
        group["proofkg_eligible_count"] += float(bool(row.get("proofkg_eligible")))
        group["process_applied_count"] += float(bool(
            (info.get("proofkg_process") or {}).get("process_applied")
        ))
        group["em_matched_nonprimary_count"] += float(
            bool(row.get("outcome_em_matched_nonprimary"))
        )
        group["f1_matched_nonprimary_count"] += float(
            bool(row.get("outcome_f1_matched_nonprimary"))
        )
        for component in ("outcome", "text", "process", "total"):
            value = float(row.get(component, 0.0))
            if not math.isfinite(value):
                raise RuntimeError(
                    f"non-finite mixed reward telemetry: dataset={dataset} "
                    f"component={component} value={value}"
                )
            group[component] += value
        raw_scores = [float(value) for value in row.get("text_raw_step_scores", [])]
        baselines = [
            float(value) for value in row.get("text_baseline_before_step", [])
        ]
        centered = [
            float(value)
            for value in row.get("text_centered_clipped_step_scores", [])
        ]
        if not (len(raw_scores) == len(baselines) == len(centered)):
            raise RuntimeError(
                f"mixed ReaRAG telemetry lengths differ for dataset={dataset}"
            )
        if any(
            not math.isfinite(value)
            for value in raw_scores + baselines + centered
        ):
            raise RuntimeError(
                f"non-finite mixed ReaRAG telemetry for dataset={dataset}"
            )
        group["text_step_count"] += len(raw_scores)
        group["text_raw_sum"] += sum(raw_scores)
        group["text_baseline_preupdate_sum"] += sum(baselines)
        residuals = [float(value) for value in (
            (info.get("source_gate") or {}).get("text_normalized_unclipped_steps")
            if (info.get("source_gate") or {}).get("text_normalized_unclipped_steps") is not None
            else [raw - baseline for raw, baseline in zip(raw_scores, baselines)]
        )]
        if len(residuals) != len(raw_scores) or any(not math.isfinite(value) for value in residuals):
            raise RuntimeError("mixed text normalization telemetry contract violation")
        group["text_centered_unclipped_sum"] += sum(residuals)
        group["text_centered_sum"] += sum(centered)
        group["text_centered_abs_sum"] += sum(abs(value) for value in centered)
        text_v2 = (info.get("source_gate") or {}).get("text_normalization_v2")
        if text_v2 is not None:
            group["text_clip_count"] += float(text_v2["hard_clip_frac"]) * len(raw_scores)
            group["text_v2_step_count"] = group.get("text_v2_step_count", 0) + len(raw_scores)
            group["text_soft_saturation_count"] = group.get("text_soft_saturation_count", 0) + float(text_v2["soft_saturation_frac"]) * len(raw_scores)
            group["text_raw_z_outside_unit_count"] = group.get("text_raw_z_outside_unit_count", 0) + float(text_v2["raw_z_outside_unit_frac"]) * len(raw_scores)
        else:
            group["text_clip_count"] += sum(abs(value) > 1.0 for value in residuals)
        # The EMA is global and causal; retain the state observed after the
        # latest row from this dataset in the current batch.
        group["text_ema_baseline"] = float(row.get("text_ema_baseline", 0.0))
        group["text_ema_n_obs"] = float(row.get("text_ema_n_obs", 0.0))

    result: Dict[str, Dict[str, Any]] = {}
    for dataset in sorted(sums):
        group = sums[dataset]
        count = int(group["count"])
        text_step_count = int(group["text_step_count"])
        result[dataset] = {
            "count": count,
            "valid_count": int(group["valid_count"]),
            "valid_rate": group["valid_count"] / count,
            "proofkg_eligible_count": int(group["proofkg_eligible_count"]),
            "proofkg_eligible_rate": group["proofkg_eligible_count"] / count,
            "process_applied_count": int(group["process_applied_count"]),
            "em_matched_nonprimary_count": int(
                group["em_matched_nonprimary_count"]
            ),
            "f1_matched_nonprimary_count": int(
                group["f1_matched_nonprimary_count"]
            ),
            "text_step_count": text_step_count,
            "text_raw_step_mean": (
                group["text_raw_sum"] / text_step_count if text_step_count else None
            ),
            "text_baseline_preupdate_step_mean": (
                group["text_baseline_preupdate_sum"] / text_step_count
                if text_step_count else None
            ),
            "text_centered_unclipped_step_mean": (
                group["text_centered_unclipped_sum"] / text_step_count
                if text_step_count else None
            ),
            "text_centered_step_mean": (
                group["text_centered_sum"] / text_step_count
                if text_step_count else None
            ),
            "text_centered_abs_mean": (
                group["text_centered_abs_sum"] / text_step_count
                if text_step_count else None
            ),
            "text_clip_frac": (
                group["text_clip_count"] / text_step_count
                if text_step_count else None
            ),
            "text_ema_baseline": group["text_ema_baseline"],
            "text_ema_n_obs": int(group["text_ema_n_obs"]),
            **({"text_v2_step_count": int(group["text_v2_step_count"]),
                "text_soft_saturation_frac": group["text_soft_saturation_count"] / group["text_v2_step_count"],
                "text_raw_z_outside_unit_frac": group["text_raw_z_outside_unit_count"] / group["text_v2_step_count"]}
               if group.get("text_v2_step_count") else {}),
            **{
                f"{component}_mean": group[component] / count
                for component in ("outcome", "text", "process", "total")
            },
        }
    return result


def _source_gate_batch_diagnostics(reward_infos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records = [info["source_gate"] for info in reward_infos if info.get("source_gate")]
    if not records:
        return {}
    result: Dict[str, Any] = {"source_gate_records": records}
    for key in ("alpha_effective", "m_graph", "graph_raw", "graph_normalized",
                "text_normalized", "text_component", "graph_component"):
        values = [float(row[key]) for row in records]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite source gate telemetry: {key}")
        result[f"source_gate_{key}_mean"] = sum(values) / len(values)
    feature_names = tuple(records[0]["features"]["values"])
    if any(set(row["features"]["values"]) != set(feature_names) for row in records):
        raise RuntimeError("source gate batch mixes different feature contracts")
    for name in feature_names:
        result[f"source_gate_feature_{name}_mean"] = sum(
            float(row["features"]["values"][name]) for row in records
        ) / len(records)
    text_v2 = [row["text_normalization_v2"] for row in records if row.get("text_normalization_v2")]
    if text_v2:
        total_v2_steps = sum(row["step_count"] for row in text_v2)
        for key in ("hard_clip_frac", "soft_saturation_frac", "raw_z_outside_unit_frac"):
            result[f"source_gate_text_v2_{key}"] = sum(row[key] * row["step_count"] for row in text_v2) / total_v2_steps
    result["source_gate_text_max_tokens"] = max(
        (int(step["total_tokens"]) for row in records
         for step in (row.get("token_budget") or {}).get("step_lengths", [])), default=0,
    )
    result["source_gate_text_truncated_tokens"] = sum(
        int((row.get("token_budget") or {}).get("truncated_tokens", 0)) for row in records
    )
    return result


def _mixed_text_batch_diagnostics(
    by_dataset: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Collapse per-dataset ReaRAG diagnostics for top-level curves."""

    total_steps = sum(int(row["text_step_count"]) for row in by_dataset.values())
    latest = max(
        by_dataset.values(), key=lambda row: int(row["text_ema_n_obs"]),
        default=None,
    )

    def weighted_mean(field: str) -> Optional[float]:
        if not total_steps:
            return None
        return sum(
            float(row[field]) * int(row["text_step_count"])
            for row in by_dataset.values()
            if row[field] is not None
        ) / total_steps

    return {
        "mixed_text_step_count": total_steps,
        "mixed_text_raw_step_mean": weighted_mean("text_raw_step_mean"),
        "mixed_text_baseline_preupdate_step_mean": weighted_mean(
            "text_baseline_preupdate_step_mean"
        ),
        "mixed_text_centered_unclipped_step_mean": weighted_mean(
            "text_centered_unclipped_step_mean"
        ),
        "mixed_text_centered_step_mean": weighted_mean("text_centered_step_mean"),
        "mixed_text_centered_abs_mean": weighted_mean("text_centered_abs_mean"),
        "mixed_text_clip_frac": weighted_mean("text_clip_frac"),
        "mixed_text_ema_baseline": (
            float(latest["text_ema_baseline"]) if latest is not None else None
        ),
        "mixed_text_ema_n_obs": (
            int(latest["text_ema_n_obs"]) if latest is not None else 0
        ),
    }


def _validate_v21_execution_preflight(
    trajectories: Sequence[Any], *, version: str = "v2_1",
) -> Dict[str, int]:
    """Require v2.1 execution only on exact-join complete ProofKG rows.

    HotpotQA, MuSiQue and ordinary 2Wiki rows are deliberately outcome-only and
    may carry empty/incomplete records.  Requiring execution on every row would
    reject the valid mixed dataset; failing to check eligible rows would defer a
    malformed proof crash until after the GPU and Experiment ID were allocated.
    """

    eligible = 0
    missing_execution: List[str] = []
    for trajectory in trajectories:
        runtime = trajectory.metadata.get("question_kg_runtime") or {}
        if not is_identity_safe_automatic_proofkg(
            runtime,
            trajectory.kg_subgraph,
            dataset=trajectory.dataset,
            qid=trajectory.qid,
        ):
            continue
        eligible += 1
        planned = list((runtime.get("query_plan") or {}).get("hops") or [])
        executed = list((runtime.get("execution") or {}).get("hops") or [])
        executed_by_index = {
            int(hop.get("hop_index", -1)): hop
            for hop in executed
            if isinstance(hop, dict)
        }
        execution_complete = bool(planned) and all(
            index in executed_by_index
            and bool(executed_by_index[index].get("matches"))
            for index in range(1, len(planned) + 1)
        )
        if not execution_complete:
            missing_execution.append(
                question_key(str(trajectory.dataset), str(trajectory.qid))
            )
    if missing_execution:
        raise ValueError(
            f"proofkg_process_version={version} requires complete execution traces "
            "on eligible rows; missing/incomplete for "
            f"{len(missing_execution)} rows, examples={missing_execution[:5]}"
        )
    if not eligible:
        raise ValueError(
            f"proofkg_process_version={version} found zero identity-safe complete "
            "automatic ProofKG eligible rows"
        )
    return {"eligible_rows": eligible, "missing_execution_rows": 0}


def _configure_fresh_value_head(
    policy,
    *,
    init_strategy: str,
    dropout: float,
) -> Dict[str, float]:
    """Configure TRL's fresh critic before cloning the frozen reference.

    Phase 3a checkpoints contain no value-head weights, so TRL constructs a new
    ``nn.Linear(hidden_size, 1)``.  Its default output had std ~=0.7 in the
    explicit-reference smoke, comparable to the return std ~=0.6, and produced
    strongly negative explained variance before the critic had learned anything.
    A zero head is the neutral baseline: it predicts V(s)=0 at update zero but
    receives ordinary gradients on the first value-loss update.
    """
    import torch.nn as nn

    if init_strategy not in {"default", "zero"}:
        raise ValueError(
            "value_head_init must be 'default' or 'zero', "
            f"got {init_strategy!r}"
        )
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"value_head_dropout must be in [0, 1), got {dropout}")

    value_head = getattr(policy, "v_head", None)
    summary = getattr(value_head, "summary", None)
    if value_head is None or summary is None:
        raise RuntimeError(
            "TRL policy has no v_head.summary; cannot apply the declared critic "
            "initialisation. Refusing to continue with an unknown value head."
        )

    value_head.dropout = nn.Identity() if dropout == 0 else nn.Dropout(float(dropout))
    if init_strategy == "zero":
        with torch.no_grad():
            summary.weight.zero_()
            if summary.bias is not None:
                summary.bias.zero_()

    if not summary.weight.requires_grad:
        raise RuntimeError("PPO value-head weight is frozen; critic cannot learn")
    weight_norm = float(summary.weight.detach().float().norm().item())
    bias_norm = (
        float(summary.bias.detach().float().norm().item())
        if summary.bias is not None else 0.0
    )
    logger.info(
        "PPO value head: init=%s dropout=%.3f weight_norm=%.6f "
        "bias_norm=%.6f trainable=true",
        init_strategy, dropout, weight_norm, bias_norm,
    )
    return {
        "weight_norm": weight_norm,
        "bias_norm": bias_norm,
        "dropout": float(dropout),
    }


def _smoke_health_guard_reason(
    history: Sequence[Dict[str, Any]],
    cfg: Phase3PPOConfig,
) -> Optional[str]:
    """Return a pre-registered smoke failure reason, else ``None``.

    The guard only saves GPU cost on a clearly unhealthy run. It does not mark
    a run successful, choose a checkpoint, or alter rewards/evaluation. A
    rolling window is required because batch_size=4 makes a single update too
    noisy for a defensible stop decision.
    """
    if not history:
        return None
    immediate = _nonfinite_training_state_reason(history[-1])
    if immediate is not None:
        return immediate
    if cfg.health_guard_after_steps <= 0:
        return None
    if int(history[-1].get("step", 0)) < cfg.health_guard_after_steps:
        return None
    window = int(cfg.health_guard_window)
    if window < 3:
        raise ValueError("health_guard_window must be >=3")
    if len(history) < window:
        return None

    tail = list(history[-window:])
    fields = ("valid_rate", "length_capped_frac", "ppo_mean_kl")
    for field_name in fields:
        values = [row.get(field_name) for row in tail]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return f"non-finite or missing {field_name} in rolling window"

    mean_valid = sum(float(row["valid_rate"]) for row in tail) / window
    mean_capped = sum(float(row["length_capped_frac"]) for row in tail) / window
    mean_kl = sum(float(row["ppo_mean_kl"]) for row in tail) / window
    if mean_valid < cfg.health_guard_min_valid_rate:
        return (
            f"rolling valid_rate={mean_valid:.4f} < "
            f"{cfg.health_guard_min_valid_rate:.4f} (window={window})"
        )
    if mean_capped > cfg.health_guard_max_length_capped_frac:
        return (
            f"rolling length_capped_frac={mean_capped:.4f} > "
            f"{cfg.health_guard_max_length_capped_frac:.4f} (window={window})"
        )
    if mean_kl > cfg.health_guard_max_mean_kl:
        return (
            f"rolling mean KL={mean_kl:.4f} > "
            f"{cfg.health_guard_max_mean_kl:.4f} (window={window})"
        )
    return None


def _nonfinite_training_state_reason(row: Dict[str, Any]) -> Optional[str]:
    """Return an immediate failure reason for a numerically corrupt update.

    Unlike the rolling smoke thresholds, this invariant is not delayed by a
    warm-up window: NaN/Inf losses, advantages, returns, rewards, or KL cannot
    become healthy evidence by averaging more paid updates.  Missing optional
    TRL statistics remain allowed for compatibility; any statistic that is
    present must be finite.
    """

    fields = (
        "mean_reward",
        "ppo_mean_kl",
        "policy_approxkl",
        "loss_total",
        "loss_policy",
        "loss_value",
        "advantage_var",
        "advantage_raw_mean",
        "advantage_raw_std",
        "advantage_raw_min",
        "advantage_raw_max",
        "advantage_raw_p50",
        "advantage_raw_p90",
        "advantage_raw_p95",
        "advantage_raw_p99",
        "advantage_whitened_var",
        "return_mean",
        "return_std",
        "value_mean",
        "value_std",
    )
    for field_name in fields:
        value = row.get(field_name)
        if value is None:
            continue
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            return f"non-finite training statistic {field_name}={value!r}"
    return None

def _build_models(cfg: Phase3PPOConfig):
    """Build policy+value, reference, and tokenizer.

    BUGFIX (2026-06-22): the previous version passed ``base_id`` = SFT-adapter
    dir together with a fresh ``peft_config``. TRL's ``from_pretrained`` ignores
    ``peft_config`` when an ``adapter_config.json`` is present and loads the
    trained adapter with ``is_trainable=False`` (the default) — so the SFT LoRA
    was loaded FROZEN, PPO produced zero gradient on it, and the saved checkpoint
    was byte-identical to SFT. We now load the SFT adapter as TRAINABLE and anchor
    the KL reference to a frozen SFT copy (not the bare base, which is what
    adapter-disabling would give and which would penalise SFT-acquired behaviour).
    """
    import os
    import torch as _torch
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, create_reference_model

    dtype_map = {"bf16": _torch.bfloat16, "fp16": _torch.float16, "fp32": _torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, _torch.bfloat16)
    base_id = cfg.sft_checkpoint or model_path(cfg.base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    has_adapter = os.path.exists(os.path.join(str(base_id), "adapter_config.json"))
    policy_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    is_peft = False

    if has_adapter:
        # base_id is a trained PEFT adapter dir (the SFT student). TRL will load
        # base + this adapter; force is_trainable=True so PPO can update it and
        # save_pretrained writes the UPDATED weights (SFT+PPO in one adapter).
        policy_kwargs["is_trainable"] = True
        is_peft = True
    elif cfg.use_lora:
        # Fresh start from the bare base model: attach a new trainable LoRA.
        try:
            from peft import LoraConfig

            policy_kwargs["peft_config"] = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            is_peft = True
        except ImportError:
            logger.warning("peft not installed; PPO will fine-tune all parameters.")

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(base_id, **policy_kwargs)
    _configure_fresh_value_head(
        policy,
        init_strategy=cfg.value_head_init,
        dropout=cfg.value_head_dropout,
    )

    # Sanity: confirm at least one LoRA parameter is actually trainable, so a
    # frozen-adapter regression can never silently return (the original bug).
    if is_peft:
        n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info("Policy trainable params: %d", n_trainable)
        if n_trainable == 0:
            raise RuntimeError(
                "Policy has 0 trainable parameters — the LoRA adapter loaded frozen. "
                "PPO would be a no-op (this was the 2026-06-22 bug). Aborting."
            )

    # Activation memory: enable gradient checkpointing on the POLICY so the 8B
    # policy + frozen SFT reference + ReaRAG-9B reward co-reside on one 96GB card
    # (the 2026-06-22 run sat at ~93/96GB without it — one long rollout from OOM).
    # The value-head wrapper proxies to .pretrained_model; checkpoint that. With
    # LoRA the base is frozen, so enable_input_require_grads() is REQUIRED — else
    # the checkpointed segments have no grad-requiring input and the LoRA grad
    # never flows (silent no-learn). We set config.use_cache=False for the
    # training forward/backward (KV-cache is incompatible with checkpointing);
    # the rollout generate() below passes use_cache=True explicitly to override
    # this, so generation stays fast (it runs under no_grad — no activation cost).
    inner = getattr(policy, "pretrained_model", policy)
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable()
        if hasattr(inner, "enable_input_require_grads"):
            inner.enable_input_require_grads()
        if hasattr(inner, "config"):
            inner.config.use_cache = False
        logger.info("Gradient checkpointing enabled on policy (use_cache=False).")

    # KL reference = frozen snapshot of the SFT-initialised policy. We always
    # build an explicit reference (deepcopy, ~+1 model of VRAM) rather than
    # ref_model=None: adapter-disabling would anchor KL to the BARE BASE, pulling
    # the policy away from SFT. Anchoring to SFT is the standard PPO setup.
    ref_model = create_reference_model(policy)
    return policy, ref_model, tokenizer


def _step_logprobs_from_scores(
    response_ids: torch.Tensor,
    scores: Sequence[torch.Tensor],
    spans,
    row: int = 0,
) -> List[Optional[List[float]]]:
    """Slice per-step token logprobs from generation ``scores`` by token span.

    ``scores`` is the tuple from ``generate(output_scores=True)``: one
    ``(batch, vocab)`` logit tensor per *generated* token. We convert each to the
    logprob of the actually-sampled token, then bucket those into the step
    spans (P1-1 feeds these to the α-gate's entropy feature).

    ``row`` selects which sequence of a batched generate() call to read. It must
    match the row ``response_ids`` came from: scores are indexed [token][row],
    and mixing the two up silently attributes one rollout's logprobs to another
    (the α-gate entropy feature would then be reading the wrong trajectory).
    """
    if not scores:
        return [None] * len(spans)
    # logprob of the sampled token at each generated position.
    tok_logprobs: List[float] = []
    n_gen = min(len(scores), response_ids.size(0))
    for t in range(n_gen):
        logits = scores[t][row]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tok_id = int(response_ids[t].item())
        tok_logprobs.append(float(lp[tok_id].item()))
    out: List[Optional[List[float]]] = []
    for start, end in spans:
        s = max(0, start)
        e = min(end, len(tok_logprobs))
        out.append(tok_logprobs[s:e] if e > s else None)
    return out


def _load_hybrid_rollout_inputs(
    cfg: Phase3PPOConfig,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    """Validate passage overrides and reproduce the frozen smoke schedule.

    This runs before model allocation.  The runtime loop still draws indices
    with the original shared torch RNG; the frozen schedule is a fail-fast
    assertion, not a replacement sampler.
    """
    if bool(cfg.passage_overrides_path) != bool(cfg.rollout_schedule_path):
        raise ValueError(
            "passage_overrides_path and rollout_schedule_path must be provided together"
        )
    if not cfg.passage_overrides_path:
        return {}, [], {}
    if cfg.batch_size <= 0 or cfg.total_steps <= 0 or cfg.total_steps % cfg.batch_size:
        raise ValueError(
            "hybrid rollout schedule requires positive total_steps divisible by batch_size"
        )

    override_path = Path(cfg.passage_overrides_path)
    schedule_path = Path(cfg.rollout_schedule_path)
    for label, path in (("passage overrides", override_path), ("rollout schedule", schedule_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    overrides: Dict[str, Dict[str, Any]] = {}
    with override_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or "")
            if not qid or qid in overrides:
                raise ValueError(f"passage overrides contain invalid/duplicate qid: {qid!r}")
            passages = row.get("retrieved_passages")
            if not isinstance(passages, list) or not passages:
                raise ValueError(f"passage override {qid} has no retrieved_passages list")
            overrides[qid] = row

    schedule_rows: List[Dict[str, Any]] = []
    with schedule_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                schedule_rows.append(json.loads(line))
    if len(schedule_rows) != cfg.total_steps:
        raise ValueError(
            f"rollout schedule has {len(schedule_rows)} rows, expected total_steps={cfg.total_steps}"
        )
    indices = [int(row.get("rollout_index", -1)) for row in schedule_rows]
    if indices != list(range(1, cfg.total_steps + 1)):
        raise ValueError("rollout schedule indices must be contiguous from 1")
    schedule_qids = [str(row.get("qid") or "") for row in schedule_rows]
    if any(not qid for qid in schedule_qids):
        raise ValueError("rollout schedule contains an empty qid")
    missing_overrides = sorted(set(schedule_qids) - set(overrides))
    if missing_overrides:
        raise ValueError(f"rollout schedule qids missing passage overrides: {missing_overrides}")

    # Reproduce the current loop's shared explore/replay RNG consumption over
    # the gold-bearing train samples.  This catches a changed seed, fold, silver
    # order, replay ratio or batch size before CUDA models are allocated.
    reader = SilverDatasetReader(
        cfg.silver_path,
        split=cfg.split,
        split_spec=cfg.build_split_spec() if cfg.split else None,
    )
    samples = [
        item for item in reader.accepted()
        if str(item.metadata.get("gold_answer") or "").strip()
    ]
    if not samples:
        raise ValueError("hybrid rollout schedule found no accepted gold-bearing PPO samples")
    generator = torch.Generator().manual_seed(cfg.seed)
    simulation_weights = None
    if cfg.rollout_sampling_weights_path:
        simulation_weights, _ = _load_rollout_sampling_weights(
            cfg.rollout_sampling_weights_path, samples,
        )
    replay_credit = 0.0
    simulated_qids: List[str] = []
    for n_seen in range(0, cfg.total_steps, cfg.batch_size):
        explore_idx = _sample_rollout_indices(
            len(samples), cfg.batch_size, cfg.rollouts_per_prompt, generator,
            sampling_weights=simulation_weights,
        )
        simulated_qids.extend(samples[index].qid for index in explore_idx)
        replay_items = 0
        if cfg.sft_replay_ratio > 0 and cfg.sft_anchor_weight > 0:
            replay_items, replay_credit = _advance_replay_credit(
                replay_credit,
                batch_size=cfg.batch_size,
                replay_ratio=cfg.sft_replay_ratio,
            )
        elif (
            cfg.sft_replay_ratio == 0
            and cfg.sft_anchor_weight > 0
            and cfg.sft_anchor_interval > 0
            and (n_seen // cfg.batch_size + 1) % cfg.sft_anchor_interval == 0
        ):
            replay_items = 1
        if replay_items:
            # _prepare_sft_anchor_data is capped at 2000. Only the number of
            # shared-generator draws affects later explore batches.
            torch.randint(0, 2000, (replay_items,), generator=generator)
    if simulated_qids != schedule_qids:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(simulated_qids, schedule_qids), start=1
                )
                if actual != expected
            ),
            None,
        )
        raise ValueError(
            "rollout schedule does not match current PPO RNG/data protocol; "
            f"first mismatch at rollout {mismatch}"
        )
    metadata = {
        "override_path": str(override_path.resolve()),
        "schedule_path": str(schedule_path.resolve()),
        "scheduled_rollouts": len(schedule_qids),
        "scheduled_unique_qids": len(set(schedule_qids)),
        "override_qids": len(overrides),
    }
    return overrides, schedule_qids, metadata


def _prepare_prompts(reader: SilverDatasetReader, tokenizer, cfg: Phase3PPOConfig,
                     question_kg_index: dict = None,
                     passage_overrides: Optional[Dict[str, Dict[str, Any]]] = None):
    """Build PPO prompts.  When ``question_kg_index`` is provided, the Knowledge
    Graph block is populated from the pre-built Q→KG lookup (instant, 100% hit).
    """
    rows = []
    skipped_no_gold = 0
    dyn_kg_hits = 0
    # 2026-08-22: counted separately from dyn_kg_hits because they mean different
    # things and only one of them indicates a broken artefact.
    #   COVERED-BUT-EMPTY: the question IS in the index, and its subgraph is
    #     legitimately empty -- entity linking abstained, or the cache has no
    #     triples for the linked QIDs. MEASURED on a full rebuild of
    #     silver_v1_reannotated (--min_keep 5 --max_keep 12, offline, 174 s):
    #     23.3% over all 24,997 questions, but 9.7% over the 9,839 ACCEPTED ones
    #     PPO actually rolls out on, and 6.3% after the silver kg_subgraph
    #     fallback rescues 333 of them. It does NOT drop by rebuilding: it is a
    #     property of the KG and the linker, not of the index. Counting these as
    #     misses is what made a *correctly built* index trip the 5% guard below.
    #   ABSENT: the question is not in the index at all -- the index was built
    #     over a different question set. This is the 100% failure that PPO(1)/(2)
    #     ran with, and the only case the guard should fire on.
    dyn_kg_absent = 0
    dyn_kg_empty = 0
    dyn_kg_mismatch = 0
    dropped_prompt_passages = 0
    mismatch_examples: List[str] = []
    passage_overrides = passage_overrides or {}
    passage_override_hits = 0
    for traj in reader.accepted():
        kg_triples = list(traj.kg_subgraph)
        if (cfg.source_gated_reward_version == "v1"
                and len(kg_triples) > cfg.ppo_max_kg_triples):
            raise ValueError(
                f"source-gated prompt KG exceeds visible budget for qid={traj.qid}; "
                "refusing to score graph triples omitted from the policy prompt"
            )
        # R9: instant Q→KG lookup from pre-built index (0.2s, 100% coverage)
        if question_kg_index is not None:
            dyn = question_kg_index.get(traj.question)
            if traj.question not in question_kg_index:
                dyn_kg_absent += 1
            elif not dyn:
                dyn_kg_empty += 1
            if (
                traj.question in question_kg_index
                and list(dyn or []) != list(traj.kg_subgraph)
            ):
                dyn_kg_mismatch += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(traj.qid)
            if dyn:
                dyn_kg_hits += 1
                kg_triples = list(dyn)
            elif kg_triples:
                # Index miss: the silver record's raw kg_subgraph has NOT been
                # through the three-layer policy, so applying it here keeps the
                # PPO prompt's KG distribution identical to the indexed case
                # instead of silently reverting to SPARQL-order noise.
                kg_triples = filter_and_rank_triples(
                    [tuple(t) for t in kg_triples if len(t) == 3],
                    question=traj.question,
                    min_keep=cfg.ppo_min_kg_triples,
                    max_keep=cfg.ppo_max_kg_triples,
                )

        source_passages = list(traj.retrieved_passages)
        passage_override_applied = traj.qid in passage_overrides
        if passage_override_applied:
            override = passage_overrides[traj.qid]
            override_question = str(override.get("question") or "").strip()
            if override_question and override_question != traj.question.strip():
                raise ValueError(f"passage override question mismatch for qid={traj.qid}")
            source_passages = list(override["retrieved_passages"])
            passage_override_hits += 1

        n_passages = min(cfg.ppo_max_passages, len(source_passages))
        while True:
            visible_passages = source_passages[:n_passages]
            msgs = build_rl_messages(
                question=traj.question,
                retrieved_passages=visible_passages,
                kg_triples=kg_triples,
                top_k=n_passages,
                max_kg_triples=cfg.ppo_max_kg_triples,
            )
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            else:
                text = "\n\n".join(m["content"] for m in msgs)
            prompt_ids = tokenizer(
                text, truncation=False, add_special_tokens=False,
            )["input_ids"]
            if len(prompt_ids) <= cfg.max_input_length:
                break
            if n_passages == 0:
                raise ValueError(
                    f"PPO prompt qid={traj.qid} is {len(prompt_ids)} tokens with "
                    f"zero passages, above max_input_length={cfg.max_input_length}; "
                    "refusing to right-truncate the trailing KG block"
                )
            n_passages -= 1
            dropped_prompt_passages += 1
        gold = str(traj.metadata.get("gold_answer") or "").strip()
        if not gold:
            skipped_no_gold += 1
            continue
        raw_aliases = traj.metadata.get("gold_answer_aliases")
        if isinstance(raw_aliases, str):
            gold_aliases = [raw_aliases]
        elif isinstance(raw_aliases, (list, tuple)):
            gold_aliases = [
                value for value in raw_aliases
                if isinstance(value, str) and value.strip()
            ]
        else:
            gold_aliases = []
        if not gold_aliases:
            gold_aliases = [gold]
        spec = RewardSpec(
            query=traj.question,
            gold_answer=gold,
            kg_subgraph=kg_triples,  # R9: may be dynamic, may be silver fallback
            # Reward-side passage evidence must match what the policy saw.
            retrieved_passages=visible_passages,
            metadata={
                "qid": traj.qid,
                "dataset": traj.dataset,
                "passage_override_applied": passage_override_applied,
                "question_kg_runtime": dict(
                    traj.metadata.get("question_kg_runtime") or {}
                ),
                "source_quality_record": dict(
                    traj.metadata.get("source_quality_record") or {}
                ),
            },
            gold_answer_aliases=gold_aliases,
        )
        rows.append({
            "prompt": text,
            "spec": spec,
            "num_passages": n_passages,
            "prompt_tokens": len(prompt_ids),
        })
    if skipped_no_gold:
        logger.warning(
            "Skipped %d accepted trajectories with no gold_answer (A4: no teacher fallback).",
            skipped_no_gold,
        )
    if dropped_prompt_passages:
        logger.warning(
            "PPO prompt preparation dropped %d low-ranked passages across %d "
            "samples to preserve the complete trailing KG block",
            dropped_prompt_passages, len(rows),
        )
    if passage_overrides:
        logger.info(
            "Versioned passage overrides applied to %d/%d prepared PPO prompts; "
            "runtime schedule guard forbids selecting an unoverridden prompt.",
            passage_override_hits, len(rows),
        )
    if question_kg_index is not None and rows and dyn_kg_hits < len(rows):
        _miss = len(rows) - dyn_kg_hits
        _miss_rate = _miss / len(rows)
        _absent_rate = dyn_kg_absent / len(rows)
        logger.warning(
            "question_kg_index gave no triples for %d/%d PPO prompts (%.1f%%): "
            "%d ABSENT from the index (%.1f%% — wrong artefact) + %d present but "
            "with an EMPTY subgraph (%.1f%% — the KG genuinely has nothing). Both "
            "fall back to the silver kg_subgraph (policy-filtered inline).",
            _miss, len(rows), 100.0 * _miss_rate,
            dyn_kg_absent, 100.0 * _absent_rate,
            dyn_kg_empty, 100.0 * dyn_kg_empty / len(rows),
        )
        # 2026-08-22 (retraining_plan R-2 / §8 病灶 1): the warning above was the
        # only signal that the dev-split index missed 100% of the train-fold
        # prompts, and it scrolled past unread for the whole of PPO(1)/PPO(2).
        # A miss degrades BOTH the prompt KG and r_kg (the same subgraph is the
        # verification reference in RewardSpec), so it is a silent quality drop,
        # not a cosmetic one. Fail fast when a threshold is configured.
        #
        # The threshold is checked against ABSENT ONLY, not against absent+empty.
        # A correct rebuild of silver_v1_reannotated measures 0.00% absent
        # (24,997/24,997 covered) but 9.7% covered-but-empty among accepted
        # prompts. That 9.7% is a property of the KG and the linker and does not
        # fall with any rebuild, so checking the combined rate against a 5% cap
        # would abort on a CORRECTLY built index -- a guard that fires on the
        # fixed state is worse than no guard, because the only way past it is to
        # disable it.
        _cap = getattr(cfg, "max_kg_index_miss_rate", 1.0)
        if _absent_rate > _cap:
            raise ValueError(
                f"question_kg_index is ABSENT for {dyn_kg_absent}/{len(rows)} "
                f"prompts ({_absent_rate:.1%}), above the configured "
                f"max_kg_index_miss_rate={_cap:.1%}. The index was almost "
                "certainly built over a different question set than this silver "
                "file — the shipped question_kg_index_v2.json is built from the "
                "DEV splits (qids dev_*) and misses 100% of the train_* prompts "
                "PPO rolls out on. Rebuild with:\n"
                "  python scripts/prepare/06_build_question_kg_index.py \\\n"
                f"    --silver {cfg.silver_path} --min_keep {cfg.ppo_min_kg_triples} --max_keep "
                f"{cfg.ppo_max_kg_triples} \\\n"
                "    --output indexes/kg_cache/question_kg_index_v2_train.json\n"
                "then point training.question_kg_index_path at it "
                "(retraining_plan §10.3). Set max_kg_index_miss_rate=1.0 to "
                "train anyway."
            )
    if dyn_kg_hits > 0:
        logger.info("R9 prompt KG: %d/%d samples got subgraphs from pre-built index", dyn_kg_hits, len(rows))
        # Print one example: show KG block from the first dynamic-KG prompt.
        # R10: the RL prompt is RL_USER_TEMPLATE = SFT_USER_TEMPLATE, which
        # delimits the block with "[Knowledge Graph Context]" (prompts.py:120).
        # The old literal "Knowledge Graph:" appears nowhere in it, so this
        # example never printed. "Knowledge Graph (2-hop):" is the phase1
        # distillation template, not this path -- do not use it here either.
        KG_HDR = "[Knowledge Graph Context]"
        for row in rows:
            prompt = row["prompt"]
            if "(empty)" not in prompt and KG_HDR in prompt:
                kg_start = prompt.index(KG_HDR)
                kg_end = prompt.index("\n\n", kg_start) if "\n\n" in prompt[kg_start:] else len(prompt)
                logger.info("R9 KG example:\n%s", prompt[kg_start:kg_end])
                break
    if getattr(cfg, "require_exact_kg_index_alignment", False) and dyn_kg_mismatch:
        raise ValueError(
            "question_kg_index triples differ from stored silver KG for "
            f"{dyn_kg_mismatch} accepted trajectories (examples: "
            f"{', '.join(mismatch_examples)}). Formal stored-silver PPO requires "
            "exact ordered alignment, not question coverage alone. Rebuild the "
            "index with scripts/prepare/build_stored_silver_kg_index.py and "
            "re-audit the Phase2-enriched silver before training."
        )
    return rows


def _prepare_sft_anchor_data(
    silver_path: str,
    tokenizer,
    cfg: Phase3PPOConfig,
    max_samples: int = 2000,
    apply_rollout_question_kg: bool = True,
    replay_split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """R7: Build tokenised SFT samples for the format-preservation anchor.

    Each sample is a ``(prompt, full_trajectory)`` pair tokenised as a single
    sequence with the prompt portion masked in the labels.  The anchor loss is
    cross-entropy on the trajectory tokens only — it nudges the policy to
    maintain the ``[Step N] ... [Final Answer]`` output format without forcing
    specific answers.

    The supervised target is the complete standardised trajectory reconstructed
    from ``traj.steps`` plus ``metadata.gold_answer``. ``traj.answer`` is only a
    short final answer in the repaired legacy silver (zero records contain a
    ``[Step N]`` marker), so using it here teaches bare-answer output and directly
    conflicts with PPO's trajectory-validity constraint.

    The fresh reader must honour ``cfg.split`` as well. The anchor computes a
    cross-entropy loss on these trajectories' tokens, so it is training on them
    every bit as much as the PPO objective is; leaving it on the whole file would
    quietly train the policy on the held-out questions.
    """
    reader2 = SilverDatasetReader(
        silver_path,
        split=replay_split if replay_split is not None else cfg.split,
        split_spec=(
            cfg.build_split_spec()
            if (replay_split is not None or cfg.split) else None
        ),
    )
    if apply_rollout_question_kg and getattr(cfg, "question_kg_records_path", None):
        replay_kg_stats = apply_training_question_kg(
            reader2.accepted(),
            read_question_kg_records(cfg.question_kg_records_path),
            min_coverage=getattr(cfg, "min_question_kg_record_coverage", 1.0),
            require_nonempty=getattr(
                cfg, "require_nonempty_question_kg_records", False
            ),
        )
        logger.info(
            "PPO replay question-KG override: %s", replay_kg_stats.to_dict()
        )
    sft_samples: List[Dict[str, Any]] = []
    from kgproweight.training.phase3_sft import _render_assistant_trace

    # Do not let JSONL order decide which questions make the replay pool.  Sort
    # first so this remains reproducible if the same records are re-serialised,
    # then take a seeded sample from the complete training fold.
    candidates = sorted(
        reader2.accepted(),
        key=lambda traj: (str(traj.dataset), str(traj.qid)),
    )
    replay_rng = random.Random(int(getattr(cfg, "seed", 42)))
    replay_rng.shuffle(candidates)

    for traj in candidates:
        answer_trace = _render_assistant_trace(traj)
        if not answer_trace.strip():
            continue
        max_total = cfg.max_input_length + cfg.max_new_tokens
        n_passages = min(cfg.ppo_max_passages, len(traj.retrieved_passages))

        # Preserve the assistant target instead of right-truncating it.  Reduce
        # only the lowest-ranked passages until both the rollout prompt limit and
        # the complete prompt+trace limit fit, matching Phase 3a's policy.
        while True:
            msgs = build_sft_messages(
                question=traj.question,
                retrieved_passages=list(traj.retrieved_passages)[:n_passages],
                kg_triples=traj.kg_subgraph,
                top_k=n_passages,
                max_kg_triples=cfg.ppo_max_kg_triples,
            )
            full_msgs = build_sft_messages(
                question=traj.question,
                retrieved_passages=list(traj.retrieved_passages)[:n_passages],
                kg_triples=traj.kg_subgraph,
                answer_trace=answer_trace,
                top_k=n_passages,
                max_kg_triples=cfg.ppo_max_kg_triples,
            )
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False,
                )
            else:
                prompt_text = "\n\n".join(m["content"] for m in msgs)
                full_text = prompt_text + answer_trace

            # Chat templates already include their own BOS/control tokens.  Adding
            # special tokens again can shift the prompt mask by one token.
            prompt_ids = tokenizer(
                prompt_text, truncation=False, add_special_tokens=False,
            )["input_ids"]
            full_ids = tokenizer(
                full_text, truncation=False, add_special_tokens=False,
            )["input_ids"]
            fits = (
                len(prompt_ids) <= cfg.max_input_length
                and len(full_ids) <= max_total
            )
            if fits or n_passages == 0:
                break
            n_passages -= 1

        if not fits:
            logger.warning(
                "Skipping replay sample %s: complete target is %d tokens even "
                "with zero passages (limit=%d)",
                traj.qid, len(full_ids), max_total,
            )
            continue
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Chat template prompt is not an exact prefix of the supervised "
                f"sequence for qid={traj.qid}; replay labels cannot be aligned safely"
            )
        if len(full_ids) == len(prompt_ids):
            logger.warning("Skipping replay sample %s: no assistant target tokens", traj.qid)
            continue

        # Labels = full_ids, but mask the prompt portion so the loss is
        # computed only on the trajectory tokens (the format prior).
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        sft_samples.append({
            "input_ids": full_ids,
            "labels": labels,
            "qid": traj.qid,
            "question": traj.question,
            "answer_trace": answer_trace,
            "num_passages": n_passages,
        })
        if len(sft_samples) >= max_samples:
            break

    return sft_samples


def _advance_replay_credit(
    credit: float,
    *,
    batch_size: int,
    replay_ratio: float,
) -> Tuple[int, float]:
    """Return supervised replay items due after one PPO batch.

    ``int(batch_size * ratio)`` made 15% replay silently equal zero at the
    formal batch size of four. Fractional credit preserves the requested sample
    ratio deterministically: at batch=4 and ratio=.10, every five PPO batches
    schedule two replay items (2 / 20 = 10%).
    """
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError(f"sft_replay_ratio must be in [0, 1], got {replay_ratio}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    credit += batch_size * replay_ratio
    due = int(credit + 1e-12)
    remainder = credit - due
    if abs(remainder) < 1e-12:
        remainder = 0.0
    return due, remainder


def _runtime_contract_v2(cfg: Phase3PPOConfig) -> bool:
    version = getattr(cfg, "runtime_contract_version", "legacy")
    if version not in {"legacy", "v2"}:
        raise ValueError(f"unknown PPO runtime_contract_version: {version!r}")
    return version == "v2"


def _rollout_eos_token_ids(policy, tokenizer) -> Tuple[int, ...]:
    """Read the effective generation EOS setting, including multiple stop IDs.

    Prefer the model's generation configuration over the tokenizer's single EOS:
    Llama may terminate on both end-of-text and end-of-turn tokens.
    """
    inner = getattr(policy, "pretrained_model", policy)
    configs = (
        getattr(inner, "generation_config", None),
        getattr(policy, "generation_config", None),
        getattr(inner, "config", None),
        tokenizer,
    )
    for config in configs:
        value = getattr(config, "eos_token_id", None)
        if value is not None:
            values = value if isinstance(value, (list, tuple)) else [value]
            result = tuple(dict.fromkeys(int(token_id) for token_id in values))
            if result and all(token_id >= 0 for token_id in result):
                return result
            raise ValueError("PPO v2 requires nonempty, nonnegative EOS token IDs")
    raise ValueError("PPO v2 cannot determine the generation EOS token IDs")


@contextmanager
def _rollout_eval_mode(policy):
    """Disable dropout during sampling and restore all module modes on exit."""
    modes = [(module, module.training) for module in policy.modules()]
    policy.eval()
    try:
        yield
    finally:
        # Preserve selectively frozen/eval submodules as well as the root mode.
        for module, was_training in modes:
            module.training = was_training


def _trim_response_v2(
    response_ids: torch.Tensor, *, eos_token_ids: Sequence[int],
    pad_token_id: int, max_new_tokens: int,
) -> torch.Tensor:
    """Keep the first true EOS; remove only filler after that termination."""
    if response_ids.ndim != 1 or not 0 < response_ids.numel() <= max_new_tokens:
        raise ValueError("PPO v2 received an empty, non-vector or oversized response")
    eos_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for eos_id in eos_token_ids:
        eos_mask |= response_ids == eos_id
    stops = eos_mask.nonzero(as_tuple=False)
    if stops.numel():
        end = int(stops[0].item()) + 1
        if not bool((response_ids[end:] == pad_token_id).all()):
            raise ValueError("PPO v2 found non-padding tokens after the first EOS")
        return response_ids[:end]
    # A pad token sampled before any EOS is a real action, not filler. This
    # includes a length-capped response whose last token happens to be PAD.
    return response_ids


def _response_is_length_capped_v2(
    response_ids: torch.Tensor, *, max_new_tokens: int, eos_token_ids: Sequence[int],
) -> bool:
    return bool(
        response_ids.numel() >= max_new_tokens
        and int(response_ids[-1].item()) not in eos_token_ids
    )


def _align_token_rewards(
    token_rewards: torch.Tensor, response_ids: torch.Tensor, *,
    trajectory_reward: float, runtime_contract_version: str = "legacy",
) -> torch.Tensor:
    """Validate successor reward conservation before TRL can mask any token."""
    if runtime_contract_version == "v2":
        if response_ids.ndim != 1 or response_ids.numel() == 0:
            raise ValueError("PPO v2 requires a nonempty response vector")
        n = response_ids.numel()
        if not isinstance(token_rewards, torch.Tensor) or token_rewards.ndim != 1:
            raise ValueError("PPO v2 token rewards must be a one-dimensional tensor")
        if token_rewards.numel() != n:
            raise ValueError(
                f"PPO v2 token reward/response length mismatch: {token_rewards.numel()} != {n}"
            )
        if not bool(torch.isfinite(token_rewards).all()) or not math.isfinite(float(trajectory_reward)):
            raise ValueError("PPO v2 received non-finite token or trajectory rewards")
        actual = float(token_rewards.detach().double().sum().item())
        if not math.isclose(actual, float(trajectory_reward), rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(
                f"PPO v2 token reward conservation failed: sum={actual}, trajectory={trajectory_reward}"
            )
    elif runtime_contract_version == "legacy":
        n = response_ids.size(0)
        if token_rewards.size(0) != n:
            token_rewards = (
                torch.cat([token_rewards, torch.zeros(n - token_rewards.size(0), dtype=token_rewards.dtype)])
                if token_rewards.size(0) < n else token_rewards[:n]
            )
    else:
        raise ValueError(f"unknown PPO runtime_contract_version: {runtime_contract_version!r}")
    return token_rewards


def _generate(policy, tokenizer, prompts: Sequence[str], cfg: Phase3PPOConfig, device: str):
    """Generate one response per prompt, decoding ``cfg.rollout_chunk_size`` at a time.

    R10 speed: this used to decode one prompt per generate() call, which made
    rollout ~batch_size x max_new_tokens single-sample decode steps per optimiser
    update -- the dominant cost in the 80.8 s/update measured on 2026-08-06.
    Batching collapses that to ~max_new_tokens batched steps per chunk.

    SCALE: when ``use_real_logprobs`` is on we collapse each chunk's generation
    ``scores`` into small per-step logprob lists and free the raw logits before
    the next chunk, so only one chunk's logits are resident
    (chunk x vocab x 4 B per generated token) rather than the whole batch's.

    Returns UNPADDED query and response tensors: padding exists only inside the
    generate() call, never in what TRL receives.
    """
    query_tensors, response_tensors, response_texts, logprobs_per_step_list = [], [], [], []
    # Read the device LIVE from the model: PPOTrainer/accelerate move the policy
    # to CUDA *after* run_phase3_ppo computed its `device` string, so the passed
    # `device` can be a stale "cpu". Trust where the params actually are now.
    try:
        device = next(p for p in policy.parameters()).device
    except StopIteration:
        pass

    contract_v2 = _runtime_contract_v2(cfg)
    eos_ids = _rollout_eos_token_ids(policy, tokenizer) if contract_v2 else ()
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ) if contract_v2 else (tokenizer.pad_token_id or tokenizer.eos_token_id)
    if contract_v2 and pad_id is None:
        raise ValueError("PPO v2 requires a padding token ID")
    chunk = max(1, int(getattr(cfg, "rollout_chunk_size", 1) or 1))

    for lo in range(0, len(prompts), chunk):
        group = list(prompts[lo:lo + chunk])

        # _prepare_prompts already shrinks low-ranked passages to preserve the
        # trailing KG block. Recheck the invariant here because silent right
        # truncation would turn a nominal +KG run into a partially no-KG run.
        for prompt in group:
            full_len = len(tokenizer(
                prompt, truncation=False, add_special_tokens=False,
            )["input_ids"])
            if full_len > cfg.max_input_length:
                raise ValueError(
                    "PPO prompt-length invariant violated after preparation: "
                    f"{full_len} > {cfg.max_input_length}. Refusing to truncate KG."
                )

        # LEFT padding is required for batched decoding: generate() appends new
        # tokens at the right edge, so right-padding would put pad tokens between
        # the prompt and the continuation and the model would attend to them as
        # context. With left padding every row's generation starts at the same
        # index, which is what lets `gen[:, plen:]` below be a single slice.
        #
        # Padding does NOT change the sampled distribution -- attention_mask
        # zeroes the pads -- so the "rollout == TRL scoring distribution"
        # invariant that top_k=0 protects is unaffected by batching.
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            enc = tokenizer(
                group,
                return_tensors="pt",
                truncation=False,
                padding=True,
                add_special_tokens=False,
            )
        finally:
            tokenizer.padding_side = prev_side

        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        plen = input_ids.size(1)

        with (_rollout_eval_mode(policy) if contract_v2 else nullcontext()), torch.no_grad():
            out = policy.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                # top_k=0 disables the default top-50 truncation: any truncation
                # (top_p<1 or top_k>0) makes the rollout distribution differ from
                # TRL's raw-logit logp recomputation and pushes KL negative.
                top_k=0,
                pad_token_id=pad_id,
                # Override the config.use_cache=False set for gradient
                # checkpointing: rollout runs under no_grad, so KV-cache is free
                # memory-wise and ~Nx faster than recomputing every step.
                use_cache=True,
                return_dict_in_generate=cfg.use_real_logprobs,
                output_scores=cfg.use_real_logprobs,
                **({"eos_token_id": list(eos_ids)} if contract_v2 else {}),
            )

        seqs = out.sequences if cfg.use_real_logprobs else out
        scores = out.scores if cfg.use_real_logprobs else None

        for row in range(len(group)):
            # Strip the LEFT pad back off the query: TRL gets the real prompt
            # tokens only. Feeding it padded queries would make logp_old be
            # recomputed over pad positions the rollout never conditioned on.
            q_row = input_ids[row]
            keep = attn[row].bool()
            query_ids = q_row[keep]

            # Every row's generation starts at `plen` thanks to left padding.
            response_ids = seqs[row][plen:]
            # Trim trailing pad/EOS filler emitted after this row finished, so
            # step spans and token-reward placement are not padded out. Keep one
            # EOS if present -- TRL expects the terminating token.
            if contract_v2:
                response_ids = _trim_response_v2(
                    response_ids, eos_token_ids=eos_ids,
                    pad_token_id=pad_id, max_new_tokens=cfg.max_new_tokens,
                )
            else:
                nz = (response_ids != pad_id).nonzero()
                if nz.numel() > 0:
                    response_ids = response_ids[: int(nz[-1].item()) + 1]
                else:
                    response_ids = response_ids[:1]

            resp_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            if cfg.use_real_logprobs:
                n_parsed = len(parse_steps(resp_text)[: cfg.max_steps])
                spans = step_spans_over_ids(response_ids, tokenizer, n_parsed)
                # row= must match the sequence: scores are [token][row].
                logprobs_per_step = _step_logprobs_from_scores(
                    response_ids, scores, spans, row=row
                )
            else:
                logprobs_per_step = None

            query_tensors.append(query_ids)
            response_tensors.append(response_ids)
            response_texts.append(resp_text)
            logprobs_per_step_list.append(logprobs_per_step)

        # SCALE: drop this chunk's raw logits before decoding the next one, so
        # only one chunk's scores are ever resident (chunk x vocab x 4 B x tokens).
        del out, seqs, scores

    return query_tensors, response_tensors, response_texts, logprobs_per_step_list


def _count_reasoning_content(response_texts: Sequence[str], min_chars: int = 20) -> Dict[str, Any]:
    """R8: Count how many responses have substantive reasoning content per step."""
    import re

    n_with_steps = 0
    n_with_final_answer = 0
    n_with_reasoning = 0
    total_steps = 0
    steps_with_content = 0

    for text in response_texts:
        if not text:
            continue
        if "[Step" in text or "Step " in text:
            n_with_steps += 1
        if "Final Answer" in text:
            n_with_final_answer += 1

        step_bodies = re.findall(
            r'\[Step \d+\]\s*(.*?)(?=\[Step|\Z)', text, re.DOTALL,
        )
        for body in step_bodies:
            total_steps += 1
            if "Reasoning:" in body:
                after = body.split("Reasoning:", 1)[1]
                reasoning = re.split(
                    r'Knowledge Used:|Conclusion:|Final Answer:', after,
                )[0].strip()
                if len(reasoning) >= min_chars:
                    steps_with_content += 1

    return {
        "n_samples": len(response_texts),
        "n_with_steps": n_with_steps,
        "n_with_final_answer": n_with_final_answer,
        "total_steps": total_steps,
        "steps_with_content": steps_with_content,
        "step_rate": n_with_steps / max(1, len(response_texts)),
        "final_answer_rate": n_with_final_answer / max(1, len(response_texts)),
        "reasoning_content_rate": (
            steps_with_content / max(1, total_steps) if total_steps > 0 else 0.0
        ),
    }


def _citation_reward_diagnostics(records: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate citation/alpha telemetry without changing reward computation.

    ``cite_match`` is a per-step fraction, so these metrics deliberately use
    ``*_step_*`` names.  They must not be reported as exact cited-triple counts.
    Missing groups are represented as ``None`` rather than a fabricated zero.
    """

    rows = list(records)
    citing = [r for r in rows if float(getattr(r, "cite_any", 0.0)) > 0.0]
    no_cite = [r for r in rows if float(getattr(r, "cite_any", 0.0)) <= 0.0]
    unknown_only = [
        r for r in citing if float(getattr(r, "cite_match", 0.0)) <= 0.0
    ]
    partial = [
        r for r in citing
        if 0.0 < float(getattr(r, "cite_match", 0.0)) < 1.0
    ]
    all_matched = [
        r for r in citing if float(getattr(r, "cite_match", 0.0)) >= 1.0
    ]
    known = partial + all_matched

    def _mean(group: Sequence[Any], field: str) -> Optional[float]:
        if not group:
            return None
        return float(sum(float(getattr(r, field)) for r in group) / len(group))

    def _zero_frac(group: Sequence[Any]) -> Optional[float]:
        if not group:
            return None
        return float(sum(float(getattr(r, "r_kg")) == 0.0 for r in group) / len(group))

    n = len(rows)
    n_citing = len(citing)
    return {
        "cite_any_step_frac": len(citing) / max(1, n),
        "cite_match_mean_citing_step": _mean(citing, "cite_match"),
        # Explicit alias: this uses only citations that survived KG-aware
        # parsing and are visible to reward; it is not raw citation accuracy.
        "cite_match_mean_reward_visible_citing_step": _mean(citing, "cite_match"),
        "cite_unknown_only_step_frac_citing": len(unknown_only) / max(1, n_citing),
        "cite_partial_match_step_frac_citing": len(partial) / max(1, n_citing),
        "cite_all_matched_step_frac_citing": len(all_matched) / max(1, n_citing),
        "alpha_mean_no_cite_step": _mean(no_cite, "alpha"),
        "alpha_mean_known_cite_step": _mean(known, "alpha"),
        "alpha_mean_unknown_cite_step": _mean(unknown_only, "alpha"),
        "r_kg_zero_frac_no_cite_step": _zero_frac(no_cite),
        "r_kg_zero_frac_known_cite_step": _zero_frac(known),
        "r_kg_zero_frac_unknown_cite_step": _zero_frac(unknown_only),
    }


def _citation_contract_diagnostics(
    parsed_responses: Sequence[Sequence[Any]],
) -> Dict[str, Any]:
    """Aggregate raw citation attempts without changing reward computation.

    Unknown citations intentionally stay out of ``ParsedStep.cited_triples`` so
    they cannot earn r_kg.  The old telemetry reused that filtered list and
    could therefore report cite_match=1.0 while silently dropping unknown raw
    attempts.  These metrics consume telemetry-only parser fields instead.
    """

    responses = [list(steps) for steps in parsed_responses]
    steps = [step for response in responses for step in response]
    n_steps = len(steps)
    known_surface_count = sum(len(getattr(step, "cited_triples", [])) for step in steps)
    unknown_surface_count = sum(
        len(getattr(step, "unknown_citation_surfaces", [])) for step in steps
    )
    known_steps = sum(bool(getattr(step, "cited_triples", [])) for step in steps)
    unknown_steps = sum(
        bool(getattr(step, "unknown_citation_surfaces", [])) for step in steps
    )
    raw_citing_steps = sum(
        bool(getattr(step, "cited_triples", []))
        or bool(getattr(step, "unknown_citation_surfaces", []))
        for step in steps
    )
    malformed_steps = sum(
        bool(getattr(step, "knowledge_used_malformed_content", False))
        for step in steps
    )
    contract_error_steps = sum(
        bool(getattr(step, "citation_contract_errors", [])) for step in steps
    )
    invalid_responses = sum(
        any(bool(getattr(step, "citation_contract_errors", [])) for step in response)
        for response in responses
    )
    recognised_surfaces = known_surface_count + unknown_surface_count
    return {
        "citation_contract_error_step_frac": contract_error_steps / max(1, n_steps),
        "citation_contract_invalid_response_frac": (
            invalid_responses / max(1, len(responses))
        ),
        "citation_raw_citing_step_frac": raw_citing_steps / max(1, n_steps),
        "citation_known_citing_step_frac": known_steps / max(1, n_steps),
        "citation_unknown_citing_step_frac": unknown_steps / max(1, n_steps),
        "citation_malformed_content_step_frac": malformed_steps / max(1, n_steps),
        "citation_known_surface_count": known_surface_count,
        "citation_unknown_surface_count": unknown_surface_count,
        "citation_known_frac_recognized_surfaces": (
            known_surface_count / recognised_surfaces
            if recognised_surfaces else None
        ),
    }


def _measure_explicit_reference_kl(
    trainer: StepRewardPPOTrainer,
    queries: Sequence[torch.Tensor],
    responses: Sequence[torch.Tensor],
) -> float:
    """Measure policy-vs-explicit-reference KL before any optimiser update."""

    if trainer.ref_model is None:
        raise RuntimeError("Explicit SFT reference is missing before PPO preflight")
    model_inputs = trainer.prepare_model_inputs(queries, responses)
    with torch.no_grad():
        policy_logprobs, _, _, masks = trainer.batched_forward_pass(
            trainer.model, queries, responses, model_inputs,
        )
        reference_logprobs, _, _, _ = trainer.batched_forward_pass(
            trainer.ref_model, queries, responses, model_inputs,
        )
    token_kl = (policy_logprobs - reference_logprobs) * masks
    return float(token_kl.sum(dim=-1).mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase3_ppo(cfg: Phase3PPOConfig) -> Dict[str, Any]:
    contract_v2 = _runtime_contract_v2(cfg)
    set_seed(cfg.seed)
    if cfg.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    _validate_mixed_reward_config(cfg)
    # Artifact validation precedes any CUDA/model allocation. Production
    # loading rejects synthetic or unvalidated calibration artifacts.
    from kgproweight.reward.source_gate_bounded_dispatch_v1 import load_referenced_bounded_before_dispatch
    source_quality_gate = load_referenced_bounded_before_dispatch(cfg)
    if cfg.source_gated_reward_version == "v1":
        if cfg.source_gate_credit_version == "v2":
            from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
            from kgproweight.training.reward_function import validate_source_gate_runtime_contract
            if source_quality_gate is None:
                source_quality_gate = SourceCreditGateV2.load(cfg.source_gate_calibration_path, runtime_config=cfg)
            validate_source_gate_runtime_contract(source_quality_gate, cfg.source_gate_format_version, cfg.source_gate_credit_version)
            from kgproweight.reward.source_gate_bounded_dispatch_v1 import validate_bounded_execution_paths
            validate_bounded_execution_paths(source_quality_gate, cfg)
        else:
            source_quality_gate = load_source_gate_for_runtime(
                cfg.source_gate_calibration_path, cfg.source_gate_format_version,
                cfg.source_gate_credit_version,
            )
    # Validate grouping before allocating any CUDA model.
    _sample_rollout_indices(1, cfg.batch_size, cfg.rollouts_per_prompt, torch.Generator())
    if cfg.fixed_rollout_schedule_path and cfg.total_steps % cfg.batch_size:
        raise ValueError(
            "fixed rollout schedule requires total_steps divisible by batch_size; "
            f"got {cfg.total_steps} and {cfg.batch_size}"
        )
    if (
        cfg.proofkg_process_reward
        or cfg.proofkg_outcome_only_reward
        or cfg.mixed_outcome_reward
    ) and not cfg.question_kg_records_path:
        raise ValueError(
            "automatic ProofKG or mixed outcome/process reward requires identity-safe "
            "question_kg_records_path with Gold-free provenance; legacy indexes are ineligible"
        )
    silver_path = Path(cfg.silver_path)
    if not silver_path.is_file():
        raise FileNotFoundError(f"silver_path does not exist: {silver_path}")
    if not cfg.sft_checkpoint or not Path(cfg.sft_checkpoint).is_dir():
        raise FileNotFoundError(
            f"sft_checkpoint must be an existing adapter directory: {cfg.sft_checkpoint}"
        )
    if cfg.sft_selection_report_path and not Path(cfg.sft_selection_report_path).is_file():
        raise FileNotFoundError(
            "sft_selection_report_path does not exist: "
            f"{cfg.sft_selection_report_path}"
        )
    if cfg.sft_replay_silver_path and not Path(cfg.sft_replay_silver_path).is_file():
        raise FileNotFoundError(
            f"sft_replay_silver_path does not exist: {cfg.sft_replay_silver_path}"
        )
    if cfg.alpha_override is None and not cfg.mixed_outcome_reward:
        if not cfg.alpha_gate_path or not Path(cfg.alpha_gate_path).is_file():
            raise FileNotFoundError(
                "alpha_gate_path must be an existing file for the main PPO run; "
                "refusing to continue with a randomly initialised gate."
            )
    if bool(cfg.question_kg_index_path) == bool(cfg.question_kg_records_path):
        raise ValueError(
            "PPO requires exactly one KG source: legacy question_kg_index_path "
            "or identity-safe question_kg_records_path"
        )
    qkg_path = Path(cfg.question_kg_index_path) if cfg.question_kg_index_path else None
    question_kg_records_path = (
        Path(cfg.question_kg_records_path) if cfg.question_kg_records_path else None
    )
    if qkg_path is not None and not qkg_path.is_file():
        raise FileNotFoundError(f"question_kg_index_path does not exist: {qkg_path}")
    if question_kg_records_path is not None and not question_kg_records_path.is_file():
        raise FileNotFoundError(
            f"question_kg_records_path does not exist: {question_kg_records_path}"
        )
    if cfg.split is None and not cfg.split_allow_none:
        raise ValueError(
            "training.split is None: PPO would train on the whole silver file; "
            "use split='train' or explicitly allow a curated train-only file"
        )
    if question_kg_records_path is not None:
        preflight_reader = SilverDatasetReader(
            cfg.silver_path,
            split=cfg.split,
            split_spec=cfg.build_split_spec() if cfg.split else None,
        )
        preflight_trajectories = list(preflight_reader.accepted())
        preflight_stats = apply_training_question_kg(
            preflight_trajectories,
            read_question_kg_records(question_kg_records_path),
            min_coverage=cfg.min_question_kg_record_coverage,
            require_nonempty=cfg.require_nonempty_question_kg_records,
        ).to_dict()
        if cfg.proofkg_process_reward and cfg.proofkg_process_version in {"v2_1", "v2_2", "v2_3"}:
            execution_stats = _validate_v21_execution_preflight(
                preflight_trajectories, version=cfg.proofkg_process_version,
            )
            preflight_stats[f"{cfg.proofkg_process_version}_execution"] = execution_stats
        logger.info("PPO question-KG CPU preflight: %s", preflight_stats)
        del preflight_reader, preflight_trajectories
    rollout_sampling_records: Dict[str, Dict[str, Any]] = {}
    if cfg.rollout_sampling_weights_path:
        sampling_reader = SilverDatasetReader(
            cfg.silver_path,
            split=cfg.split,
            split_spec=cfg.build_split_spec() if cfg.split else None,
        )
        sampling_population = [
            item for item in sampling_reader.accepted()
            if str(item.metadata.get("gold_answer") or "").strip()
        ]
        _, rollout_sampling_records = _load_rollout_sampling_weights(
            cfg.rollout_sampling_weights_path, sampling_population,
        )
        logger.info(
            "PPO weighted sampling CPU preflight: %d identity-safe rows, mass=1.0",
            len(rollout_sampling_records),
        )
        del sampling_reader, sampling_population
    fixed_rollout_indices: List[int] = []
    fixed_rollout_rows: List[Dict[str, Any]] = []
    if cfg.fixed_rollout_schedule_path:
        fixed_reader = SilverDatasetReader(
            cfg.silver_path,
            split=cfg.split,
            split_spec=cfg.build_split_spec() if cfg.split else None,
        )
        fixed_population = [
            item for item in fixed_reader.accepted()
            if str(item.metadata.get("gold_answer") or "").strip()
        ]
        fixed_rollout_indices, fixed_rollout_rows = _load_fixed_rollout_schedule(
            cfg.fixed_rollout_schedule_path,
            fixed_population,
            total_steps=cfg.total_steps,
            rollouts_per_prompt=cfg.rollouts_per_prompt,
            sampling_records=(
                rollout_sampling_records if cfg.rollout_sampling_weights_path else None
            ),
        )
        logger.info(
            "Fixed PPO rollout schedule CPU preflight: %d trajectories / %d prompt groups",
            len(fixed_rollout_indices),
            len(fixed_rollout_indices) // cfg.rollouts_per_prompt,
        )
        del fixed_reader, fixed_population
    passage_overrides, rollout_schedule_qids, hybrid_input_metadata = (
        _load_hybrid_rollout_inputs(cfg)
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 3b PPO requires CUDA, but torch.cuda.is_available() is False. "
            "Fix the NVIDIA driver/container runtime before reserving an Experiment ID."
        )
    if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 3b dtype=bf16 but the active GPU does not support bf16")

    out_dir, experiment_id = prepare_new_run_dir(
        cfg.output_dir,
        extra={
            "phase": "phase3_ppo",
            "config": asdict(cfg),
            "input_artifacts": {
                "silver": artifact_identity(silver_path),
                "sft_checkpoint": artifact_identity(cfg.sft_checkpoint),
                "sft_selection_report": (
                    artifact_identity(cfg.sft_selection_report_path)
                    if cfg.sft_selection_report_path else None
                ),
                "alpha_gate": (
                    artifact_identity(cfg.alpha_gate_path)
                    if cfg.alpha_gate_path and not cfg.mixed_outcome_reward else None
                ),
                "source_quality_gate": (
                    artifact_identity(cfg.source_gate_calibration_path)
                    if cfg.source_gated_reward_version == "v1" else None
                ),
                "question_kg_index": (
                    artifact_identity(qkg_path) if qkg_path is not None else None
                ),
                "question_kg_records": (
                    artifact_identity(question_kg_records_path)
                    if question_kg_records_path is not None else None
                ),
                "passage_overrides": (
                    artifact_identity(cfg.passage_overrides_path)
                    if cfg.passage_overrides_path else None
                ),
                "rollout_schedule": (
                    artifact_identity(cfg.rollout_schedule_path)
                    if cfg.rollout_schedule_path else None
                ),
                "rollout_sampling_weights": (
                    artifact_identity(cfg.rollout_sampling_weights_path)
                    if cfg.rollout_sampling_weights_path else None
                ),
                "rearag": (
                    artifact_identity(model_path("rearag"))
                    if (not cfg.mixed_outcome_reward or cfg.mixed_text_reward)
                    else None
                ),
                "fixed_rollout_schedule": (
                    artifact_identity(cfg.fixed_rollout_schedule_path)
                    if cfg.fixed_rollout_schedule_path else None
                ),
            },
            "hybrid_rollout_inputs": hybrid_input_metadata or None,
        },
    )

    from trl import PPOConfig

    policy, ref_model, tokenizer = _build_models(cfg)
    # NOTE: at this point the policy still lives on CPU — PPOTrainer/accelerate
    # move it to CUDA later. So don't read the device off the (CPU) policy here;
    # pick the real training device directly. Reward components (text_reward etc.)
    # built below must land on the SAME device the policy ends up on.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Build reward components ----------------------------------------
    alpha_gate = None
    annotator = None
    reward_subgraph_retriever = None
    if cfg.mixed_outcome_reward:
        logger.info(
            "Mixed reward fast path: legacy α-gate and PRM are not loaded or consumed"
        )
    else:
        alpha_gate = AlphaGate()
        if cfg.alpha_gate_path and Path(cfg.alpha_gate_path).exists():
            alpha_gate.load_state_dict(torch.load(cfg.alpha_gate_path, map_location="cpu"))
            logger.info("Loaded α-gate from %s", cfg.alpha_gate_path)
        elif cfg.alpha_override is None:
            raise RuntimeError("alpha gate preflight invariant was violated")
        alpha_gate.eval()

        # P0-2 / Finding 2: the legacy path needs the live entity cache so its
        # link-confidence feature matches Phase 2. The mixed path deliberately
        # skips this entire block: no AlphaGate, PRMAnnotator, or dynamic KG.
        from kgproweight.retrieval.bootstrap import resolve_entity_cache_path
        from kgproweight.kg.entity_linker import EntityLinker

        _entity_linker = EntityLinker(cache_path=resolve_entity_cache_path(), offline=True)
        logger.info(
            "PPO link_confidence: EntityLinker cache=%s (%d entries)",
            resolve_entity_cache_path(), len(list(_entity_linker.cache.items())),
        )
        annotator = PRMAnnotator(
            entity_linker=_entity_linker,
            min_subgraph_for_verify=cfg.prm_min_subgraph_for_verify,
            verbose=False,
        )
        logger.info(
            "PPO PRM sparse-verification threshold: %d triples",
            cfg.prm_min_subgraph_for_verify,
        )
        logger.info(
            "PPO KG reward backend: rule PRMAnnotator. The Phase-2 learned 3-way "
            "PRM head is diagnostic-only and is not loaded by this run."
        )
        reward_subgraph_retriever = WikidataSubgraphRetriever(
            max_hops=2,
            max_neighbors=30,
            cache_dir=str(Path(index_dir()) / "kg_cache"),
            offline=True,
            relation_filter=_QA_RELATION_FILTER,
        )
    text_reward_backend = (
        "rearag"
        if cfg.mixed_text_reward
        else "dummy"
        if cfg.pure_em_reward or cfg.mixed_outcome_reward or (
            (cfg.proofkg_process_reward or cfg.proofkg_outcome_only_reward)
            and cfg.proofkg_require_all_eligible
        )
        else cfg.text_reward_backend
    )
    if text_reward_backend == "dummy" and cfg.text_reward_backend != "dummy":
        logger.info(
            "Skipping text reward model because the selected fast-path does not "
            "consume it (pure outcome, mixed outcome, or all-eligible ProofKG)"
        )
    text_reward = build_text_reward_model(
        backend=text_reward_backend,
        fallback_head_path=cfg.text_reward_fallback_path,
        device=str(device),
        dtype=cfg.dtype,
    )
    _assert_mixed_text_backend(cfg, text_reward)
    reward_fn = KGProWeightRewardFunction(
        runtime_contract_version=cfg.runtime_contract_version,
        source_gated_reward_version=cfg.source_gated_reward_version,
        source_gate_format_version=cfg.source_gate_format_version,
        answer_format_reward_version=cfg.answer_format_reward_version,
        source_gate_credit_version=cfg.source_gate_credit_version,
        source_gate_mode=cfg.source_gate_mode,
        source_gate_calibration_path=cfg.source_gate_calibration_path,
        source_quality_gate=source_quality_gate,
        alpha_gate=alpha_gate,
        prm_annotator=annotator,
        text_reward_model=text_reward,
        tokenizer=tokenizer,
        outcome_weight=cfg.outcome_weight,
        discount=cfg.gamma,
        alpha_override=cfg.alpha_override,
        max_steps=cfg.max_steps,
        text_reward_scale=cfg.text_reward_scale,
        min_valid_steps=cfg.min_valid_steps,
        min_reasoning_chars=cfg.min_reasoning_chars,
        step_reward_scale=cfg.step_reward_scale,
        shortfall_coef=cfg.shortfall_coef,
        target_steps=cfg.target_steps,
        center_text_reward=cfg.center_text_reward,
        text_baseline_momentum=cfg.text_baseline_momentum,
        subgraph_retriever=reward_subgraph_retriever,
        pure_em=cfg.pure_em_reward,
        proofkg_process_reward=cfg.proofkg_process_reward,
        proofkg_outcome_only_reward=cfg.proofkg_outcome_only_reward,
        proofkg_process_version=cfg.proofkg_process_version,
        proofkg_process_weight=cfg.proofkg_process_weight,
        proofkg_f1_weight=cfg.proofkg_f1_weight,
        proofkg_dynamic_validity=cfg.proofkg_dynamic_validity,
        mixed_outcome_reward=cfg.mixed_outcome_reward,
        mixed_text_reward=cfg.mixed_text_reward,
    )

    # ---- Data ------------------------------------------------------------
    # Filtering at the reader is safe here: PPO reads trajectories to build
    # prompts and never writes them back out, so a narrowed reader cannot
    # truncate a downstream artefact the way it would in Phase 2.
    reader = SilverDatasetReader(
        cfg.silver_path,
        split=cfg.split,
        split_spec=cfg.build_split_spec() if cfg.split else None,
    )
    if cfg.split is None:
        # §13-1: this used to be a bare warning, and kg_proweight_ppo_v2 was
        # trained over its own eval folds because of it. Refuse by default.
        if not cfg.split_allow_none:
            raise ValueError(
                "training.split is None: PPO would roll out over the WHOLE silver "
                f"file ({len(reader.trajectories)} trajectories, "
                f"{len(reader.accepted())} accepted), including the val and test "
                "folds, making every downstream eval number leak-contaminated. "
                "Set training.split: train (the default since 2026-08-22), or pass "
                "--split_allow_none to deliberately reproduce a pre-split run."
            )
        logger.warning(
            "Phase 3b split: NONE (explicitly allowed via split_allow_none) — PPO "
            "rolls out over the whole file (%d trajectories, %d accepted). "
            "Nothing is held back; results MUST NOT be reported as held-out.",
            len(reader.trajectories), len(reader.accepted()),
        )
    else:
        logger.info(
            "Phase 3b split: fold=%s -> %d/%d trajectories, %d accepted "
            "(val=%.3f test=%.3f split_seed=%d)",
            cfg.split, len(reader.trajectories), reader.n_total_in_file,
            len(reader.accepted()), cfg.val_ratio, cfg.test_ratio,
            cfg.build_split_spec().seed,
        )
    if cfg.binary_labels_only:
        for traj in reader.trajectories:
            for step in traj.steps:
                # Continuous labels: everything that is not clearly positive
                # evidence collapses to negative for this ablation.
                if float(step.label) < 0.5:
                    step.label = -1.0

    question_kg_record_stats = None
    if cfg.question_kg_records_path:
        question_kg_record_stats = apply_training_question_kg(
            reader.accepted(),
            read_question_kg_records(cfg.question_kg_records_path),
            min_coverage=cfg.min_question_kg_record_coverage,
            require_nonempty=cfg.require_nonempty_question_kg_records,
        ).to_dict()
        logger.info("PPO question-KG override: %s", question_kg_record_stats)

    # R9: dynamic prompt KG disabled for speed (9839 prompts).
    # EntityLinker needed for reward-side dynamic KG (below).
    # R9: prompt KG uses silver data (instant). Reward-side dynamic KG provides
    # the KG signal (already works — α=0.85, r_kg broke zero). Prompt-side
    # injection needs pre-built entity index for speed; TODO separately.
    # R9 v6: pre-built Q→KG index with filtered & ranked triples.
    # Identity-safe question-KG records have already replaced traj.kg_subgraph
    # above.  Keep the legacy question-text index as None in that mode: passing
    # an empty dict would report every record as ABSENT, trip the inherited 0%
    # miss guard, and re-filter the deliberately short Proof-KGs.
    _q_kg_index: Optional[Dict[str, List[Tuple[str, str, str]]]] = None
    _q_kg_path = None
    if cfg.question_kg_index_path:
        _q_kg_index = {}
        _q_kg_path = Path(cfg.question_kg_index_path)
        # A relative path resolves against the CWD, and the launchers cd to
        # $REMOTE_ROOT before running, so "indexes/kg_cache/..." lands under the
        # project root there. Fall back to index_dir() (KGPW_INDEX_DIR) when the
        # CWD-relative path misses, so the same YAML works from either place
        # instead of failing on a box where only the env var is set.
        if not _q_kg_path.exists() and not _q_kg_path.is_absolute():
            _alt = Path(index_dir()) / _q_kg_path.name
            if _alt.exists():
                logger.info(
                    "question_kg_index_path %s not found relative to CWD; using "
                    "%s from KGPW_INDEX_DIR.", _q_kg_path, _alt,
                )
                _q_kg_path = _alt
        if not _q_kg_path.exists():
            raise FileNotFoundError(
                f"question_kg_index_path does not exist: {_q_kg_path} "
                f"(cwd={Path.cwd()}, index_dir={index_dir()}). It was "
                "requested explicitly, so falling back to a default index would "
                "silently train on a different KG than intended (§10.3). Build it "
                "with 06_build_question_kg_index.py --silver "
                f"{cfg.silver_path} --min_keep 5 --max_keep "
                f"{cfg.ppo_max_kg_triples}."
            )
    if _q_kg_path is not None and _q_kg_path.exists():
        import json as _json
        _q_kg_raw = _json.loads(_q_kg_path.read_text(encoding="utf-8"))
        is_v2 = "builder_version" in (_q_kg_raw[0] if _q_kg_raw else {})
        for _entry in _q_kg_raw:
            _q = _entry.get("question", _entry.get("q", ""))
            if is_v2:
                # v2 rich format: triples is list of dicts {h, pid, r, t, score}
                _q_kg_index[_q] = [(t["h"], t["r"], t["t"]) for t in _entry["triples"]]
            else:
                # v1 format: t is list of lists [h, r, t]
                _q_kg_index[_q] = _entry["t"]
        logger.info("R9 v6: Loaded %d question→KG entries from %s (v%s)",
                    len(_q_kg_index), _q_kg_path, "2" if is_v2 else "1")
    samples = _prepare_prompts(
        reader,
        tokenizer,
        cfg,
        question_kg_index=_q_kg_index,
        passage_overrides=passage_overrides,
    )
    if not samples:
        raise ValueError(f"No PPO samples derived from {cfg.silver_path}")
    rollout_sampling_weights: Optional[List[float]] = None
    rollout_sampling_strata: Dict[str, str] = {}
    if cfg.rollout_sampling_weights_path:
        rollout_sampling_weights, runtime_sampling_records = (
            _load_rollout_sampling_weights(cfg.rollout_sampling_weights_path, samples)
        )
        if set(runtime_sampling_records) != set(rollout_sampling_records):
            raise RuntimeError("rollout sampling population changed after model initialisation")
        rollout_sampling_strata = {
            key: str(value.get("stratum") or "")
            for key, value in runtime_sampling_records.items()
        }
        logger.info(
            "PPO prompt sampler: weighted identity-safe distribution from %s",
            cfg.rollout_sampling_weights_path,
        )
    proofkg_eligible_count = sum(
        (
            is_identity_safe_automatic_proofkg(
                row["spec"].metadata.get("question_kg_runtime") or {},
                row["spec"].kg_subgraph,
                dataset=row["spec"].metadata.get("dataset"),
                qid=row["spec"].metadata.get("qid"),
            )
            if cfg.mixed_outcome_reward
            else is_automatic_proofkg(
                row["spec"].metadata.get("question_kg_runtime") or {},
                row["spec"].kg_subgraph,
            )
        )
        for row in samples
    )
    logger.info(
        "Automatic ProofKG reward eligibility: %d/%d prompts (%.1f%%)",
        proofkg_eligible_count, len(samples),
        100.0 * proofkg_eligible_count / max(1, len(samples)),
    )
    if cfg.proofkg_require_all_eligible and proofkg_eligible_count != len(samples):
        raise ValueError(
            "proofkg_require_all_eligible=true but only "
            f"{proofkg_eligible_count}/{len(samples)} PPO prompts have a complete "
            "Gold-free automatic proof; refusing to use the unloaded legacy scorer"
        )

    # R7: prepare SFT anchor data from silver trajectories for format
    # preservation. We use accepted trajectories (including those without
    # gold answers — the anchor only cares about output format).
    sft_anchor_data: List[Dict[str, Any]] = []
    if cfg.sft_anchor_weight > 0:
        replay_silver_path = cfg.sft_replay_silver_path or cfg.silver_path
        sft_anchor_data = _prepare_sft_anchor_data(
            silver_path=replay_silver_path,
            tokenizer=tokenizer,
            cfg=cfg,
            max_samples=2000,
            apply_rollout_question_kg=(replay_silver_path == cfg.silver_path),
            replay_split=cfg.sft_replay_split,
        )
        logger.info(
            "Supervised trajectory replay: %d matched full-trace samples prepared "
            "from %s (sample_ratio=%.3f, loss_weight=%.3f; never mixed into PPO prompts)",
            len(sft_anchor_data), replay_silver_path,
            cfg.sft_replay_ratio, cfg.sft_anchor_weight,
        )
    if not 0.0 <= cfg.sft_replay_ratio <= 1.0:
        raise ValueError(
            f"sft_replay_ratio must be in [0, 1], got {cfg.sft_replay_ratio}"
        )
    if cfg.sft_replay_ratio > 0 and not sft_anchor_data:
        raise ValueError(
            "sft_replay_ratio is positive but no full-trajectory replay samples "
            f"could be built from {cfg.sft_replay_silver_path or cfg.silver_path}"
        )
    if (
        rollout_schedule_qids
        and cfg.sft_replay_ratio > 0
        and cfg.sft_anchor_weight > 0
        and len(sft_anchor_data) != 2000
    ):
        raise ValueError(
            "frozen rollout schedule assumes a 2000-item supervised replay pool; "
            f"runtime prepared {len(sft_anchor_data)} items"
        )
    # SCALE: total_steps counts trajectories SEEN (n_seen += batch_size per
    # iteration), not epochs. For one full pass over the data it should be
    # >= ceil(len(samples)/batch_size) * batch_size. We do NOT change the
    # semantics here; just report so under-coverage is visible.
    full_coverage_steps = math.ceil(len(samples) / max(1, cfg.batch_size)) * cfg.batch_size
    logger.info(
        "Phase 3b PPO with %d prompts; batch_size=%d, total_steps=%d "
        "(>= %d needed for one full pass over the data).",
        len(samples), cfg.batch_size, cfg.total_steps, full_coverage_steps,
    )

    # ---- Trainer ---------------------------------------------------------
    ppo_cfg = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        cliprange=cfg.cliprange,
        cliprange_value=cfg.cliprange_value,
        kl_penalty="kl",
        # KL control. init_kl_coef is the initial penalty coefficient for the
        # adaptive controller; cfg.target_kl is that controller's TARGET KL —
        # TRL's `target`, NOT TRL's `target_kl` (the latter is an early-stop
        # threshold that only fires when early_stopping=True). The previous code
        # wired cfg.target_kl into the early-stop knob (inert here) and left the
        # adaptive `target` at its default. Route it to the correct knob and make
        # adaptive control explicit.
        adap_kl_ctrl=True,
        init_kl_coef=cfg.kl_coef,
        target=cfg.target_kl,
        horizon=cfg.kl_horizon,
        gamma=cfg.gamma,
        lam=cfg.lam,
        max_grad_norm=cfg.max_grad_norm,
        vf_coef=cfg.vf_coef,
        early_stopping=cfg.early_stopping,
        log_with=None,  # R7: use custom tb_writer, not TRL's tracker
        seed=cfg.seed,
    )
    trainer = StepRewardPPOTrainer(
        config=ppo_cfg,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        runtime_contract_version=cfg.runtime_contract_version,
    )
    if not getattr(trainer, "_uses_explicit_reference", False):
        raise RuntimeError(
            "PPO trainer did not retain the explicit frozen SFT reference. "
            "Refusing to train against a bare-base reference."
        )

    # One writer owns custom and TRL metrics (the TRL tracker is disabled above).
    tb_writer = None
    if cfg.log_with == "tensorboard":
        tb_writer, tb_record = create_ppo_writer(out_dir, experiment_id)
        log_run_metadata(tb_writer, config=asdict(cfg), metadata=tb_record)
        tb_writer.flush()
        logger.info("TensorBoard logging to %s", tb_record["log_dir"])

    # ---- Loop ------------------------------------------------------------
    rng = torch.Generator().manual_seed(cfg.seed)
    n_seen = 0
    replay_credit = 0.0
    replay_items_seen = 0
    history: List[Dict[str, float]] = []
    # Keep the most recent 20 rollouts across batches.  The old checkpoint
    # sampler wrote ``response_texts[:20]`` but batch_size=4, so each file held
    # only four examples and the three smoke checkpoints together exposed just
    # 12 outputs.  This rolling buffer changes telemetry only.
    sample_buffer: List[Dict[str, Any]] = []
    while n_seen < cfg.total_steps:
        batch_started = time.perf_counter()
        # PPO always rolls out from ordinary question prompts. Supervised replay
        # is a separate CE update after trainer.step(), so gold trajectories can
        # never leak into generation or be scored against another question.
        prompts: List[str] = []
        specs: List[RewardSpec] = []
        explore_idx = _select_rollout_batch_indices(
            population_size=len(samples),
            batch_size=cfg.batch_size,
            rollouts_per_prompt=cfg.rollouts_per_prompt,
            generator=rng,
            sampling_weights=rollout_sampling_weights,
            fixed_indices=fixed_rollout_indices,
            offset=n_seen,
        )
        if fixed_rollout_indices:
            expected_keys = [
                question_key(
                    str(row.get("dataset") or ""), str(row.get("qid") or "")
                )
                for row in fixed_rollout_rows[n_seen : n_seen + cfg.batch_size]
            ]
            actual_keys = [
                question_key(
                    str(samples[index]["spec"].metadata.get("dataset") or ""),
                    str(samples[index]["spec"].metadata.get("qid") or ""),
                )
                for index in explore_idx
            ]
            if actual_keys != expected_keys:
                raise RuntimeError(
                    "fixed rollout schedule population drifted after CPU preflight: "
                    f"step={n_seen + cfg.batch_size}"
                )
        selected_strata = [
            rollout_sampling_strata.get(
                question_key(
                    str(samples[index]["spec"].metadata.get("dataset") or ""),
                    str(samples[index]["spec"].metadata.get("qid") or ""),
                ),
                "",
            )
            for index in explore_idx
        ]
        if rollout_schedule_qids:
            actual_qids = [samples[index]["spec"].metadata["qid"] for index in explore_idx]
            expected_qids = rollout_schedule_qids[n_seen : n_seen + cfg.batch_size]
            if actual_qids != expected_qids:
                raise RuntimeError(
                    "PPO runtime rollout schedule drifted after preflight: "
                    f"step={n_seen + cfg.batch_size} actual={actual_qids} "
                    f"expected={expected_qids}"
                )
            if any(
                not samples[index]["spec"].metadata.get("passage_override_applied")
                for index in explore_idx
            ):
                raise RuntimeError(
                    "PPO selected a rollout without its required passage override"
                )
        for i in explore_idx:
            prompts.append(samples[i]["prompt"])
            specs.append(samples[i]["spec"])

        # R10 timing: the smoke run measured 80.8 s/update (→44.9 h for 16000
        # trajectories) but only timestamped whole updates, so the split across
        # the three serial stages was unknown and every speed-up argument was a
        # guess about which one dominates. Time them separately. The stages have
        # very different per-forward counts at batch_size=8 / max_new_tokens=256:
        #   rollout  ~8x250 = 2000 single-sample decode steps (KV-cached, cheap
        #            each, but batch=1 -- _generate loops one prompt at a time)
        #   reward   ~1 ReaRAG-9B forward per parsed step (~19/batch), serial
        #            (text_reward_model.score_steps is a Python loop)
        #   ppo      batch_size/mini_batch_size * ppo_epochs = 16 fwd+bwd, each
        #            recomputing activations under gradient checkpointing
        # Counting forwards alone does not rank them -- a cached decode step and
        # a checkpointed backward differ by orders of magnitude -- which is
        # exactly why this is measured rather than reasoned about.
        _t0 = time.perf_counter()
        query_tensors, response_tensors, response_texts, logprobs_per_step_list = _generate(
            policy, tokenizer, prompts, cfg, device
        )
        if n_seen == 0:
            preupdate_reference_kl = _measure_explicit_reference_kl(
                trainer, query_tensors, response_tensors,
            )
            logger.info(
                "Pre-update explicit-SFT-reference KL: %.6f "
                "(required finite and abs<=1.0)",
                preupdate_reference_kl,
            )
            if (
                not math.isfinite(preupdate_reference_kl)
                or abs(preupdate_reference_kl) > 1.0
            ):
                raise RuntimeError(
                    "Pre-update policy/reference KL invariant failed before any "
                    f"optimizer step: KL={preupdate_reference_kl}."
                )
        response_token_counts = [int(t.numel()) for t in response_tensors]
        # If no EOS was emitted, generation returns exactly max_new_tokens.  This
        # conservative flag identifies responses that may have been cut by the
        # generation ceiling; it does not alter or re-decode them.
        if contract_v2:
            eos_ids = _rollout_eos_token_ids(policy, tokenizer)
            length_capped_flags = [
                int(_response_is_length_capped_v2(
                    ids, max_new_tokens=cfg.max_new_tokens, eos_token_ids=eos_ids,
                ))
                for ids in response_tensors
            ]
        else:
            length_capped_flags = [
                int(n_tokens >= cfg.max_new_tokens)
                for n_tokens in response_token_counts
            ]
        for text, ids, spec, capped in zip(
            response_texts, response_tensors, specs, length_capped_flags
        ):
            sample_buffer.append({
                "qid": str(spec.metadata.get("qid") or ""),
                "response_tokens": int(ids.numel()),
                "length_capped": bool(capped),
                "text": text,
            })
        sample_buffer = sample_buffer[-20:]
        _t_rollout = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        token_reward_list: List[torch.Tensor] = []
        traj_rewards: List[float] = []
        reward_infos: List[Dict[str, Any]] = []
        all_per_step_records = []  # R9: collect from all responses
        parsed_contract_responses: List[List[Any]] = []
        for resp_text, response_ids, spec, logprobs_per_step in zip(
            response_texts, response_tensors, specs, logprobs_per_step_list
        ):
            # #6: compute step spans in response_ids coordinates ONCE; the reward
            # fn places per-step rewards on those same spans, so the placement
            # matches the trainer's scatter (no decode∘re-tokenise drift). The
            # entropy logprobs were already bucketed onto these spans in
            # _generate (SCALE: done per-prompt to avoid holding raw logits).
            n_parsed = len(parse_steps(resp_text)[: cfg.max_steps])
            parsed_for_contract = parse_steps(
                resp_text, known_kg=spec.kg_subgraph,
            )[: cfg.max_steps]
            parsed_contract_responses.append(parsed_for_contract)
            aligned_spans = step_spans_over_ids(response_ids, tokenizer, n_parsed)
            info = reward_fn(
                prompt="",
                response=resp_text,
                spec=spec,
                logprobs_per_step=logprobs_per_step,
                response_ids=response_ids,
                step_spans=aligned_spans,
            )
            reward_infos.append(info)
            traj_rewards.append(info["trajectory_reward"])
            all_per_step_records.extend(info.get("per_step_records", []))
            # token_rewards is already built in response_ids space (#6), so it
            # matches the response tensor length exactly — but stay defensive.
            tr = _align_token_rewards(
                info["token_rewards"], response_ids,
                trajectory_reward=info["trajectory_reward"],
                runtime_contract_version=cfg.runtime_contract_version,
            )
            token_reward_list.append(tr)

        # ── R9: extract reward components from all responses for diagnostics ──
        reward_rc = {"alpha_mean": 0.0, "r_kg_mean": 0.0, "r_text_mean": 0.0,
                     "r_total_mean": 0.0, "n_steps": 0, "kg_reward_share": 0.0,
                     # §9.4-1 (量纲): the centered channel and its baseline.
                     "r_text_used_mean": 0.0, "text_baseline": 0.0,
                     "text_baseline_n_obs": 0,
                     # d r_total / d alpha, the quantity the fix exists to move.
                     "dr_dalpha": 0.0,
                     # §9.6: direct evidence for the "r_kg is sparse" claim, which
                     # until now rested only on the batch-aggregated mean.
                     "r_kg_zero_frac": 0.0}
        if all_per_step_records:
            reward_rc["alpha_mean"] = float(sum(r.alpha for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["r_kg_mean"] = float(sum(r.r_kg for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["r_text_mean"] = float(sum(r.r_text for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["r_total_mean"] = float(sum(r.r_total for r in all_per_step_records) / len(all_per_step_records))
            reward_rc["n_steps"] = len(all_per_step_records)
            _n = len(all_per_step_records)
            # ── §9.4-1 (量纲) diagnostics ──────────────────────────────────
            # r_text_used is what actually entered r_total; r_text is the raw
            # scorer output. With centering ON the first must sit near 0 and the
            # second near +0.63 -- if r_text_used_mean is also near +0.63 the
            # centering is not running (config not forwarded), and if r_text_mean
            # is near 0 the SCORER has changed, not the centering. Logging both
            # is what makes those two cases distinguishable.
            reward_rc["r_text_used_mean"] = float(
                sum(r.r_text_used for r in all_per_step_records) / _n)
            reward_rc["text_baseline"] = float(reward_fn.composite.text_baseline)
            reward_rc["text_baseline_n_obs"] = int(
                reward_fn.composite.text_baseline_n_obs
            )
            # The sensitivity of reward to the gate:
            #   d r_total / d alpha = (r_kg - c_text * r_text_used) * c_step
            # MEASURED at -0.148 before the fix (r_kg 0.0896, r_text 0.6284,
            # c_text 0.3, c_step 1.5) -- the reward was paying the policy to lower
            # alpha, i.e. to cite a sparser subgraph. After centering this should
            # hover around 0 with the sign varying batch to batch. A persistent
            # negative value means the KG channel is still being outbid.
            reward_rc["dr_dalpha"] = float(
                sum(r.r_kg - cfg.text_reward_scale * r.r_text_used
                    for r in all_per_step_records) / _n) * cfg.step_reward_scale
            # §9.6: what fraction of step records score EXACTLY 0 on r_kg (the
            # PRM's NEUTRAL branch -- "the subgraph cannot judge this step").
            # The claim that r_kg_mean 0.0896 is driven by frequency rather than
            # by accuracy was inferred from the batch mean plus the 13/13
            # precision reading; this measures it directly. Note 0 is a legitimate
            # label here (prm_annotator.py:192 C2), not a missing value.
            reward_rc["r_kg_zero_frac"] = float(
                sum(1 for r in all_per_step_records if r.r_kg == 0.0) / _n)
            # What fraction of the batch's total reward magnitude the KG channel
            # actually contributes. Log it so "we trained with a KG process
            # reward" is a measured claim rather than an architectural one.
            #
            # R10 (2026-08-06): under the old outcome_weight=10 /
            # step_reward_scale=0.3 this was 0.9%, not the "near 3%" the previous
            # comment estimated. PPO was optimising format-validity and EM
            # essentially alone. Now 4.0/1.5, and the smoke run MEASURED a mean
            # share of 0.118 (two batches, 0.024 and 0.213) — on target.
            #
            # The mechanism is not what the earlier version of this comment
            # assumed. It reasoned from r_kg≈0.15 as if citation quality were the
            # binding factor; the diagnostics say otherwise:
            #   - precision of cited triples against the real subgraph: 13/13,
            #     stable from fuzzy_threshold 0.95 down to 0.50
            #   - relevance factor: mean 0.556
            # so r_kg per CITING step is high. What holds r_kg_mean near 0.10 is
            # the denominator: r_kg_mean averages over ALL step records, and
            # roughly half of them cite nothing at all and score a NEUTRAL 0.
            # The lever on this share is therefore citation FREQUENCY, not
            # citation accuracy — do not "improve" it by tightening the matcher.
            #
            # Single-batch readings are near-useless here: at batch_size=8 the
            # share swung 0.024 → 0.213 between consecutive logged batches. Judge
            # it on a mean over 3+ batches (check_ppo_smoke.sh does this).
            #
            # `max(1, ...)` below is deliberate but easy to misread: the second
            # argument is a bool, so the expression is 1 whenever no trajectory
            # scored and 1 when some did — i.e. always 1. The denominator
            # therefore always includes one outcome_weight, which is the point:
            # it keeps the share comparable across batches instead of spiking on
            # all-zero-reward batches. Kept as-is so R10 numbers stay comparable
            # with the r9 history; do not "fix" it without re-baselining.
            kg_mass = sum(abs(r.alpha * r.r_kg) for r in all_per_step_records) * cfg.step_reward_scale
            total_mass = sum(abs(r.r_total) for r in all_per_step_records) + \
                cfg.outcome_weight * max(1, sum(traj_rewards) > 0)
            reward_rc["kg_reward_share"] = float(kg_mass / total_mass) if total_mass else 0.0
        elif cfg.mixed_text_reward:
            # The mixed route deliberately has no legacy StepReward records.
            # Populate the long-standing text diagnostics from its explicit
            # ReaRAG telemetry so history/TensorBoard cannot misleadingly show
            # a permanently-zero text channel.
            mixed_rows = [
                info.get("mixed_reward") for info in reward_infos
                if isinstance(info.get("mixed_reward"), dict)
            ]
            raw_scores = [
                float(value) for row in mixed_rows
                for value in row.get("text_raw_step_scores", [])
            ]
            centered_scores = [
                float(value) for row in mixed_rows
                for value in row.get("text_centered_clipped_step_scores", [])
            ]
            if raw_scores:
                reward_rc["r_text_mean"] = sum(raw_scores) / len(raw_scores)
                reward_rc["r_text_used_mean"] = (
                    sum(centered_scores) / len(centered_scores)
                )
                reward_rc["n_steps"] = len(raw_scores)
            reward_rc["text_baseline"] = float(reward_fn.composite.text_baseline)
            reward_rc["text_baseline_n_obs"] = int(
                reward_fn.composite.text_baseline_n_obs
            )

        reward_rc.update(_citation_reward_diagnostics(all_per_step_records))
        reward_rc.update(_citation_contract_diagnostics(parsed_contract_responses))

        # P0-1 / #6: hand the per-token step rewards to the trainer so GAE runs
        placeholder_scores = [torch.zeros((), dtype=torch.float32) for _ in token_reward_list]
        trainer.set_pending_step_rewards(token_reward_list)
        _t_reward = time.perf_counter() - _t0
        _t0 = time.perf_counter()
        stats = trainer.step(query_tensors, response_tensors, placeholder_scores)
        _t_ppo = time.perf_counter() - _t0
        # One line per update, so the split is visible in the log without a
        # separate profiling run and can be re-checked after any config change.
        logger.info(
            "TIMING upd=%d rollout=%.1fs reward=%.1fs ppo=%.1fs total=%.1fs",
            n_seen // max(1, cfg.batch_size),
            _t_rollout, _t_reward, _t_ppo,
            _t_rollout + _t_reward + _t_ppo,
        )
        n_seen += cfg.batch_size

        # Correctness gate, not a tunable evaluation threshold. ``objective/kl``
        # is computed before the first optimiser update. Because policy and the
        # explicit frozen SFT reference are exact copies at startup, the first
        # sequence-level KL must be approximately zero. The historical value
        # 65.44 proved TRL had silently disabled LoRA and compared against the
        # bare base. Abort after the first batch if that regression returns.
        if n_seen == cfg.batch_size:
            try:
                initial_reference_kl = float(stats.get("objective/kl", 0.0))
            except (TypeError, ValueError):
                initial_reference_kl = float("nan")
            logger.info(
                "Initial explicit-SFT-reference KL check: objective/kl=%.6f "
                "(required finite and abs<=1.0)",
                initial_reference_kl,
            )
            if (
                not math.isfinite(initial_reference_kl)
                or abs(initial_reference_kl) > 1.0
            ):
                raise RuntimeError(
                    "Initial policy/reference KL invariant failed: "
                    f"objective/kl={initial_reference_kl}. Policy and the frozen "
                    "SFT reference start from identical weights, so this indicates "
                    "the wrong reference path or a scoring mismatch."
                )

        # ── Matched full-trajectory supervised replay ---------------------
        # Ratio is defined in samples, not batches. At batch=4 and ratio=.10,
        # fractional credit schedules 0/0/1/0/1... replay items: exactly two
        # supervised samples per twenty PPO samples in the long run.
        sft_loss_val = 0.0
        replay_items_this_update = 0
        if cfg.sft_replay_ratio > 0 and cfg.sft_anchor_weight > 0:
            replay_items_this_update, replay_credit = _advance_replay_credit(
                replay_credit,
                batch_size=cfg.batch_size,
                replay_ratio=cfg.sft_replay_ratio,
            )
        elif (
            cfg.sft_replay_ratio == 0
            and cfg.sft_anchor_interval > 0
            and cfg.sft_anchor_weight > 0
            and sft_anchor_data
            and n_seen % (cfg.batch_size * cfg.sft_anchor_interval) == 0
        ):
            # Explicit legacy reproduction only.
            replay_items_this_update = 1

        if replay_items_this_update > 0:
            inner = getattr(policy, "pretrained_model", policy)
            # Read live device (trainer may have moved the policy).
            try:
                _dev = next(p for p in policy.parameters()).device
            except StopIteration:
                _dev = torch.device(device)

            replay_idx = torch.randint(
                0, len(sft_anchor_data), (replay_items_this_update,), generator=rng,
            ).tolist()
            replay_losses: List[float] = []
            trainer.optimizer.zero_grad()
            for sft_idx in replay_idx:
                item = sft_anchor_data[sft_idx]
                sft_input_ids = torch.tensor(
                    [item["input_ids"]], dtype=torch.long, device=_dev,
                )
                sft_labels = torch.tensor(
                    [item["labels"]], dtype=torch.long, device=_dev,
                )
                sft_out = inner(input_ids=sft_input_ids, labels=sft_labels)
                replay_losses.append(float(sft_out.loss.item()))
                weighted = (
                    cfg.sft_anchor_weight
                    * sft_out.loss
                    / replay_items_this_update
                )
                weighted.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            sft_loss_val = sum(replay_losses) / len(replay_losses)
            replay_items_seen += replay_items_this_update
            logger.info(
                "Supervised replay step=%d items=%d cumulative=%d/%d PPO samples "
                "(%.3f actual) mean_loss=%.4f weight=%.3f",
                n_seen, replay_items_this_update, replay_items_seen, n_seen,
                replay_items_seen / max(1, n_seen), sft_loss_val,
                cfg.sft_anchor_weight,
            )

        # Collect trajectory validity stats for monitoring (R8: content-aware).
        # Reward owns the validity protocol (including input-conditioned
        # ProofKG hop counts), so telemetry must consume its decision instead of
        # recomputing the legacy fixed-three-step predicate.
        n_valid = sum(bool(info.get("trajectory_valid")) for info in reward_infos)
        mixed_reward_by_dataset = _mixed_reward_dataset_diagnostics(reward_infos)
        mixed_text_batch = _mixed_text_batch_diagnostics(mixed_reward_by_dataset)
        source_gate_batch = _source_gate_batch_diagnostics(reward_infos)
        if source_gate_batch:
            reward_rc["alpha_mean"] = source_gate_batch["source_gate_alpha_effective_mean"]
            reward_rc["r_kg_mean"] = source_gate_batch["source_gate_graph_normalized_mean"]
            reward_rc["r_text_used_mean"] = source_gate_batch["source_gate_text_normalized_mean"]
            reward_rc["r_total_mean"] = (
                source_gate_batch["source_gate_text_component_mean"]
                + source_gate_batch["source_gate_graph_component_mean"]
            )
        proof_infos = [
            info["proofkg_process"] for info in reward_infos
            if info.get("proofkg_process")
        ]
        eligible_proof_infos = [
            row for row in proof_infos if bool(row.get("eligible"))
        ]
        proofkg_diag = {
            "proofkg_eligible_count": len(eligible_proof_infos),
            "proofkg_eligible_frac": len(eligible_proof_infos) / max(1, cfg.batch_size),
            "proofkg_process_applied_count": sum(
                bool(row.get("process_applied")) for row in proof_infos
            ),
            "proofkg_process_mean": (
                sum(row["process_score"] for row in eligible_proof_infos)
                / len(eligible_proof_infos)
                if eligible_proof_infos else None
            ),
            "proofkg_outcome_em_mean": (
                sum(row["outcome_em"] for row in eligible_proof_infos)
                / len(eligible_proof_infos)
                if eligible_proof_infos else None
            ),
            "proofkg_outcome_f1_mean": (
                sum(row["outcome_f1"] for row in eligible_proof_infos)
                / len(eligible_proof_infos)
                if eligible_proof_infos else None
            ),
            "mixed_outcome_em_mean": (
                sum(row["outcome_em"] for row in proof_infos) / len(proof_infos)
                if cfg.mixed_outcome_reward and proof_infos else None
            ),
            "mixed_outcome_f1_mean": (
                sum(row["outcome_f1"] for row in proof_infos) / len(proof_infos)
                if cfg.mixed_outcome_reward and proof_infos else None
            ),
        }
        mixed_groups = 0
        n_groups = 0
        if cfg.rollouts_per_prompt > 1:
            for start in range(0, len(reward_infos), cfg.rollouts_per_prompt):
                group = reward_infos[start : start + cfg.rollouts_per_prompt]
                outcomes = [
                    float((item.get("proofkg_process") or {}).get("outcome_em", 0.0))
                    for item in group
                ]
                n_groups += 1
                mixed_groups += int(bool(outcomes) and min(outcomes) < max(outcomes))
        proofkg_diag["within_prompt_mixed_outcome_frac"] = (
            mixed_groups / n_groups if n_groups else None
        )

        # Advantage diagnostics directly from our custom trainer's last batch.
        # `advantage_var` is PRE-whitening; the historical post-whitening value
        # was ~1 by construction and could not reveal reward/GAE oscillation.
        adv_var = getattr(trainer, "_last_adv_var", 0.0)
        adv_stats = getattr(trainer, "_last_adv_stats", {})

        def _stat(key):
            """Pull a TRL stat as a python float (handles numpy scalars/arrays)."""
            v = stats.get(key)
            if v is None:
                return None
            try:
                import numpy as _np
                return float(_np.asarray(v).mean())
            except Exception:
                return None

        objective_kl = _stat("objective/kl")
        response_tokens_mean = float(
            sum(response_token_counts) / max(1, len(response_token_counts))
        )
        try:
            adaptive_kl_coef = float(trainer.kl_ctl.value)
        except (AttributeError, TypeError, ValueError):
            adaptive_kl_coef = None

        history.append(
            {
                "step": n_seen,
                "mean_reward": float(sum(traj_rewards) / max(1, len(traj_rewards))),
                "ppo_mean_kl": objective_kl,
                "adaptive_kl_coef": adaptive_kl_coef,
                "uses_explicit_sft_reference": bool(
                    getattr(trainer, "_uses_explicit_reference", False)
                ),
                # TRL objective/kl is sequence-aggregated.  Dividing by the
                # observed mean response length is an explicit estimate, not a
                # replacement for the frozen sequence-level metric.
                "objective_kl_per_response_token_estimate": (
                    objective_kl / response_tokens_mean
                    if objective_kl is not None and response_tokens_mean > 0
                    else None
                ),
                "response_tokens_mean": response_tokens_mean,
                "length_capped_count": sum(length_capped_flags),
                "length_capped_frac": (
                    sum(length_capped_flags) / max(1, len(length_capped_flags))
                ),
                "advantage_var": adv_var,
                "advantage_raw_mean": adv_stats.get("raw_mean"),
                "advantage_raw_std": adv_stats.get("raw_std"),
                "advantage_raw_min": adv_stats.get("raw_min"),
                "advantage_raw_max": adv_stats.get("raw_max"),
                "advantage_raw_p50": adv_stats.get("raw_p50"),
                "advantage_raw_p90": adv_stats.get("raw_p90"),
                "advantage_raw_p95": adv_stats.get("raw_p95"),
                "advantage_raw_p99": adv_stats.get("raw_p99"),
                "advantage_whitened_var": adv_stats.get("whitened_var"),
                "value_mean": adv_stats.get("value_mean"),
                "value_std": adv_stats.get("value_std"),
                "return_mean": adv_stats.get("return_mean"),
                "return_std": adv_stats.get("return_std"),
                "explained_variance": adv_stats.get("explained_variance"),
                # PPO losses (so loss/clip curves are recoverable from history.jsonl).
                "loss_total": _stat("ppo/loss/total"),
                "loss_policy": _stat("ppo/loss/policy"),
                "loss_value": _stat("ppo/loss/value"),
                "policy_clipfrac": _stat("ppo/policy/clipfrac"),
                "policy_entropy": _stat("ppo/policy/entropy"),
                "policy_approxkl": _stat("ppo/policy/approxkl"),
                # R7: trajectory validity monitoring.
                "n_valid": n_valid,
                "valid_rate": n_valid / max(1, cfg.batch_size),
                "hard_curriculum_recovery_frac": (
                    sum(value == "recovery" for value in selected_strata)
                    / max(1, len(selected_strata))
                    if rollout_sampling_weights is not None else None
                ),
                "rollout_qids": [
                    str(samples[index]["spec"].metadata.get("qid") or "")
                    for index in explore_idx
                ],
                "rollout_strata": selected_strata,
                **proofkg_diag,
                # Mixed PPO fast-path records no legacy per_step_records, so
                # preserve its outcome/text/KG decomposition explicitly.
                "mixed_reward_by_dataset": mixed_reward_by_dataset,
                **mixed_text_batch,
                **source_gate_batch,
                "sft_anchor_loss": sft_loss_val,
                "sft_replay_loss": sft_loss_val,
                "sft_replay_items": replay_items_this_update,
                "sft_replay_items_seen": replay_items_seen,
                "sft_replay_actual_ratio": replay_items_seen / max(1, n_seen),
                # R9: reward component diagnostics
                "alpha_mean": reward_rc["alpha_mean"],
                "r_kg_mean": reward_rc["r_kg_mean"],
                "r_text_mean": reward_rc["r_text_mean"],
                "r_total_mean": reward_rc["r_total_mean"],
                "n_steps_sample": reward_rc["n_steps"],
                "kg_reward_share": reward_rc["kg_reward_share"],
                # §9.4-1 (量纲): centered text reward diagnostics.
                "r_text_used_mean": reward_rc["r_text_used_mean"],
                "text_baseline": reward_rc["text_baseline"],
                "text_baseline_n_obs": reward_rc["text_baseline_n_obs"],
                "dr_dalpha": reward_rc["dr_dalpha"],
                "r_kg_zero_frac": reward_rc["r_kg_zero_frac"],
                "cite_any_step_frac": reward_rc["cite_any_step_frac"],
                "cite_match_mean_citing_step": reward_rc["cite_match_mean_citing_step"],
                "cite_match_mean_reward_visible_citing_step": reward_rc["cite_match_mean_reward_visible_citing_step"],
                "cite_unknown_only_step_frac_citing": reward_rc["cite_unknown_only_step_frac_citing"],
                "cite_partial_match_step_frac_citing": reward_rc["cite_partial_match_step_frac_citing"],
                "cite_all_matched_step_frac_citing": reward_rc["cite_all_matched_step_frac_citing"],
                "alpha_mean_no_cite_step": reward_rc["alpha_mean_no_cite_step"],
                "alpha_mean_known_cite_step": reward_rc["alpha_mean_known_cite_step"],
                "alpha_mean_unknown_cite_step": reward_rc["alpha_mean_unknown_cite_step"],
                "r_kg_zero_frac_no_cite_step": reward_rc["r_kg_zero_frac_no_cite_step"],
                "r_kg_zero_frac_known_cite_step": reward_rc["r_kg_zero_frac_known_cite_step"],
                "r_kg_zero_frac_unknown_cite_step": reward_rc["r_kg_zero_frac_unknown_cite_step"],
                "citation_contract_error_step_frac": reward_rc["citation_contract_error_step_frac"],
                "citation_contract_invalid_response_frac": reward_rc["citation_contract_invalid_response_frac"],
                "citation_raw_citing_step_frac": reward_rc["citation_raw_citing_step_frac"],
                "citation_known_citing_step_frac": reward_rc["citation_known_citing_step_frac"],
                "citation_unknown_citing_step_frac": reward_rc["citation_unknown_citing_step_frac"],
                "citation_malformed_content_step_frac": reward_rc["citation_malformed_content_step_frac"],
                "citation_known_surface_count": reward_rc["citation_known_surface_count"],
                "citation_unknown_surface_count": reward_rc["citation_unknown_surface_count"],
                "citation_known_frac_recognized_surfaces": reward_rc["citation_known_frac_recognized_surfaces"],
            }
        )
        immediate_health_failure = _nonfinite_training_state_reason(history[-1])
        if n_seen % (cfg.batch_size * 4) == 0:
            # R10: kg_share and upd are on this line deliberately.
            #   kg_share — the fraction of reward magnitude the KG channel
            #     carries. It was 0.009 under outcome_weight=10 /
            #     step_reward_scale=0.3, i.e. the process reward the method is
            #     named after was numerically noise, and nobody noticed because
            #     the number only existed in tensorboard and history.jsonl.
            #   upd — n_seen counts TRAJECTORIES; every past run's "step 500"
            #     was really 63 optimiser updates. Print both so the horizontal
            #     axis can never be misread again.
            logger.info(
                "step=%d (upd=%d) reward=%.2f kl=%.1f clip=%.3f valid=%d/%d "
                "α=%.3f r_kg=%.3f r_text=%.3f (used %.3f base %.3f) "
                "kg_share=%.3f dR/dα=%.3f r_kg_0=%d%% n_steps=%d",
                n_seen,
                n_seen // max(1, cfg.batch_size),
                history[-1]["mean_reward"],
                history[-1]["ppo_mean_kl"],
                history[-1]["policy_clipfrac"] if history[-1]["policy_clipfrac"] is not None else float("nan"),
                n_valid,
                cfg.batch_size,
                reward_rc["alpha_mean"],
                reward_rc["r_kg_mean"],
                reward_rc["r_text_mean"],
                reward_rc["r_text_used_mean"],
                reward_rc["text_baseline"],
                reward_rc["kg_reward_share"],
                reward_rc["dr_dalpha"],
                int(100 * reward_rc["r_kg_zero_frac"]),
                reward_rc["n_steps"],
            )
            # R7: dump one sample response for qualitative monitoring.
            if response_texts:
                sample = response_texts[0][:500]
                logger.info("  [sample] %s", sample.replace("\n", "\\n"))

        # ── R8: Periodic reasoning-content sampling ──
        # Every save_every_steps, sample 20 responses and check Reasoning
        # content rate so we can see whether the content gate is working.
        sample_log: Optional[Dict[str, Any]] = None
        if (
            cfg.save_every_steps > 0
            and (n_seen // cfg.save_every_steps) != ((n_seen - cfg.batch_size) // cfg.save_every_steps)
        ):
            sampled_records = list(sample_buffer)
            sampled_texts = [record["text"] for record in sampled_records]
            sample_log = _count_reasoning_content(
                sampled_texts, min_chars=cfg.min_reasoning_chars,
            )
            sample_log["step"] = n_seen
            logger.info(
                "R8 sample step=%d: step_rate=%.2f fa_rate=%.2f reasoning_content=%.2f "
                "(%d/%d steps have >=%d chars)",
                n_seen,
                sample_log["step_rate"],
                sample_log["final_answer_rate"],
                sample_log["reasoning_content_rate"],
                sample_log["steps_with_content"],
                sample_log["total_steps"],
                cfg.min_reasoning_chars,
            )
            # Dump samples to disk for offline inspection.
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / f"step_{n_seen:05d}.txt"
            with open(sample_path, "w", encoding="utf-8") as sfh:
                sfh.write(f"# Step {n_seen} — Reasoning content sampling\n")
                sfh.write(f"# step_rate={sample_log['step_rate']:.2f} ")
                sfh.write(f"fa_rate={sample_log['final_answer_rate']:.2f} ")
                sfh.write(f"reasoning_content={sample_log['reasoning_content_rate']:.2f}\n\n")
                for i, record in enumerate(sampled_records, 1):
                    sfh.write(
                        f"--- Sample {i} qid={record['qid']} "
                        f"response_tokens={record['response_tokens']} "
                        f"length_capped={str(record['length_capped']).lower()} ---\n"
                        f"{record['text']}\n\n"
                    )
            logger.info("  Saved %d samples → %s", len(sampled_records), sample_path)

        # TensorBoard x-axis counts completed trajectories, matching checkpoints.
        if tb_writer is not None:
            global_step = n_seen
            log_ppo_batch(tb_writer, step=global_step, stats=stats,
                          reward_infos=reward_infos, update_index=len(history))
            log_runtime(tb_writer, step=global_step, update_index=len(history),
                        batch_started=batch_started, response_lengths=response_token_counts,
                        trainer=trainer)
            # --- Core training signals ---
            tb_writer.add_scalar("custom/mean_reward", history[-1]["mean_reward"], global_step)
            tb_writer.add_scalar("custom/valid_rate", history[-1]["valid_rate"], global_step)
            tb_writer.add_scalar("custom/n_valid", n_valid, global_step)
            if sft_loss_val > 0:
                tb_writer.add_scalar("custom/sft_anchor_loss", sft_loss_val, global_step)
                tb_writer.add_scalar("custom/sft_replay_loss", sft_loss_val, global_step)
            tb_writer.add_scalar(
                "custom/sft_replay_actual_ratio",
                replay_items_seen / max(1, n_seen),
                global_step,
            )
            tb_writer.add_scalar("custom/advantage_var", adv_var, global_step)
            for _name in (
                "advantage_raw_mean", "advantage_raw_std", "advantage_raw_min",
                "advantage_raw_max", "advantage_raw_p50", "advantage_raw_p90",
                "advantage_raw_p95", "advantage_raw_p99",
                "advantage_whitened_var", "value_mean", "value_std",
                "return_mean", "return_std", "explained_variance",
            ):
                _value = history[-1].get(_name)
                if _value is not None:
                    tb_writer.add_scalar(f"custom/{_name}", _value, global_step)
            for _name in (
                "loss_total", "loss_policy", "loss_value", "policy_approxkl",
                "proofkg_process_mean", "proofkg_outcome_em_mean",
                "proofkg_outcome_f1_mean", "within_prompt_mixed_outcome_frac",
            ):
                _value = history[-1].get(_name)
                if _value is not None:
                    tb_writer.add_scalar(f"custom/{_name}", _value, global_step)
            for _dataset, _components in history[-1][
                "mixed_reward_by_dataset"
            ].items():
                for _name in (
                    "outcome_mean", "text_mean", "process_mean", "total_mean",
                    "valid_rate", "proofkg_eligible_rate",
                    "process_applied_count", "text_raw_step_mean",
                    "text_baseline_preupdate_step_mean",
                    "text_centered_unclipped_step_mean", "text_centered_step_mean",
                    "text_centered_abs_mean", "text_clip_frac",
                    "text_ema_baseline", "text_ema_n_obs",
                    "em_matched_nonprimary_count",
                    "f1_matched_nonprimary_count",
                ):
                    _value = _components[_name]
                    if _value is not None:
                        tb_writer.add_scalar(
                            f"mixed_reward/{_dataset}/{_name}",
                            _value,
                            global_step,
                        )
            for _name in (
                "mixed_text_raw_step_mean", "mixed_text_baseline_preupdate_step_mean",
                "mixed_text_centered_unclipped_step_mean", "mixed_text_centered_step_mean",
                "mixed_text_centered_abs_mean", "mixed_text_clip_frac",
                "mixed_text_ema_baseline", "mixed_text_ema_n_obs",
            ):
                _value = history[-1].get(_name)
                if _value is not None:
                    tb_writer.add_scalar(f"custom/{_name}", _value, global_step)
            # --- KL divergence (TRL's objective/kl) ---
            kl_val = history[-1]["ppo_mean_kl"]
            if kl_val is not None:
                tb_writer.add_scalar("custom/kl_divergence", kl_val, global_step)
            for _name in (
                "adaptive_kl_coef", "objective_kl_per_response_token_estimate",
                "response_tokens_mean", "length_capped_frac",
                "hard_curriculum_recovery_frac",
            ):
                _value = history[-1].get(_name)
                if _value is not None:
                    tb_writer.add_scalar(f"custom/{_name}", _value, global_step)
            # --- Clip fraction ---
            clip_val = history[-1]["policy_clipfrac"]
            if clip_val is not None:
                tb_writer.add_scalar("custom/clip_fraction", clip_val, global_step)
            # --- Policy entropy (diversity signal) ---
            ent_val = history[-1]["policy_entropy"]
            if ent_val is not None:
                tb_writer.add_scalar("custom/policy_entropy", ent_val, global_step)
            # --- Reward distribution ---
            if traj_rewards:
                import numpy as _np
                tb_writer.add_scalar("custom/reward_std", float(_np.std(traj_rewards)), global_step)
                tb_writer.add_scalar("custom/reward_max", float(max(traj_rewards)), global_step)
                tb_writer.add_scalar("custom/reward_min", float(min(traj_rewards)), global_step)
            # --- R8: Reasoning content quality ---
            if sample_log is not None:
                tb_writer.add_scalar("r8/step_rate", sample_log["step_rate"], global_step)
                tb_writer.add_scalar("r8/final_answer_rate", sample_log["final_answer_rate"], global_step)
                tb_writer.add_scalar("r8/reasoning_content_rate", sample_log["reasoning_content_rate"], global_step)
                tb_writer.add_scalar("r8/total_steps", sample_log["total_steps"], global_step)
                tb_writer.add_scalar("r8/steps_with_content", sample_log["steps_with_content"], global_step)
            # ── R9: Reward component diagnostics ──
            for _name, _value in source_gate_batch.items():
                if isinstance(_value, (int, float)):
                    tb_writer.add_scalar(f"source_gate/{_name}", _value, global_step)
            tb_writer.add_scalar("reward/alpha_mean", reward_rc["alpha_mean"], global_step)
            tb_writer.add_scalar("reward/r_kg_mean", reward_rc["r_kg_mean"], global_step)
            tb_writer.add_scalar("reward/r_text_mean", reward_rc["r_text_mean"], global_step)
            tb_writer.add_scalar("reward/r_total_mean", reward_rc["r_total_mean"], global_step)
            tb_writer.add_scalar("reward/n_steps", reward_rc["n_steps"], global_step)
            tb_writer.add_scalar("reward/kg_reward_share", reward_rc["kg_reward_share"], global_step)
            # §9.4-1 (量纲): the three new diagnostics.
            tb_writer.add_scalar("reward/r_text_used_mean", reward_rc["r_text_used_mean"], global_step)
            tb_writer.add_scalar("reward/text_baseline", reward_rc["text_baseline"], global_step)
            tb_writer.add_scalar(
                "reward/text_baseline_n_obs",
                reward_rc["text_baseline_n_obs"],
                global_step,
            )
            tb_writer.add_scalar("reward/dr_dalpha", reward_rc["dr_dalpha"], global_step)
            tb_writer.add_scalar("reward/r_kg_zero_frac", reward_rc["r_kg_zero_frac"], global_step)
            for _name in (
                "cite_any_step_frac", "cite_match_mean_citing_step",
                "cite_match_mean_reward_visible_citing_step",
                "cite_unknown_only_step_frac_citing",
                "cite_partial_match_step_frac_citing",
                "cite_all_matched_step_frac_citing",
                "alpha_mean_no_cite_step", "alpha_mean_known_cite_step",
                "alpha_mean_unknown_cite_step",
                "r_kg_zero_frac_no_cite_step",
                "r_kg_zero_frac_known_cite_step",
                "r_kg_zero_frac_unknown_cite_step",
                "citation_contract_error_step_frac",
                "citation_contract_invalid_response_frac",
                "citation_raw_citing_step_frac",
                "citation_known_citing_step_frac",
                "citation_unknown_citing_step_frac",
                "citation_malformed_content_step_frac",
                "citation_known_surface_count",
                "citation_unknown_surface_count",
                "citation_known_frac_recognized_surfaces",
            ):
                _value = reward_rc.get(_name)
                if _value is not None:
                    tb_writer.add_scalar(f"reward/{_name}", _value, global_step)

            tb_writer.flush()

        # Intermediate checkpoint: save the (PEFT) adapter whenever n_seen crosses
        # a save_every_steps boundary, so a run that collapses can be rolled back
        # to the last healthy step. The crossing test (rather than `% == 0`) is
        # robust to a batch_size that does not divide save_every_steps. The final
        # save below writes a separate `final/` dir, so there is no double-save.
        if (
            cfg.save_every_steps > 0
            and n_seen < cfg.total_steps
            and (n_seen // cfg.save_every_steps) != ((n_seen - cfg.batch_size) // cfg.save_every_steps)
        ):
            ckpt_dir = out_dir / f"step_{n_seen}"
            trainer.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(ckpt_dir)
            # Persist history incrementally so a killed run keeps its curves and
            # the saved step's metrics are recoverable alongside the weights.
            with open(out_dir / "history.jsonl", "w", encoding="utf-8") as fh:
                for h in history:
                    fh.write(json.dumps(h) + "\n")
            logger.info("Saved intermediate PPO checkpoint at %s (step %d)", ckpt_dir, n_seen)

        # Paid-smoke cost guard. Evaluate after telemetry and any scheduled
        # checkpoint/history save above, so a failed experiment stays auditable.
        # The last update is part of the experiment too.  Historical code
        # skipped the rolling thresholds when ``n_seen == total_steps``, which
        # could let a collapsed final window be saved as COMPLETE.  Always run
        # the pre-registered guard, including on the terminal update.
        health_failure = immediate_health_failure or _smoke_health_guard_reason(
            history, cfg
        )
        if health_failure is not None:
            history_path = out_dir / "history.jsonl"
            with open(history_path, "w", encoding="utf-8") as fh:
                for h in history:
                    fh.write(json.dumps(h) + "\n")
            existing_step = out_dir / f"step_{n_seen}"
            if existing_step.is_dir():
                failure_checkpoint = existing_step
            else:
                failure_checkpoint = out_dir / f"aborted_step_{n_seen}"
                trainer.save_pretrained(str(failure_checkpoint))
                tokenizer.save_pretrained(failure_checkpoint)
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()
            dump_manifest(
                out_dir,
                status="FAILED",
                extra={
                    "phase": "phase3_ppo",
                    "experiment_id": experiment_id,
                    "failure_type": (
                        "nonfinite_training_state"
                        if immediate_health_failure else "pre_registered_smoke_health_guard"
                    ),
                    "failure_reason": health_failure,
                    "failed_at_step": n_seen,
                    "config": asdict(cfg),
                    "history": artifact_identity(history_path),
                    "failure_checkpoint": artifact_identity(failure_checkpoint),
                },
            )
            raise RuntimeError(
                "Pre-registered PPO smoke health guard stopped the run at "
                f"step {n_seen}: {health_failure}. Artifacts were retained in "
                f"{out_dir}."
            )

    final_dir = out_dir / "final"
    trainer.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    history_path = out_dir / "history.jsonl"
    with open(history_path, "w", encoding="utf-8") as fh:
        for h in history:
            fh.write(json.dumps(h) + "\n")

    dump_manifest(
        out_dir,
        extra={
            "phase": "phase3_ppo",
            "experiment_id": experiment_id,
            "reference_mode": "explicit_frozen_sft_snapshot",
            "initial_reference_kl": (
                history[0].get("ppo_mean_kl") if history else None
            ),
            "config": asdict(cfg),
            "input_artifacts": {
                "silver": artifact_identity(silver_path),
                "sft_checkpoint": artifact_identity(cfg.sft_checkpoint),
                "sft_selection_report": (
                    artifact_identity(cfg.sft_selection_report_path)
                    if cfg.sft_selection_report_path else None
                ),
                "alpha_gate": (
                    artifact_identity(cfg.alpha_gate_path)
                    if cfg.alpha_gate_path and not cfg.mixed_outcome_reward else None
                ),
                "source_quality_gate": (
                    artifact_identity(cfg.source_gate_calibration_path)
                    if cfg.source_gated_reward_version == "v1" else None
                ),
                "question_kg_index": (
                    artifact_identity(qkg_path) if qkg_path is not None else None
                ),
                "question_kg_records": (
                    artifact_identity(question_kg_records_path)
                    if question_kg_records_path is not None else None
                ),
                "rearag": (
                    artifact_identity(model_path("rearag"))
                    if (not cfg.mixed_outcome_reward or cfg.mixed_text_reward)
                    else None
                ),
                "fixed_rollout_schedule": (
                    artifact_identity(cfg.fixed_rollout_schedule_path)
                    if cfg.fixed_rollout_schedule_path else None
                ),
            },
            "question_kg_override": question_kg_record_stats,
            "output_artifacts": {
                "final_checkpoint": artifact_identity(final_dir),
                "history": artifact_identity(history_path),
            },
            "history_tail": history[-5:],
        },
    )
    if tb_writer is not None:
        tb_writer.close()
        logger.info("TensorBoard writer closed.")

    logger.info("Phase 3b PPO done. Final checkpoint at %s", final_dir)
    return {"output_dir": str(out_dir), "final_checkpoint": str(final_dir)}
