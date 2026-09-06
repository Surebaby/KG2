"""Typed configuration schemas.

We use pydantic v2. Every field has a default appropriate for Pro 6000
Blackwell (96 GB, bf16). All paths default to ``None`` and are filled in
by ``loader.expand_paths`` using ``kgproweight.utils.paths``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from kgproweight.data.silver_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
)


class _Base(BaseModel):
    """Base model: allow extras for forward-compat with YAML overrides."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalConfig(_Base):
    name: str = "hybrid_rrf_top50"
    use_multi_retriever: bool = True
    merge_method: Literal["rrf", "concat", "rerank"] = "rrf"
    rrf_k: int = 60
    retrieval_topk: int = 50
    dense_model: str = "e5"  # logical name resolved by kgproweight.utils.paths.model_path
    dense_index: Optional[str] = None  # filled by loader
    sparse_method: Literal["bm25", "bm25s", "none"] = "bm25s"
    bm25_backend: Literal["bm25s", "pyserini", "lucene", "none"] = "bm25s"
    sparse_index: Optional[str] = None  # filled by loader
    corpus_path: Optional[str] = None
    use_sentence_transformer: bool = True
    instruction: Optional[str] = None
    retrieval_batch_size: int = 256
    pooling_method: Optional[str] = "mean"
    # ── Two-stage retrieval (R9 v6) ──
    # These were present in configs/retrieval/hybrid_rrf_top50.yaml but MISSING
    # from this schema, so any load through `validate=ProjectConfig` dropped
    # them and the run reverted to single-stage top-k.
    dense_candidate_topk: int = 100
    sparse_candidate_topk: int = 100
    rrf_candidate_topk: int = 50
    rerank_method: Literal["cross-encoder", "bm25", "none"] = "cross-encoder"
    cross_encoder_model: str = "models/bge-reranker-v2-m3"
    rerank_topk: int = 10
    prompt_passage_token_budget: int = 3860


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class AlphaGateConfig(_Base):
    initial_W: list[float] = Field(default_factory=lambda: [1.0, 1.5, -0.8, 0.9, 1.0])
    initial_b: float = -2.0
    temperature: float = 0.5
    calibration_weight: float = 0.1
    feature_dim: int = 5


class RewardConfig(_Base):
    text_reward_backend: Literal["rearag", "llama_head", "auto"] = "auto"
    text_reward_model: Optional[str] = "rearag"  # logical model name
    text_reward_fallback_path: Optional[str] = None  # path to fine-tuned reward head
    gamma_discount: float = 0.95
    outcome_em_weight: float = 1.0
    alpha_gate: AlphaGateConfig = Field(default_factory=AlphaGateConfig)
    use_real_logprobs: bool = True
    kg_embedding_model: Optional[str] = None  # path to a PyKEEN TransE/RotatE checkpoint


# ---------------------------------------------------------------------------
# Silver / Phase 1
# ---------------------------------------------------------------------------

class SilverDataConfig(_Base):
    teacher_model: str = "deepseek-chat"
    teacher_backend: Literal["openai", "deepseek"] = "deepseek"
    teacher_base_url: Optional[str] = None
    teacher_temperature: float = 0.3
    max_queries: int = 25000
    max_workers: int = 8
    output_path: Optional[str] = None  # data/silver_data/silver_trajectories.jsonl
    min_steps: int = 3
    max_steps: int = 7
    min_triple_rate: float = 0.4
    min_coverage: float = 0.5
    min_token_f1: float = 0.5
    use_retrieval: bool = True
    retrieval_top_k: int = 50
    # Experimental Phase-1 retrieval. Default OFF: formal control behavior is
    # unchanged unless a run explicitly opts into additive_v3.
    bridge_mode: Literal["off", "additive_v3"] = "off"
    bridge_first_round_topk: int = 5
    bridge_max_queries: int = 2
    bridge_only_k: int = 50
    # KG budget the Teacher sees. MUST equal the student's budget on every other
    # path or teacher and student are trained/served on different KG views
    # (retraining_plan §12.3): Phase3PPOConfig.ppo_max_kg_triples=12,
    # Phase3GRPOConfig.max_kg_triples=12, phase2_prm.py:150/214 max_keep=12,
    # KGProWeightPipeline.max_kg_triples=12 -- all with min_keep=5.
    # These are DECLARED FIELDS on purpose: _Base sets extra="allow", so an
    # undeclared key here would be swallowed into model_extra and read by
    # nobody, which is exactly how ppo_max_kg_triples stayed at its default
    # while the YAML appeared to set it.
    max_kg_triples: int = 12
    min_kg_keep: int = 5


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

class PPOConfig(_Base):
    # Explicit successor for corrected rollout modes, EOS retention and strict
    # reward alignment. Historical YAMLs retain their original runtime.
    runtime_contract_version: Literal["legacy", "v2"] = "legacy"
    learning_rate: float = 1.0e-5
    batch_size: int = 64    # overridden in YAML to 8 for VRAM
    mini_batch_size: int = 2  # R4: 2 (was 8 schema default, overridden to 1 then 2)
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    kl_coef: float = 0.1  # init_kl_coef: medium regime — space to change answers but keep format
    gamma: float = 0.95
    lam: float = 0.95
    max_grad_norm: float = 1.0
    total_ppo_steps: int = 5000  # R6: 5000 steps over full 9,839 silver set
    save_every_steps: int = 256  # 0 disables intermediate checkpointing
    early_stopping: bool = False
    target_kl: float = 8.0  # adaptive-controller target KL (TRL's `target`)
    kl_horizon: float = 2000.0  # adaptive KL controller horizon (TRL default 10000)
    outcome_weight: float = 8.0  # historical no-config default; formal YAML overrides
    text_reward_scale: float = 0.3  # R5: scale down R_text so EM+R_KG dominate
    step_reward_scale: float = 1.0  # historical no-config default; formal YAML overrides
    # Prompt-side KG budget. Declared (rather than accepted as model_extra) so
    # YAML values can be explicitly forwarded to Phase3PPOConfig.
    ppo_min_kg_triples: int = 5
    ppo_max_kg_triples: int = 12
    # Sparse legacy neighbourhoods cannot refute absent facts, so historical
    # runs required >=3 triples.  Exact Proof-KG citations can be verified from
    # a one-edge proof; Proof-KG experiments opt into 1 explicitly.
    prm_min_subgraph_for_verify: int = Field(default=3, ge=1)
    # Rollout controls are declared fields, not permissive ``model_extra``.
    # Every one is explicitly forwarded by scripts/train/phase3_ppo.py; this
    # prevents a YAML such as ``max_new_tokens: 384`` from looking accepted
    # while Phase3PPOConfig silently keeps its default of 256.
    max_new_tokens: int = 256
    # PPO recomputes log-probabilities from raw logits.  Sampling from a
    # temperature/top-p modified distribution would invalidate the ratio, so
    # these are intentionally fixed no-op values rather than tuning knobs.
    temperature: Literal[1.0] = 1.0
    top_p: Literal[1.0] = 1.0
    rollout_chunk_size: int = 8
    max_steps: int = 5
    ppo_max_passages: int = 15
    # retraining_plan §9.4-1 (量纲/D2): subtract R_Text's running DC offset before
    # mixing. Measured r_text mean 0.6284 vs r_kg 0.0896 -- a 7x gap on the same
    # nominal [-1,1] scale, which made d r_total/d alpha = -0.148 (the reward paid
    # the policy to LOWER alpha). False reproduces every pre-2026-08-23 run.
    center_text_reward: bool = False
    text_baseline_momentum: float = 0.99  # EMA momentum; ~100-sample window
    # R7: format bonus REMOVED. Format is a constraint (ValidTrajectory gate),
    # not a reward target. See docs/problem_and_solutions.md and docs/R7_experiment_log.md.
    min_valid_steps: int = 3  # min parsed [Step N] blocks for outcome eligibility
    shortfall_coef: float = 0.0
    target_steps: int = 3
    min_reasoning_chars: int = 20  # R8: minimum chars in Reasoning field for content gate
    sft_anchor_weight: float = 0.02  # λ: lightweight format-preservation anchor
    # Legacy periodic anchor is disabled in formal runs. Supervised replay uses
    # a sample-ratio scheduler, which remains exact when batch_size * ratio < 1.
    sft_anchor_interval: int = 0
    sft_replay_ratio: float = 0.10
    pure_em_reward: bool = False  # skip R_KG+R_text — reward = EM (conditional on ValidTrajectory)
    # Experimental automatic-ProofKG reward.  Defaults are deliberately off so
    # historical configs remain reproducible.  Eligible rows must carry a
    # Gold-free query plan and a complete materialised proof path.
    proofkg_process_reward: bool = False
    # Paired automatic-ProofKG control arm: use the same eligibility, dynamic
    # validity and EM/F1 outcome term, but no process score.
    proofkg_outcome_only_reward: bool = False
    # Frozen scorer implementation used by the automatic-ProofKG fast path.
    # v1 reproduces every historical run; v2_1 is retained byte-for-byte for
    # completed experiments; v2_2 repairs temporal/multi-value derivation and
    # requires the same execution provenance.
    proofkg_process_version: Literal["v1", "v2_1", "v2_2", "v2_3"] = "v1"
    proofkg_process_weight: float = Field(default=1.0, ge=0.0)
    proofkg_f1_weight: float = Field(default=0.0, ge=0.0)
    proofkg_dynamic_validity: bool = False
    # Default-off mixed-dataset fast path: every row gets the same EM/F1 outcome
    # reward, while the optional ProofKG process term is restricted to exact-join
    # complete automatic proofs. Historical configs remain unchanged.
    mixed_outcome_reward: bool = False
    # Optional shared frozen-ReaRAG trajectory reward for paired PPO-T/PPO-TK.
    # Requires mixed_outcome_reward and backend=rearag; defaults off.
    mixed_text_reward: bool = False
    # Frozen source-quality successor; independent of the historical alpha gate.
    source_gated_reward_version: Literal["disabled", "v1"] = "disabled"
    source_gate_format_version: Literal["v1", "v2"] = "v1"
    # Independent objective ablation; legacy preserves all frozen experiments.
    answer_format_reward_version: Literal["legacy", "v2"] = "legacy"
    source_gate_credit_version: Literal["disabled", "v1", "v2"] = "disabled"
    source_gate_mode: Literal["text", "fixed", "learned"] = "learned"
    source_gate_calibration_path: Optional[str] = None
    # If true, abort unless every rollout row is eligible for the automatic
    # ProofKG branch.  This also permits skipping the unused 9B text reward
    # model, reducing VRAM and reward-stage latency.
    proofkg_require_all_eligible: bool = False
    # Draw K independent responses for the same prompt inside a PPO batch.
    # 1 exactly reproduces the historical independent-question sampler.
    rollouts_per_prompt: int = Field(default=1, ge=1)
    vf_coef: float = 0.5
    # TRL creates a fresh value head when Phase 3b starts from an SFT adapter.
    # Keep the historical random/dropout behaviour as the schema default for
    # reproducibility; stability experiments must opt into the neutral critic
    # explicitly and the CLI forwards both fields to the runtime dataclass.
    value_head_init: Literal["default", "zero"] = "default"
    value_head_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    # Optional smoke-only cost guard. 0 disables it for historical/formal runs.
    health_guard_after_steps: int = Field(default=0, ge=0)
    health_guard_window: int = Field(default=15, ge=3)
    health_guard_min_valid_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    health_guard_max_length_capped_frac: float = Field(default=1.0, ge=0.0, le=1.0)
    health_guard_max_mean_kl: float = Field(default=1.0e9, gt=0.0)
    max_input_length: int = 4096
    log_with: Optional[str] = None  # "tensorboard", "wandb", or None


class TrainingConfig(_Base):
    seed: int = 42
    phase: Literal["phase1", "phase2", "phase3_sft", "phase3_ppo", "phase3_grpo"] = "phase3_ppo"

    # ---- common
    base_model: str = "llama3-8B-instruct"
    silver_path: Optional[str] = None
    output_dir: Optional[str] = None
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    use_qlora: bool = False  # disabled by default on Pro 6000 96 GB
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ---- phase 2
    prm_epochs: int = 3
    prm_lr: float = 5.0e-5
    prm_batch_size: int = 8
    prm_grad_accum: int = 2
    prm_max_length: int = 2048
    # Alpha calibration target only. The 3-way auxiliary PRM target stays
    # unchanged and is not consumed by PPO.
    alpha_target: Literal["hard_verdict", "soft_abs_rkg"] = "hard_verdict"
    max_input_length: int = 4096

    # ---- phase 3a (SFT)
    sft_epochs: int = 1
    sft_lr: float = 2.0e-5
    sft_batch_size: int = 8
    sft_grad_accum: int = 4
    sft_max_length: int = 4096
    sft_save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    sft_save_steps: int = Field(default=500, gt=0)
    sft_save_total_limit: Optional[int] = Field(default=None, gt=0)
    sft_save_only_model: bool = False
    # Optional observability only; it does not alter the SFT objective.
    sft_log_with: Optional[Literal["tensorboard"]] = None
    sft_logging_dir: Optional[str] = None
    # Optional continued-SFT adapter.  Without this, a curriculum SFT run starts
    # again from the naked base model and discards the strong existing SFT
    # policy, which is not the intended experiment.
    sft_init_adapter_path: Optional[str] = None

    # ---- phase 3b (PPO / GRPO)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    reference_model: Optional[str] = None  # path to SFT checkpoint
    sft_checkpoint: Optional[str] = None
    # Frozen SFT checkpoint-selection report used to authorize this PPO start.
    # It is provenance/fail-fast metadata, not a training or reward variable.
    sft_selection_report_path: Optional[str] = None
    # Optional independent CE replay source.  This prevents an automatic-
    # ProofKG rollout cohort from silently replaying Gold-derived traces from
    # that same file.  None retains the historical same-as-silver behaviour.
    sft_replay_silver_path: Optional[str] = None
    sft_replay_split: Optional[Literal["train", "val", "test"]] = None
    alpha_gate_path: Optional[str] = None
    text_reward_model: Optional[str] = None
    prm_checkpoint: Optional[str] = None
    question_kg_index_path: Optional[str] = None
    max_kg_index_miss_rate: float = 1.0
    require_exact_kg_index_alignment: bool = False
    # Identity-safe per-question KG records (dataset::qid + question hash).
    # This is intentionally separate from the legacy question-text index.
    question_kg_records_path: Optional[str] = None
    min_question_kg_record_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    require_nonempty_question_kg_records: bool = False
    # Optional versioned PPO rollout inputs.  They are declared here (rather
    # than accepted through _Base.extra="allow") so the CLI can forward them
    # explicitly and a YAML cannot appear to work while being ignored.
    passage_overrides_path: Optional[str] = None
    rollout_schedule_path: Optional[str] = None
    # Optional identity-safe categorical weights for PPO prompt selection.
    # This changes only which train qid is drawn; K responses for a selected
    # prompt remain grouped by ``rollouts_per_prompt``.
    rollout_sampling_weights_path: Optional[str] = None
    # Cross-environment deterministic paired experiments may consume a frozen
    # rollout-by-rollout qid schedule directly. This avoids torch.multinomial
    # implementation drift across local/remote PyTorch versions.
    fixed_rollout_schedule_path: Optional[str] = None

    # ---- train/val/test split (shared by phase 2 and phase 3)
    # Applies to every phase that reads silver data, so the same fold definition
    # is reused rather than each phase inventing its own. ``split=None`` keeps
    # the pre-split behaviour of training on the whole file.
    split: Optional[Literal["train", "val", "test"]] = None
    split_allow_none: bool = False
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    # Deliberately independent of ``seed``: a sweep over training randomness must
    # not also redraw the held-out set, or the resulting variance mixes two
    # different sources.
    split_seed: int = DEFAULT_SPLIT_SEED

    # ---- runtime
    silver_data: SilverDataConfig = Field(default_factory=SilverDataConfig)
    alpha_override: Optional[float] = None  # for ablations
    binary_labels_only: bool = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class BaselineConfig(_Base):
    name: str
    pipeline_class: str
    generator_model_name: str  # logical name; resolved at runtime
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(_Base):
    datasets: list[str] = Field(default_factory=lambda: ["hotpotqa", "2wikimultihopqa", "musique"])
    split: str = "dev"
    test_sample_num: Optional[int] = None
    save_intermediate_data: bool = True
    save_metric_score: bool = True
    metrics: list[str] = Field(default_factory=lambda: ["em", "f1"])
    gpu_id: str = "0"
    seeds: list[int] = Field(default_factory=lambda: [13, 42, 2024])
    use_real_alpha: bool = True  # KGProWeight runs honour the trained gate
    # Inference KG supply mode. "legacy" (default) reads the question-KG index;
    # Non-legacy modes read versioned per-question records via
    # question_kg_records_path. QPEG is passage-only; ProofKG may use Wikidata.
    kg_supply_mode: Literal[
        "legacy", "qpeg_v1", "proofkg_v1",
        "legacy_plus_proofkg", "legacy_plus_complete_proofkg",
    ] = "legacy"
    question_kg_records_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level project config
# ---------------------------------------------------------------------------

class ProjectConfig(_Base):
    name: str = "kg_proweight"
    project_root: Optional[str] = None
    data_dir: Optional[str] = None
    index_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    output_dir: Optional[str] = None

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # ---- FlashRAG passthrough (anything not validated explicitly)
    flashrag: dict[str, Any] = Field(default_factory=dict)
