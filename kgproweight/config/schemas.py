"""Typed configuration schemas.

We use pydantic v2. Every field has a default appropriate for Pro 6000
Blackwell (96 GB, bf16). All paths default to ``None`` and are filled in
by ``loader.expand_paths`` using ``kgproweight.utils.paths``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class AlphaGateConfig(_Base):
    initial_W: list[float] = Field(default_factory=lambda: [1.0, 1.5, -0.8])
    initial_b: float = -2.0
    temperature: float = 0.5
    calibration_weight: float = 0.1
    feature_dim: int = 3


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


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

class PPOConfig(_Base):
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
    outcome_weight: float = 8.0  # R5: stronger EM signal for hard examples
    text_reward_scale: float = 0.3  # R5: scale down R_text so EM+R_KG dominate
    # R7: format bonus REMOVED. Format is a constraint (ValidTrajectory gate),
    # not a reward target. See docs/problem_and_solutions.md and docs/R7_experiment_log.md.
    min_valid_steps: int = 3  # min parsed [Step N] blocks for outcome eligibility
    sft_anchor_weight: float = 0.02  # λ: lightweight format-preservation anchor
    sft_anchor_interval: int = 50  # run SFT anchor every N PPO steps
    pure_em_reward: bool = False  # skip R_KG+R_text — reward = EM (conditional on ValidTrajectory)
    vf_coef: float = 0.5
    max_input_length: int = 4096
    log_with: Optional[str] = None  # "tensorboard", "wandb", or None


class TrainingConfig(_Base):
    seed: int = 42
    phase: Literal["phase1", "phase2", "phase3_sft", "phase3_ppo", "phase3_grpo"] = "phase3_ppo"

    # ---- common
    base_model: str = "llama3-8B-instruct"
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
    max_input_length: int = 4096

    # ---- phase 3a (SFT)
    sft_epochs: int = 1
    sft_lr: float = 2.0e-5
    sft_batch_size: int = 8
    sft_grad_accum: int = 4
    sft_max_length: int = 4096

    # ---- phase 3b (PPO / GRPO)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    reference_model: Optional[str] = None  # path to SFT checkpoint
    text_reward_model: Optional[str] = None
    prm_checkpoint: Optional[str] = None

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
