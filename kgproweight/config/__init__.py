"""Configuration loading and schema validation."""

from kgproweight.config.loader import load_config, merge_yaml, expand_paths
from kgproweight.config.schemas import (
    ProjectConfig,
    RetrievalConfig,
    RewardConfig,
    TrainingConfig,
    EvalConfig,
    AlphaGateConfig,
    PPOConfig,
    SilverDataConfig,
    BaselineConfig,
)

__all__ = [
    "load_config",
    "merge_yaml",
    "expand_paths",
    "ProjectConfig",
    "RetrievalConfig",
    "RewardConfig",
    "TrainingConfig",
    "EvalConfig",
    "AlphaGateConfig",
    "PPOConfig",
    "SilverDataConfig",
    "BaselineConfig",
]
