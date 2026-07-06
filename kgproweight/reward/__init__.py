"""Reward components: alpha-gate, PRM annotator, text reward, composite, IHR judge."""

from kgproweight.reward.alpha_gate import (
    AlphaGate,
    AlphaCalibrationLoss,
    compute_features,
    compute_graph_density,
    compute_link_confidence,
    compute_semantic_entropy,
    entropy_from_logprobs,
)
from kgproweight.reward.prm_value_head import PRMValueHead
from kgproweight.reward.prm_annotator import (
    POSITIVE,
    NEUTRAL,
    NEGATIVE,
    ParsedStep,
    PRMAnnotator,
    parsed_step_from_silver_dict,
)
from kgproweight.reward.text_reward_model import (
    TextRewardModel,
    LlamaTextRewardHead,
    build_text_reward_model,
)
from kgproweight.reward.composite_reward import CompositeRewardModel, StepReward
from kgproweight.reward.ihr_judge import IHRJudge, compute_cohen_kappa

__all__ = [
    "AlphaGate",
    "AlphaCalibrationLoss",
    "compute_features",
    "compute_graph_density",
    "compute_link_confidence",
    "compute_semantic_entropy",
    "entropy_from_logprobs",
    "PRMValueHead",
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE",
    "ParsedStep",
    "PRMAnnotator",
    "parsed_step_from_silver_dict",
    "TextRewardModel",
    "LlamaTextRewardHead",
    "build_text_reward_model",
    "CompositeRewardModel",
    "StepReward",
    "IHRJudge",
    "compute_cohen_kappa",
]
