"""Data layer: prompts, parsers, silver datasets, dropout loader."""

from kgproweight.data.parsers import (
    DISCOURSE_RE,
    ENTITY_RE,
    TRIPLE_RE,
    ParsedStep,
    extract_final_answer,
    parse_steps,
    parse_teacher_output,
    parsed_step_from_silver_dict,
)
from kgproweight.data.prompts import (
    TEACHER_SYSTEM_PROMPT,
    TEACHER_USER_TEMPLATE,
    SFT_USER_TEMPLATE,
    RL_USER_TEMPLATE,
    INFERENCE_USER_TEMPLATE,
    KG_BLOCK_TEMPLATE,
    RETRIEVED_BLOCK_TEMPLATE,
    build_teacher_messages,
    build_sft_messages,
    build_rl_messages,
    build_inference_messages,
    format_kg_block,
    format_retrieved_block,
)
from kgproweight.data.silver_dataset import (
    SilverDatasetReader,
    SilverStepRecord,
    SilverTrajectory,
    iter_silver_trajectories,
)
from kgproweight.data.d_dropout_loader import (
    DropoutItem,
    DropoutDataset,
    load_dropout_dataset,
)
from kgproweight.data.flashrag_loader import get_dataset

__all__ = [
    "DISCOURSE_RE",
    "ENTITY_RE",
    "TRIPLE_RE",
    "ParsedStep",
    "extract_final_answer",
    "parse_steps",
    "parse_teacher_output",
    "parsed_step_from_silver_dict",
    "TEACHER_SYSTEM_PROMPT",
    "TEACHER_USER_TEMPLATE",
    "SFT_USER_TEMPLATE",
    "RL_USER_TEMPLATE",
    "INFERENCE_USER_TEMPLATE",
    "KG_BLOCK_TEMPLATE",
    "RETRIEVED_BLOCK_TEMPLATE",
    "build_teacher_messages",
    "build_sft_messages",
    "build_rl_messages",
    "build_inference_messages",
    "format_kg_block",
    "format_retrieved_block",
    "SilverDatasetReader",
    "SilverStepRecord",
    "SilverTrajectory",
    "iter_silver_trajectories",
    "DropoutItem",
    "DropoutDataset",
    "load_dropout_dataset",
    "get_dataset",
]
