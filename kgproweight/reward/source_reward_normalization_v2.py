"""Train-only step-unit normalization with explicit hierarchical weighting.

This opt-in successor leaves the historical normalizer untouched. Its fixed
softsign map changes the reward contract; zero hard clipping is a mathematical
property of that map, not evidence of improved answers or reward utility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from typing import Any


VERSION = "source-reward-text-normalization-v2"
INPUT_CONTRACT = "finite_raw_rearag_step_scores_in_closed_minus1_plus1_v1"
FIT_WEIGHTING = "equal_question_then_equal_valid_candidate_then_equal_step_v1"
APPLICATION_CONTRACT = "step_center_scale_softsign_then_mean_v2"
SCALE_FLOOR = 0.1
SOFT_SATURATION_THRESHOLD = 0.95


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _scores(raw: Sequence[float], *, allow_empty: bool = False) -> list[float]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("raw ReaRAG steps must be a sequence")
    values = [_number(value, "raw ReaRAG score") for value in raw]
    if not allow_empty and not values:
        raise ValueError("valid text normalization requires at least one step")
    if any(not -1 <= value <= 1 for value in values):
        raise ValueError("raw ReaRAG score outside [-1,1]")
    return values


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def validate_text_normalization_v2(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed contract and numerical invariants, returning a copy."""
    if not isinstance(stats, Mapping):
        raise ValueError("text normalization statistics must be a mapping")
    result = deepcopy(dict(stats))
    required = {"version": VERSION, "input_contract": INPUT_CONTRACT,
                "fit_unit": "step", "fit_weighting": FIT_WEIGHTING,
                "application_contract": APPLICATION_CONTRACT, "fit_split": "train"}
    if any(result.get(name) != expected for name, expected in required.items()):
        raise ValueError("unknown text normalization version/unit/weighting/application contract")
    if (result.get("hard_clipping_used") is not False
            or result.get("graph_eligibility_used_for_text_population") is not False
            or result.get("soft_saturation_threshold") != SOFT_SATURATION_THRESHOLD
            or not math.isclose(_number(result.get("weight_sum"), "weight_sum"), 1.0,
                                rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("text normalization mapping/weighting telemetry contract mismatch")
    center = _number(result.get("text_center"), "text_center")
    scale = _number(result.get("text_scale"), "text_scale")
    raw_std = _number(result.get("raw_step_std"), "raw_step_std")
    floor = _number(result.get("scale_floor"), "scale_floor")
    if not -1 <= center <= 1 or not 0 <= raw_std <= 1 or floor != SCALE_FLOOR:
        raise ValueError("text normalization center/std/floor outside fixed bounds")
    if not math.isclose(scale, max(raw_std, SCALE_FLOOR), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("text_scale must equal max(raw_step_std, fixed floor)")
    if raw_std * raw_std > 1 - center * center + 1e-12:
        raise ValueError("step variance exceeds bounded-score support")
    counts = result.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("text normalizer requires train observation counts")
    values = {name: _count(counts.get(name), name) for name in (
        "input_candidates", "valid_candidates", "input_questions", "questions_with_valid_scores", "steps")}
    if not (values["input_candidates"] >= values["valid_candidates"] >= values["questions_with_valid_scores"] >= 1
            and values["input_questions"] >= values["questions_with_valid_scores"]
            and values["input_candidates"] >= values["input_questions"]
            and values["steps"] >= values["valid_candidates"]):
        raise ValueError("inconsistent train text observation counts")
    components = result.get("variance_components")
    if not isinstance(components, Mapping):
        raise ValueError("step normalization requires hierarchical variance components")
    within = _number(components.get("within_trajectory"), "within-trajectory variance")
    between = _number(components.get("between_trajectory_means"), "between-trajectory variance")
    if min(within, between) < 0 or not math.isclose(within + between, raw_std * raw_std,
                                                   rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("step variance decomposition mismatch")
    return result


def fit_text_normalization_v2(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit only the caller-supplied train rows; never construct a new split.

    The caller must select and bind membership using the frozen family split.
    Explicit split fields, when present, must say train. Every observed question
    gets equal mass, every valid candidate of that question gets equal mass, and
    each step of a candidate shares its mass. Invalid candidates remain counted
    but contribute no text observations.
    """
    by_question: dict[tuple[str, str], list[tuple[str, list[float]]]] = defaultdict(list)
    seen, all_questions, digest_rows = set(), set(), []
    for row in train_rows:
        if not isinstance(row, Mapping):
            raise ValueError("train rows must be mappings")
        for field in ("split", "family_split"):
            if field in row and row[field] != "train":
                raise ValueError("non-train row passed to train-only text fit")
        candidate = _identity(row.get("candidate_id"), "candidate_id")
        if candidate in seen:
            raise ValueError("duplicate candidate_id in text normalization fit")
        seen.add(candidate)
        dataset, qid = _identity(row.get("dataset"), "dataset"), _identity(row.get("qid"), "qid")
        key = (dataset, qid)
        all_questions.add(key)
        valid = row.get("trajectory_valid")
        if not isinstance(valid, bool):
            raise ValueError("trajectory_valid must be an explicit boolean")
        values = _scores(row.get("raw_text"), allow_empty=not valid)
        if not valid and values:
            raise ValueError("invalid trajectories cannot supply text observations")
        digest_rows.append({"candidate_id": candidate, "dataset": dataset, "qid": qid,
                            "trajectory_valid": valid, "raw_text": values})
        if valid:
            by_question[key].append((candidate, values))
    if not by_question:
        raise ValueError("no valid train text observations")
    # Sort for deterministic accumulation independent of JSONL row ordering.
    question_mass = 1.0 / len(by_question)
    observations = []
    for key, candidates in sorted(by_question.items()):
        candidate_mass = question_mass / len(candidates)
        for _candidate, values in sorted(candidates):
            observations.append((key, candidate_mass, values, math.fsum(values) / len(values)))
    center = math.fsum(mass * mean for _key, mass, _values, mean in observations)
    within = math.fsum(mass * math.fsum((value - mean) ** 2 for value in values) / len(values)
                       for _key, mass, values, mean in observations)
    between = math.fsum(mass * (mean - center) ** 2 for _key, mass, _values, mean in observations)
    raw_std = math.sqrt(within + between)
    by_dataset = defaultdict(Counter)
    for (dataset, _qid), candidates in by_question.items():
        by_dataset[dataset]["questions_with_valid_scores"] += 1
        by_dataset[dataset]["valid_candidates"] += len(candidates)
        by_dataset[dataset]["steps"] += sum(len(values) for _candidate, values in candidates)
    digest = hashlib.sha256(json.dumps(sorted(digest_rows, key=lambda row: row["candidate_id"]),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    result = {
        "version": VERSION, "input_contract": INPUT_CONTRACT,
        "fit_unit": "step", "fit_weighting": FIT_WEIGHTING,
        "application_contract": APPLICATION_CONTRACT, "fit_split": "train",
        "membership_contract": "caller_selected_frozen_train_family_membership_no_resplit",
        "text_center": center, "text_scale": max(raw_std, SCALE_FLOOR),
        "raw_step_std": raw_std, "scale_floor": SCALE_FLOOR,
        "variance_components": {"within_trajectory": within, "between_trajectory_means": between},
        "counts": {"input_candidates": len(seen), "valid_candidates": len(observations),
                   "input_questions": len(all_questions), "questions_with_valid_scores": len(by_question),
                   "steps": sum(len(values) for _key, _mass, values, _mean in observations)},
        "by_dataset": {key: dict(value) for key, value in sorted(by_dataset.items())},
        "fit_input_sha256": digest,
        "hard_clipping_used": False, "soft_saturation_threshold": SOFT_SATURATION_THRESHOLD,
        "weight_sum": math.fsum(mass for _key, mass, _values, _mean in observations),
        "graph_eligibility_used_for_text_population": False,
        "scientific_boundary": "Fixed statistical contract repair; source mix remains that of train questions. Zero hard clipping is a map property, not EM/F1 or process-utility evidence.",
    }
    return validate_text_normalization_v2(result)


def normalize_text_steps_v2(raw: Sequence[float], stats: Mapping[str, Any]) -> dict[str, Any]:
    """Apply fixed per-step standardization and softsign before averaging."""
    stats = validate_text_normalization_v2(stats)
    values = _scores(raw)
    z = [(value - stats["text_center"]) / stats["text_scale"] for value in values]
    bounded = [value / (1.0 + abs(value)) for value in z]
    return {
        "version": VERSION, "application_contract": APPLICATION_CONTRACT,
        "raw_step_scores": values, "normalized_unclipped_step_scores": z,
        "bounded_step_scores": bounded, "mean_bounded": math.fsum(bounded) / len(bounded),
        "hard_clip_frac": 0.0,
        "soft_saturation_frac": sum(abs(value) >= SOFT_SATURATION_THRESHOLD for value in bounded) / len(bounded),
        "soft_saturation_threshold": SOFT_SATURATION_THRESHOLD,
        "raw_z_outside_unit_frac": sum(abs(value) > 1.0 for value in z) / len(z),
        "step_count": len(values), "raw_step_std": stats["raw_step_std"],
        "fit_counts": deepcopy(stats["counts"]),
    }
