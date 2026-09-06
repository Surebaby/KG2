"""TensorBoard telemetry for PPO; never changes rewards or optimizer state.

The event step is the number of trajectories consumed, not optimizer steps.
TRL arrays are logged explicitly as *raw*: upstream padding/masks are not
available here. Use the trainer's mask-aware advantage/value diagnostics for
training health, rather than treating raw array means as token-level estimates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

HISTOGRAM_INITIAL_BATCHES = 3
HISTOGRAM_EVERY_BATCHES = 10


def _numeric_array(value: Any) -> np.ndarray | None:
    """Detach tensors without retaining graphs; reject strings/object arrays."""
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            return None
        return array.astype(np.float64, copy=False).reshape(-1)
    except (TypeError, ValueError, RuntimeError, OverflowError):
        return None


def _scalar(value: Any) -> float | None:
    array = _numeric_array(value)
    if array is None or array.size != 1 or not np.isfinite(array[0]):
        return None
    return float(array[0])


def _tag_component(value: Any) -> str:
    # Dataset names are metadata, not a way to create extra tag hierarchy.
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _leaves(values: Mapping[str, Any], prefix: str = ""):
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _leaves(value, name)
        else:
            yield name, value


def _distribution(writer: Any, tag: str, values: Sequence[Any], step: int,
                  histograms: bool) -> None:
    array = _numeric_array(values)
    if array is None or not array.size:
        return
    finite = array[np.isfinite(array)]
    if finite.size != array.size:
        writer.add_scalar(f"telemetry/nonfinite/{tag}", array.size - finite.size, step)
    if not finite.size:
        return
    writer.add_scalar(f"{tag}_mean", float(finite.mean()), step)
    writer.add_scalar(f"{tag}_std", float(finite.std()), step)
    if histograms:
        writer.add_histogram(f"{tag}_distribution", finite, step)


def log_ppo_stats(writer: Any, stats: Mapping[str, Any], *, step: int,
                  histograms: bool = False) -> None:
    """Write all numeric TRL stats, explicitly counting nonfinite omissions.

    Numeric scalar values retain official TRL tags. Multi-value arrays retain
    their tag prefix but use raw_mean/raw_std/raw_histogram suffixes: no mask is
    guessed, and upstream -1 padding is not silently removed.
    """
    for tag, value in _leaves(stats):
        array = _numeric_array(value)
        if array is None or not array.size:
            continue
        finite = array[np.isfinite(array)]
        if finite.size != array.size:
            writer.add_scalar(f"telemetry/nonfinite/{tag}", array.size - finite.size, step)
        if not finite.size:
            continue
        if array.size == 1:
            writer.add_scalar(tag, float(finite[0]), step)
        else:
            writer.add_scalar(f"{tag}/raw_mean", float(finite.mean()), step)
            writer.add_scalar(f"{tag}/raw_std", float(finite.std()), step)
            if histograms:
                writer.add_histogram(f"{tag}/raw_histogram", finite, step)


def _reward_group(writer: Any, prefix: str, rows: Sequence[Mapping[str, Any]],
                  step: int, histograms: bool) -> None:
    writer.add_scalar(f"{prefix}/count", len(rows), step)
    writer.add_scalar(f"{prefix}/valid_rate", sum(
        bool(row.get("trajectory_valid")) for row in rows
    ) / len(rows), step)
    for name, container, field in (
        ("em", "proofkg_process", "outcome_em"),
        ("f1", "proofkg_process", "outcome_f1"),
        ("outcome", "mixed_reward", "outcome"),
        ("text_component", "mixed_reward", "text"),
        ("graph_component", "mixed_reward", "process"),
        ("total", "mixed_reward", "total"),
        ("answer_component", "answer_format_reward", "answer_component"),
        ("format_component", "answer_format_reward", "format_component"),
        ("canonical_em", "answer_format_reward", "canonical_em"),
        ("canonical_f1", "answer_format_reward", "canonical_f1"),
    ):
        values = [(row.get(container) or {}).get(field) for row in rows]
        values = [value for value in values if value is not None]
        _distribution(writer, f"{prefix}/{name}", values, step, histograms)

    objective_rows = [row["answer_format_reward"] for row in rows
                      if row.get("answer_format_reward")]
    if objective_rows:
        for name, case in (("shortfall_salvage_rate", "format_invalid_answer_retained"),
                           ("severe_invalid_rate", "invalid_answer_unavailable")):
            writer.add_scalar(f"{prefix}/{name}", sum(
                row["case"] == case for row in objective_rows
            ) / len(objective_rows), step)
        writer.add_scalar(f"{prefix}/answer_signal_applied_rate", sum(
            bool(row["answer_signal_applied"]) for row in objective_rows
        ) / len(objective_rows), step)

    valid = [row for row in rows if row.get("trajectory_valid")]
    text_raw = [value for row in valid for value in
                (row.get("mixed_reward") or {}).get("text_raw_step_scores", [])]
    text_norm = [value for row in valid for value in
                 (row.get("mixed_reward") or {}).get("text_centered_clipped_step_scores", [])]
    _distribution(writer, f"{prefix}/text_raw_step", text_raw, step, histograms)
    _distribution(writer, f"{prefix}/text_normalized_step", text_norm, step, histograms)
    # v2 softsign is bounded without hard clipping.  |z| > 1 remains useful
    # tail telemetry, but must not be labelled clipping in AutoDL's curves.
    text_count, hard_clipped = 0, 0.0
    soft_count, soft_saturated, soft_tail = 0, 0.0, 0.0
    for row in valid:
        gate = row.get("source_gate") or {}
        finite_text = [value for value in map(_scalar,
            gate.get("text_normalized_unclipped_steps", [])) if value is not None]
        count = len(finite_text)
        text_count += count
        soft = gate.get("text_normalization_v2")
        if soft:
            hard_clipped += count * float(soft["hard_clip_frac"])
            soft_count += count
            soft_saturated += count * float(soft["soft_saturation_frac"])
            soft_tail += count * float(soft["raw_z_outside_unit_frac"])
        else:
            hard_clipped += sum(abs(value) > 1 for value in finite_text)
    if text_count:
        writer.add_scalar(f"{prefix}/text_clip_frac", hard_clipped / text_count, step)
    if soft_count:
        writer.add_scalar(f"{prefix}/text_softsign_saturation_frac", soft_saturated / soft_count, step)
        writer.add_scalar(f"{prefix}/text_raw_z_outside_unit_frac", soft_tail / soft_count, step)

    graph_scored = [row["source_gate"] for row in valid
                    if (row.get("source_gate") or {}).get("m_graph") == 1
                    and not row["source_gate"].get("invalid_not_scored")]
    for field in ("graph_raw", "graph_normalized"):
        values = [row[field] for row in graph_scored if field in row]
        _distribution(writer, f"{prefix}/{field}", values, step, histograms)
    graph_unclipped = [_scalar(row.get("graph_normalized_unclipped")) for row in graph_scored]
    graph_unclipped = [value for value in graph_unclipped if value is not None]
    if graph_unclipped:
        writer.add_scalar(f"{prefix}/graph_clip_frac", sum(
            abs(value) > 1 for value in graph_unclipped
        ) / len(graph_unclipped), step)


def log_ppo_batch(writer: Any, *, step: int, stats: Mapping[str, Any],
                  reward_infos: Sequence[Mapping[str, Any]],
                  histogram_every: int = HISTOGRAM_EVERY_BATCHES, update_index: int | None = None) -> None:
    """Log official PPO stats and answer/process/gate telemetry for one batch.

    Histogram cadence uses the *batch update index*: the first three batches,
    then every tenth by default. This covers the H/W/M probe12 schedule while
    the event x-axis stays ``step`` (consumed trajectories). Missing strata
    emit no fabricated means.
    Alpha for all/eligible rows is applied credit (invalid rows are zero);
    eligible_valid isolates actual scored gate predictions.
    """
    if histogram_every < 0:
        raise ValueError("histogram_every must be nonnegative")
    if update_index is None:
        update_index = step
    histograms = histogram_every > 0 and (
        update_index <= HISTOGRAM_INITIAL_BATCHES or update_index % histogram_every == 0
    )
    log_ppo_stats(writer, stats, step=step, histograms=histograms)
    if not reward_infos:
        return
    groups = defaultdict(list)
    for row in reward_infos:
        groups["reward/all"].append(row)
        dataset = _tag_component((row.get("mixed_reward") or {}).get("dataset") or "unknown")
        groups[f"reward/dataset/{dataset}"].append(row)
        gate = row.get("source_gate") or {}
        mask = gate.get("m_graph")
        if mask in (0, 1):
            groups[f"reward/m_graph/{int(mask)}"].append(row)
            groups[f"reward/dataset/{dataset}/m_graph/{int(mask)}"].append(row)
    for prefix, rows in groups.items():
        _reward_group(writer, prefix, rows, step, histograms)

    records = [(row, row["source_gate"]) for row in reward_infos if row.get("source_gate")]
    if not records:
        return
    eligible = [(row, gate) for row, gate in records if gate.get("m_graph") == 1]
    eligible_valid = [(row, gate) for row, gate in eligible
                      if row.get("trajectory_valid") and not gate.get("invalid_not_scored")]
    writer.add_scalar("gate/all/eligible_count", len(eligible), step)
    writer.add_scalar("gate/all/eligible_valid_count", len(eligible_valid), step)
    writer.add_scalar("gate/all/eligible_rate", len(eligible) / len(records), step)
    for name, cohort in (("all", records), ("eligible", eligible),
                         ("eligible_valid", eligible_valid)):
        if not cohort:
            continue
        writer.add_scalar(f"gate/{name}/count", len(cohort), step)
        _distribution(writer, f"gate/{name}/alpha_effective", [
            gate["alpha_effective"] for _, gate in cohort if "alpha_effective" in gate
        ], step, histograms)
    if eligible_valid:
        _distribution(writer, "gate/eligible_valid/alpha_predicted", [
            gate["alpha_predicted"] for _, gate in eligible_valid if "alpha_predicted" in gate
        ], step, histograms)
        feature_names = sorted({feature for _, gate in eligible_valid
                                for feature in ((gate.get("features") or {}).get("values") or {})})
        for feature in feature_names:
            values = [((gate.get("features") or {}).get("values") or {}).get(feature)
                      for _, gate in eligible_valid]
            _distribution(writer, f"gate/eligible_valid/feature_{feature}", [
                value for value in values if value is not None
            ], step, histograms)


_SENSITIVE_KEYS = {"password", "passwd", "secret", "api_key", "access_key",
                   "private_key", "access_token", "auth_token", "hf_token"}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): ("[REDACTED]" if str(key).lower() in _SENSITIVE_KEYS
                           else _json_safe(item)) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def log_run_metadata(writer: Any, config: Any, metadata: Mapping[str, Any] | None = None,
                     *, step: int = 0) -> None:
    """Write caller-provided configuration/identity, without reading environment.

    Avoid add_hparams: it creates child runs and can confuse the AutoDL run list.
    Never pass credentials, answers or generated trajectories as metadata.
    """
    config = _json_safe(config)
    details = _json_safe(metadata or {})
    for tag, value in (("run/config", config), ("run/metadata", details)):
        writer.add_text(tag, "```json\n" + json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ) + "\n```", step)
    writer.add_text("run/metric_semantics", (
        "X-axis: consumed PPO trajectories; update_index controls histogram cadence. "
        "Histograms cover the first three PPO batches, then every tenth batch. "
        "Training rollout EM/F1 is not held-out evaluation. "
        "TRL arrays marked raw may include upstream padding; mask-aware trainer "
        "advantage/value/return diagnostics are the health metrics. "
        "Alpha all/eligible includes invalid zero credit; eligible_valid excludes "
        "unscored invalid trajectories. Missing cohorts produce no zero means. "
        "Graph/Text scores are process proxies, not factual correctness probabilities."
    ), step)
    if isinstance(config, Mapping):
        for tag, value in _leaves(config):
            scalar = _scalar(value)
            if scalar is not None:
                writer.add_scalar(f"config/{tag}", scalar, step)
