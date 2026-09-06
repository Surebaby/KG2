"""Versioned, frozen trajectory credit gate for structural Graph/Text rewards.

The ratio training target is a *heuristic score ratio*, not an independently
identified probability of source reliability.  Features never use live policy
entropy, gold answers, dataset-specific routing, or a live entity linker.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.reward.proofkg_process_v2_3 import SCORER_VERSION
from kgproweight.reward.trajectory_source_gate import evaluate_graph_gate


ARTIFACT_SCHEMA = "source-quality-gate-artifact-v1"
GATE_VERSION = "source-quality-gate-v1"
FEATURE_VERSION = "source-quality-trajectory-features-v1"
FEATURE_NAMES = ("density", "link_confidence", "cite_any", "cite_match")
SPLIT_VERSION = "source-quality-family-split-v1"
TARGET_VERSION = "structural-graph-rearag-heuristic-ratio-v1"
QUALITY_ABSTAIN_THRESHOLD = 0.05
DENOMINATOR_EPSILON = 1e-8
SOURCE_SCALE_FLOOR = 0.10


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def assign_family_splits(families: Iterable[str], seed: int = 42) -> dict[str, str]:
    """Exactly 60/20/20 (floor/floor/remainder) of unique frozen families."""
    unique = set(families)
    if not unique or any(not isinstance(family, str) or len(family) != 64 for family in unique):
        raise ValueError("nonempty current-family SHA256 values required")
    ordered = sorted(unique, key=lambda family: hashlib.sha256(f"{SPLIT_VERSION}\0{seed}\0{family}".encode()).hexdigest())
    train_end, calibration_end = int(len(ordered) * .6), int(len(ordered) * .6) + int(len(ordered) * .2)
    return {family: "train" if i < train_end else "calibration" if i < calibration_end else "confirmation" for i, family in enumerate(ordered)}


def compute_gate_features(spec: Any, steps: Sequence[Any], proof_result: Mapping[str, Any]) -> dict[str, Any]:
    """Compute four trajectory features from one complete frozen evidence view.

    ``spec.metadata['source_quality_record']`` must contain the complete original
    question-KG record.  Missing identity fields are never synthesized from the
    request to make a compact legacy runtime record pass the hard gate.
    """
    metadata = getattr(spec, "metadata", {}) or {}
    record = metadata.get("source_quality_record") or {}
    if not isinstance(record, Mapping):
        record = {}
    triples = [tuple(str(v).strip() for v in triple) for triple in getattr(spec, "kg_subgraph", []) if len(triple) == 3]
    cutoff = str((record.get("provenance") or {}).get("historical_cutoff") or "")
    try:
        decision = evaluate_graph_gate(
            record, dataset=str(metadata.get("dataset") or ""), qid=str(metadata.get("qid") or ""),
            question=str(getattr(spec, "query", "")), historical_cutoff=cutoff,
        ).to_dict()
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        decision = {"m_graph": 0, "graph_eligible": False, "routing_reason": "malformed_source_record", "checks": {"record_evaluable": False}, "exception_type": type(exc).__name__}
    record_triples = [tuple(str(v).strip() for v in triple) for triple in record.get("kg_subgraph") or [] if isinstance(triple, (list, tuple)) and len(triple) == 3]
    decision["checks"]["prompt_graph_exact_source_view"] = triples == record_triples
    eligible = bool(decision.get("m_graph")) and triples == record_triples
    decision["m_graph"], decision["graph_eligible"] = int(eligible), eligible
    if not triples == record_triples:
        decision["routing_reason"] = "prompt_graph_source_mismatch"
    if eligible and proof_result and proof_result.get("scorer_version") != SCORER_VERSION:
        raise ValueError("eligible source gate requires the explicit structural/answer v2.3 proof result")

    unique_triples = set(triples)
    nodes = {entity for h, _relation, t in unique_triples for entity in (h, t)}
    density = len(unique_triples) / (len(nodes) + 1e-6) if nodes else 0.0
    confidence_values, n_entities = [], 0
    for hop in (record.get("execution") or {}).get("hops") or []:
        for entity in hop.get("input_entities") or []:
            n_entities += 1
            raw = entity.get("score") if isinstance(entity, Mapping) else None
            try:
                value = _finite(raw, "frozen entity score")
            except (TypeError, ValueError):
                continue
            if 0 <= value <= 1:
                confidence_values.append(value)
    # Missing confidences are explicitly zero contributions, not fabricated
    # confident links.  This score is a frozen resolver proxy, not calibration.
    confidence = sum(confidence_values) / n_entities if n_entities else 0.0
    known = {tuple(triple) for step in steps for triple in getattr(step, "cited_triples", [])}
    unknown = {str(surface).strip() for step in steps for surface in getattr(step, "unknown_citation_surfaces", []) if str(surface).strip()}
    malformed = sum(bool(getattr(step, "knowledge_used_malformed_content", False)) for step in steps)
    denominator = len(known) + len(unknown) + malformed
    values = {
        "density": float(density), "link_confidence": float(confidence),
        "cite_any": float(denominator > 0),
        "cite_match": len(known & unique_triples) / denominator if denominator else 0.0,
    }
    return {
        "feature_version": FEATURE_VERSION, "values": values, "m_graph": int(eligible), "hard_gate": decision,
        "telemetry": {"confidence_observed": len(confidence_values), "confidence_occurrences": n_entities,
                      "known_unique_citations": len(known), "unknown_unique_citations": len(unknown),
                      "malformed_citation_fields": malformed, "policy_entropy_used": False},
    }


def heuristic_ratio_target(raw_graph: float, raw_text: Sequence[float], *, m_graph: int, trajectory_valid: bool) -> dict[str, Any]:
    """The predeclared score-ratio target with traceable abstention reasons."""
    if not m_graph:
        return {"target": None, "q_graph": None, "q_text": None, "abstain_reason": "graph_ineligible"}
    if not trajectory_valid:
        return {"target": None, "q_graph": None, "q_text": None, "abstain_reason": "invalid_trajectory"}
    graph = _finite(raw_graph, "raw_graph")
    if graph < -1e-12 or graph > .85 + 1e-12:
        raise ValueError("valid raw_graph must be a v2.3 score in [0,.85]")
    if not raw_text:
        return {"target": None, "q_graph": None, "q_text": None, "abstain_reason": "missing_text_scores"}
    text = [_finite(value, "raw_text") for value in raw_text]
    if any(value < -1 or value > 1 for value in text):
        raise ValueError("raw ReaRAG scores must be in [-1,1]")
    q_graph = max(0.0, min(1.0, graph / .85))
    q_text = sum((value + 1.0) / 2.0 for value in text) / len(text)
    denominator = q_graph + q_text
    reason = "zero_denominator" if denominator <= DENOMINATOR_EPSILON else "both_quality_low" if max(q_graph, q_text) <= QUALITY_ABSTAIN_THRESHOLD else None
    return {"target": None if reason else q_graph / denominator, "q_graph": q_graph, "q_text": q_text, "abstain_reason": reason}


def _feature_vector(features: Mapping[str, Any]) -> list[float]:
    if features.get("feature_version") != FEATURE_VERSION or set(features.get("values") or {}) != set(FEATURE_NAMES):
        raise ValueError("source quality feature schema/version mismatch")
    values = [_finite(features["values"][key], key) for key in FEATURE_NAMES]
    if values[0] < 0 or any(value < 0 or value > 1 for value in values[1:]):
        raise ValueError("source quality features outside the registered range")
    return values


class SourceQualityGateV1:
    """A tiny logistic gate with JSON-only, hashed, explicit successor artifacts."""

    def __init__(self, artifact: Mapping[str, Any], *, allow_synthetic: bool = False, allow_unvalidated: bool = False):
        data = deepcopy(dict(artifact))
        digest = data.pop("payload_sha256", None)
        if digest != canonical_sha256(data):
            raise ValueError("source quality gate payload hash mismatch")
        if data.get("schema_version") != ARTIFACT_SCHEMA or data.get("gate_version") != GATE_VERSION or data.get("feature_version") != FEATURE_VERSION or tuple(data.get("feature_names") or ()) != FEATURE_NAMES:
            raise ValueError("legacy/unknown alpha artifacts are not source-quality-gate-v1")
        if data.get("bank_source") != "real_frozen_policy_rollouts" and not allow_synthetic:
            raise ValueError("synthetic gate is diagnostic-only; no production load")
        if data.get("training_clearance") is not True and not allow_unvalidated:
            raise ValueError("gate heuristic calibration/confirmation clearance is missing")
        if data.get("target_version") != TARGET_VERSION:
            raise ValueError("source gate target version mismatch")
        self._weights = tuple(_finite(v, "weight") for v in data.get("weights") or [])
        if len(self._weights) != len(FEATURE_NAMES):
            raise ValueError("source gate must have exactly four weights")
        self._bias = _finite(data.get("bias"), "bias")
        standardization = data.get("feature_standardization") or {}
        self._means = tuple(_finite(standardization["mean"][key], "feature mean") for key in FEATURE_NAMES)
        self._scales = tuple(_finite(standardization["scale"][key], "feature scale") for key in FEATURE_NAMES)
        if any(scale <= 0 for scale in self._scales):
            raise ValueError("feature scales must be positive")
        norm = data.get("normalization") or {}
        for key in ("graph_center", "graph_scale", "text_center", "text_scale", "fixed_alpha"):
            norm[key] = _finite(norm.get(key), key)
        if min(norm["graph_scale"], norm["text_scale"]) < SOURCE_SCALE_FLOOR or not 0 <= norm["fixed_alpha"] <= 1:
            raise ValueError("source scales/fixed alpha violate frozen bounds")
        if norm.get("input_contract") != "raw_v23_graph_score_and_mean_raw_rearag_step_scores":
            raise ValueError("source reward normalization input contract mismatch")
        data["payload_sha256"] = digest
        self.artifact = data

    @classmethod
    def load(cls, path: str | Path, *, allow_synthetic: bool = False, allow_unvalidated: bool = False) -> "SourceQualityGateV1":
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("only versioned source-quality JSON artifacts are supported; legacy .pt is forbidden") from exc
        return cls(artifact, allow_synthetic=allow_synthetic, allow_unvalidated=allow_unvalidated)

    @property
    def normalization(self) -> dict[str, Any]:
        return deepcopy(self.artifact["normalization"])

    def predict(self, features: Mapping[str, Any]) -> float:
        values = _feature_vector(features)
        if features.get("m_graph") != 1:
            return 0.0
        hard_gate = features.get("hard_gate") or {}
        if hard_gate.get("m_graph") != 1 or not hard_gate.get("checks") or not all(value is True for value in hard_gate["checks"].values()):
            raise ValueError("nonzero alpha requires the verified fail-closed graph decision")
        logit = self._bias + sum(weight * (value - mean) / scale for weight, value, mean, scale in zip(self._weights, values, self._means, self._scales))
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
