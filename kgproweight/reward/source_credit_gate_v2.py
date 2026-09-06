"""Explicit experimental successor for step-scale and trajectory features.

The old source-credit mask is reused exactly.  Fitting or reanalysing an already
consumed bank never constitutes fresh confirmation; unconfirmed artifacts are
loadable only with the explicit diagnostic flag, never by the PPO CLI default.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from kgproweight.reward.source_credit_gate_v1 import (
    CREDIT_SCOPE, MASK_VERSION, FrozenSourceCreditMask, _bound_path, _verify_payload,
)
from kgproweight.reward.source_quality_gate_v1 import (
    FEATURE_NAMES as V1_NAMES, FEATURE_VERSION as V1_VERSION, TARGET_VERSION,
    compute_gate_features,
)
from kgproweight.reward.source_reward_normalization_v2 import validate_text_normalization_v2


ARTIFACT_SCHEMA = "source-quality-gate-source-credit-artifact-v2"
GATE_VERSION = "source-credit-gate-v2"
NORMALIZATION_CONTRACT = "raw_v23_graph_and_step_rearag_text_v2"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def registered_features(version: str) -> tuple[str, ...]:
    if version == V1_VERSION:
        return V1_NAMES
    from kgproweight.reward.source_trajectory_features_v2 import FEATURE_NAMES, FEATURE_VERSION
    if version == FEATURE_VERSION:
        return FEATURE_NAMES
    raise ValueError("unregistered source-credit-v2 feature version")


class SourceCreditGateV2:
    """Frozen logistic gate with versioned features and softsign text scores."""

    def __init__(self, artifact: Mapping[str, Any], *, mask: FrozenSourceCreditMask,
                 allow_synthetic: bool = False, allow_unvalidated: bool = False,
                 runtime_config: Any = None, artifact_path: str | Path | None = None):
        data = _verify_payload(artifact, "source credit v2 gate")
        if "execution_scope" in data:
            from kgproweight.reward.source_gate_bounded_dispatch_v1 import validate_bounded_scope
            self.execution_scope_validation = validate_bounded_scope(data, runtime_config, artifact_path, mask)
        if (data.get("schema_version") != ARTIFACT_SCHEMA
                or data.get("gate_version") != GATE_VERSION
                or data.get("source_credit_version") != MASK_VERSION
                or data.get("source_credit_scope") != CREDIT_SCOPE
                or data.get("source_credit_clearance") is not True):
            raise ValueError("source credit v2 schema, scope or clearance mismatch")
        if data.get("format_contract_version") != "source-gate-runtime-v2-format-v2":
            raise ValueError("source credit v2 requires the unchanged format-v2 contract")
        if data.get("target_version") != TARGET_VERSION:
            raise ValueError("source credit v2 does not silently replace the ratio target")
        if data.get("bank_source") != "real_frozen_policy_rollouts" and not allow_synthetic:
            raise ValueError("synthetic gates are diagnostic only")
        if data.get("training_clearance") is not True and not allow_unvalidated:
            raise ValueError("source credit v2 requires fresh confirmation before production load")
        if data.get("training_clearance") is True and data.get("independent_confirmation_clearance") is not True:
            raise ValueError("reused development diagnostics cannot authorize source credit v2 training")
        if data.get("training_clearance") is True and "execution_scope" not in data:
            raise ValueError("v2 training clearance requires registered bounded execution scope")
        bound = data.get("source_credit_mask") or {}
        if (not isinstance(mask, FrozenSourceCreditMask)
                or bound.get("sha256") != mask.manifest_sha256
                or bound.get("payload_sha256") != mask.payload_sha256):
            raise ValueError("source credit v2 mask binding mismatch")
        names = registered_features(data.get("feature_version"))
        if tuple(data.get("feature_names") or ()) != names:
            raise ValueError("source credit v2 feature names/order mismatch")
        self._names = names
        self._weights = tuple(_finite(v, "weight") for v in data.get("weights") or ())
        if len(self._weights) != len(names):
            raise ValueError("weight count must equal registered feature count")
        self._bias = _finite(data.get("bias"), "bias")
        standard = data.get("feature_standardization") or {}
        if set(standard.get("mean") or {}) != set(names) or set(standard.get("scale") or {}) != set(names):
            raise ValueError("feature standardization must bind all registered features")
        self._means = tuple(_finite(standard["mean"][key], "feature mean") for key in names)
        self._scales = tuple(_finite(standard["scale"][key], "feature scale") for key in names)
        if any(value <= 0 for value in self._scales):
            raise ValueError("feature scales must be positive")
        norm = data.get("normalization") or {}
        if norm.get("input_contract") != NORMALIZATION_CONTRACT:
            raise ValueError("source credit v2 normalization contract mismatch")
        text = validate_text_normalization_v2(norm.get("text_v2") or {})
        for key in ("graph_center", "graph_scale", "text_center", "text_scale", "fixed_alpha"):
            norm[key] = _finite(norm.get(key), key)
        if norm["graph_scale"] < .1 or not 0 <= norm["fixed_alpha"] <= 1:
            raise ValueError("graph scale or fixed alpha violates frozen limits")
        if (norm["text_center"] != text["text_center"] or norm["text_scale"] != text["text_scale"]
                or norm.get("text_application_scope") != text["application_contract"]):
            raise ValueError("text normalization metadata contradicts step-scale contract")
        if norm.get("graph_application_scope") != "trajectory_normalize_then_clip_v1":
            raise ValueError("source credit v2 preserves the graph normalization application")
        self.artifact = data
        self.mask = mask

    @classmethod
    def load(cls, path: str | Path, *, allow_synthetic: bool = False,
             allow_unvalidated: bool = False, runtime_config: Any = None) -> "SourceCreditGateV2":
        path = Path(path).resolve()
        data = _verify_payload(json.loads(path.read_text(encoding="utf-8")), "source credit v2 gate")
        if "execution_scope" in data and runtime_config is None:
            raise ValueError("scoped probe child requires exact runtime configuration")
        mask = FrozenSourceCreditMask.load(_bound_path(data.get("source_credit_mask"), path.parent))
        return cls(data, mask=mask, allow_synthetic=allow_synthetic, allow_unvalidated=allow_unvalidated,
                   runtime_config=runtime_config, artifact_path=path)

    @property
    def normalization(self) -> dict[str, Any]:
        return deepcopy(self.artifact["normalization"])

    def compute_features(self, spec: Any, steps, proof_result) -> dict[str, Any]:
        if self.artifact["feature_version"] == V1_VERSION:
            return compute_gate_features(spec, steps, proof_result)
        from kgproweight.reward.source_trajectory_features_v2 import compute_gate_features_v2
        return compute_gate_features_v2(spec, steps, proof_result)

    def mask_features(self, spec: Any, features: Mapping[str, Any]) -> dict[str, Any]:
        return self.mask.mask_features(spec, features)

    def predict(self, features: Mapping[str, Any]) -> float:
        self.mask.validate_masked_features(features)
        if (features.get("feature_version") != self.artifact["feature_version"]
                or set(features.get("values") or {}) != set(self._names)):
            raise ValueError("source credit v2 feature version or fields mismatch")
        values = [_finite(features["values"][name], name) for name in self._names]
        for name, value in zip(self._names, values):
            if value < 0 or (name != "density" and value > 1):
                raise ValueError("source credit v2 feature outside its registered range")
        if features.get("m_graph") != 1:
            return 0.0
        hard = features.get("hard_gate") or {}
        if hard.get("m_graph") != 1 or not hard.get("checks") or not all(v is True for v in hard["checks"].values()):
            raise ValueError("source credit v2 requires all bound source checks for nonzero alpha")
        logit = self._bias + sum(w * (x - m) / s for w, x, m, s in zip(self._weights, values, self._means, self._scales))
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
