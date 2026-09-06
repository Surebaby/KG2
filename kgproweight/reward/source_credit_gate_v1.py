"""Frozen source credit exclusion; candidate text and visible input stay intact.

This opt-in successor only withdraws Graph reward credit. A source PASS is a
bound integrity check, never a claim that model input or benchmark truth was
repaired. Old gate loaders reject the distinct artifact schema.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.reward import source_integrity_v1 as integrity
from kgproweight.reward.source_quality_gate_v1 import (
    ARTIFACT_SCHEMA as LEGACY_ARTIFACT_SCHEMA,
    SourceQualityGateV1, canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
MASK_SCHEMA = "source-credit-mask-manifest-v1"
MASK_VERSION = "source-credit-mask-v1"
ARTIFACT_SCHEMA = "source-quality-gate-source-credit-artifact-v1"
CREDIT_SCOPE = "reward_credit_only_input_unchanged"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_path(binding: Mapping[str, Any], base: Path) -> Path:
    if (not isinstance(binding, Mapping) or not isinstance(binding.get("path"), str)
            or not binding["path"] or not isinstance(binding.get("sha256"), str)
            or not _SHA.fullmatch(binding["sha256"])):
        raise ValueError("source credit requires a path and SHA256 file binding")
    raw = Path(binding["path"])
    for path in ([raw] if raw.is_absolute() else [base / raw, ROOT / raw]):
        if path.is_file() and _file_sha(path) == binding["sha256"]:
            return path.resolve()
    raise ValueError(f"source credit bound file missing or SHA256 mismatch: {raw}")


def _verify_payload(data: Mapping[str, Any], label: str) -> dict[str, Any]:
    result = deepcopy(dict(data))
    digest = result.pop("payload_sha256", None)
    if digest != canonical_sha256(result):
        raise ValueError(f"{label} payload hash mismatch")
    result["payload_sha256"] = digest
    return result


class FrozenSourceCreditMask:
    """Immutable verified decisions indexed by question identity and record hash."""

    @classmethod
    def load(cls, path: str | Path) -> "FrozenSourceCreditMask":
        manifest_path = Path(path).resolve()
        data = _verify_payload(json.loads(manifest_path.read_text(encoding="utf-8")), "source credit mask")
        if data.get("schema_version") != MASK_SCHEMA or data.get("mask_version") != MASK_VERSION:
            raise ValueError("unsupported source credit mask schema/version")
        if not isinstance(data.get("experiment_id"), str) or not data["experiment_id"].strip():
            raise ValueError("source credit mask requires experiment_id")
        paths = {name: _bound_path(data[name], manifest_path.parent) for name in (
            "inputs", "question_checks", "source_evidence", "verifier_code")}
        if _file_sha(Path(integrity.__file__)) != data["verifier_code"]["sha256"]:
            raise ValueError("source credit verifier code is not the frozen bound implementation")
        evidence = json.loads(paths["source_evidence"].read_text(encoding="utf-8"))
        # Check every nested evidence binding against bytes, once per identity.
        discovered: dict[str, str] = {}
        def inspect(value: Any) -> None:
            if isinstance(value, Mapping):
                bound = value.get("bindings")
                if isinstance(bound, Mapping):
                    for name, digest in bound.items():
                        if isinstance(digest, str):
                            if name in discovered and discovered[name] != digest:
                                raise ValueError("conflicting source evidence file bindings")
                            discovered[name] = digest
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
        inspect(evidence)
        for name, digest in discovered.items():
            _bound_path({"path": name, "sha256": digest}, paths["source_evidence"].parent)
        inputs = [json.loads(line) for line in paths["inputs"].read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [json.loads(line) for line in paths["question_checks"].read_text(encoding="utf-8").splitlines() if line.strip()]
        checked = {row["question_key"]: row for row in rows}
        if len(checked) != len(rows) or not inputs:
            raise ValueError("duplicate/empty source credit identities")
        entries = {}
        for row in inputs:
            key = question_key(row["dataset"], row["qid"])
            if key in entries or key != row.get("question_key") or key not in checked:
                raise ValueError("source credit question identity mismatch or missing audit")
            record = row.get("fullsource_record")
            if not isinstance(record, Mapping):
                raise ValueError("source credit requires complete source record")
            digest = canonical_sha256(record)
            qhash = question_sha256(row["question"])
            if (digest != row.get("source_record_sha256")
                    or question_key(record.get("dataset", ""), record.get("qid", "")) != key
                    or record.get("question_sha256") != qhash
                    or row.get("question_sha256") != qhash):
                raise ValueError("source credit original record digest or question identity mismatch")
            actual = integrity.validate_source_integrity_v1(record, evidence)
            expected = {**actual, "question_key": key, "original_m_graph": row["m_graph"],
                        "input_sha256": row["input_sha256"]}
            if canonical_sha256(expected) != canonical_sha256(checked[key]):
                raise ValueError("source credit audit does not reproduce from bound evidence/record")
            entries[key] = {"question_sha256": qhash, "record_sha256": digest,
                            "status": actual["status"], "clearance": actual["clearance"],
                            "original_m_graph": row["m_graph"],
                            "graph_sha256": canonical_sha256(record.get("kg_subgraph") or [])}
        if set(entries) != set(checked):
            raise ValueError("source credit audit/input question populations differ")
        instance = cls()
        instance._manifest = data
        instance._entries = entries
        instance.manifest_path = manifest_path
        instance.manifest_sha256 = _file_sha(manifest_path)
        instance.payload_sha256 = data["payload_sha256"]
        return instance

    def mask_features(self, spec: Any, features: Mapping[str, Any]) -> dict[str, Any]:
        """Return a new feature view; only eligibility/diagnostics may change."""
        result = deepcopy(dict(features))
        metadata = getattr(spec, "metadata", {}) or {}
        record = metadata.get("source_quality_record")
        parent = features.get("m_graph")
        key, record_digest, status, accepted = "", "", "MISSING", False
        try:
            key = question_key(metadata.get("dataset", ""), metadata.get("qid", ""))
            entry = self._entries.get(key)
            if isinstance(record, Mapping):
                record_digest = canonical_sha256(record)
            if entry:
                status = entry["status"]
                identity = (
                    record_digest == entry["record_sha256"]
                    and question_sha256(getattr(spec, "query", "")) == entry["question_sha256"]
                    and canonical_sha256(getattr(spec, "kg_subgraph", []) or []) == entry["graph_sha256"]
                )
                if not identity:
                    status = "IDENTITY_MISMATCH"
                accepted = bool(identity and entry["clearance"] is True and status == "PASS"
                                and entry["original_m_graph"] == 1)
        except (TypeError, ValueError, KeyError):
            status = "IDENTITY_MISMATCH"
        eligible = parent == 1 and accepted
        result["m_graph"] = int(eligible)
        hard_gate = result.setdefault("hard_gate", {})
        hard_gate.setdefault("checks", {})["source_credit_pass"] = bool(accepted)
        hard_gate["m_graph"], hard_gate["graph_eligible"] = int(eligible), bool(eligible)
        if parent == 1 and not accepted:
            hard_gate["routing_reason"] = "source_credit_excluded"
        result.pop("source_credit_mask", None)
        marker = {"version": MASK_VERSION, "mask_payload_sha256": self.payload_sha256,
                  "question_key": key, "record_sha256": record_digest,
                  "status": status, "parent_m_graph": parent,
                  "source_credit_pass": accepted, "features_sha256": canonical_sha256(result)}
        result["source_credit_mask"] = marker
        return result

    def validate_masked_features(self, features: Mapping[str, Any]) -> None:
        marker = features.get("source_credit_mask")
        if (not isinstance(marker, Mapping) or marker.get("version") != MASK_VERSION
                or marker.get("mask_payload_sha256") != self.payload_sha256):
            raise ValueError("source credit predict requires features processed by its frozen mask")
        raw = dict(features)
        raw.pop("source_credit_mask")
        if marker.get("features_sha256") != canonical_sha256(raw):
            raise ValueError("source credit masked features changed after masking")
        if features.get("m_graph") == 1:
            entry = self._entries.get(marker.get("question_key"))
            if (not entry or entry["clearance"] is not True or entry["status"] != "PASS"
                    or marker.get("record_sha256") != entry["record_sha256"]
                    or marker.get("source_credit_pass") is not True
                    or marker.get("status") != "PASS" or marker.get("parent_m_graph") != 1):
                raise ValueError("source credit nonzero alpha lacks a matching cleared source")


class SourceCreditGateV1(SourceQualityGateV1):
    """Old four-dimensional fit with a mandatory frozen source credit mask."""

    def __init__(self, artifact: Mapping[str, Any], *, mask: FrozenSourceCreditMask,
                 allow_synthetic: bool = False, allow_unvalidated: bool = False):
        data = _verify_payload(artifact, "source credit gate")
        if (data.get("schema_version") != ARTIFACT_SCHEMA
                or data.get("source_credit_version") != MASK_VERSION
                or data.get("source_credit_scope") != CREDIT_SCOPE
                or data.get("source_credit_clearance") is not True):
            raise ValueError("source credit gate schema/scope/clearance mismatch")
        bound = data.get("source_credit_mask") or {}
        if (not isinstance(mask, FrozenSourceCreditMask)
                or bound.get("sha256") != mask.manifest_sha256
                or bound.get("payload_sha256") != mask.payload_sha256):
            raise ValueError("source credit gate mask binding mismatch")
        compatible = deepcopy(data)
        compatible["schema_version"] = LEGACY_ARTIFACT_SCHEMA
        compatible.pop("payload_sha256")
        compatible["payload_sha256"] = canonical_sha256(compatible)
        super().__init__(compatible, allow_synthetic=allow_synthetic, allow_unvalidated=allow_unvalidated)
        self.artifact = data
        self.mask = mask

    @classmethod
    def load(cls, path: str | Path, *, allow_synthetic: bool = False,
             allow_unvalidated: bool = False) -> "SourceCreditGateV1":
        path = Path(path).resolve()
        data = _verify_payload(json.loads(path.read_text(encoding="utf-8")), "source credit gate")
        bound_path = _bound_path(data.get("source_credit_mask"), path.parent)
        mask = FrozenSourceCreditMask.load(bound_path)
        return cls(data, mask=mask, allow_synthetic=allow_synthetic, allow_unvalidated=allow_unvalidated)

    def mask_features(self, spec: Any, features: Mapping[str, Any]) -> dict[str, Any]:
        return self.mask.mask_features(spec, features)

    def predict(self, features: Mapping[str, Any]) -> float:
        self.mask.validate_masked_features(features)
        return super().predict(features)


def load_source_quality_gate(path: str | Path, *, allow_synthetic: bool = False,
                             allow_unvalidated: bool = False) -> SourceQualityGateV1:
    """Explicit schema dispatch; never silently adapt a new artifact as legacy."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = data.get("schema_version")
    if schema == LEGACY_ARTIFACT_SCHEMA:
        return SourceQualityGateV1.load(path, allow_synthetic=allow_synthetic, allow_unvalidated=allow_unvalidated)
    if schema == ARTIFACT_SCHEMA:
        return SourceCreditGateV1.load(path, allow_synthetic=allow_synthetic, allow_unvalidated=allow_unvalidated)
    raise ValueError("unsupported source quality/credit gate schema")
