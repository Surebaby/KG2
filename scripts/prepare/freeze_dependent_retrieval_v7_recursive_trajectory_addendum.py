#!/usr/bin/env python
"""Freeze the pre-execution v7 recursive-trajectory estimand addendum.

The original design, cohort preregistration and producer-view truncation
addendum remain immutable.  This CPU-only command verifies all parents and
writes a new append-only effective addendum.  It performs no planner, model,
retrieval, Gold or scoring call and grants no execution authorization.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-1"
STATUS = "FROZEN_APPEND_ONLY_BEFORE_V7_EXECUTION"
SCOPE = "RECURSIVE_ARM_SPECIFIC_PRODUCER_CONTEXT_ESTIMAND_CLARIFICATION"
EXPERIMENT_ID = (
    "SUBQUESTION-DEPENDENT-RETRIEVAL-V7-DEV20X2-SEED20260904-"
    "RECURSIVE-TRAJECTORY-ADDENDUM"
)

DEFAULT_PATHS = {
    "design_protocol": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/protocol.json"
    ),
    "design_manifest": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/manifest.json"
    ),
    "parent_preregistration": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/protocol.json"
    ),
    "parent_preregistration_manifest": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_development_preregistration/manifest.json"
    ),
    "producer_truncation_addendum": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_development_"
        "preregistration_addendum_producer_truncation_v1/protocol.json"
    ),
    "producer_truncation_addendum_manifest": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_development_"
        "preregistration_addendum_producer_truncation_v1/manifest.json"
    ),
    "design_trajectory_addendum": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/"
        "addendum_recursive_trajectory_v1.json"
    ),
    "design_trajectory_addendum_manifest": Path(
        "outputs/audits/subquestion_dependent_retrieval_v7_design_freeze/"
        "addendum_recursive_trajectory_v1.manifest.json"
    ),
}
EXPECTED_SHA256 = {
    "design_protocol": "a11babfa816ea8373d1e2725d115dd3ce85028af8986a2b00fed1ca1efd54d7e",
    "design_manifest": "39f880a818df8353857ed357c7d856b12940af4f905869a03fca7a5110f3b657",
    "parent_preregistration": "c7f1674f62a191671a22844e5589c3f9b80a990aae3c0344cd4001e47a50395d",
    "parent_preregistration_manifest": "01ab042c1f824dc649cddc5aa929393efe17aab9631ec07de70fcb5f8dda19cb",
    "producer_truncation_addendum": "0c93f29ab9356b0a818ce398482f096a67533c487e9bdca44271eb8c9a8ecf78",
    "producer_truncation_addendum_manifest": "ef392b197cd5037614a206ae703d5f2b4b06759f7d6d8187e279a8f4b0bf3cd4",
    "design_trajectory_addendum": "daa35f368cd6ac6cad8ab7646560c7e3086fb45cc765912241a9e7979e752bba",
}
DEFAULT_OUT = Path(
    "outputs/audits/subquestion_dependent_retrieval_v7_development_"
    "preregistration_addendum_recursive_trajectory_v1"
)
EXPECTED_INVARIANTS = {
    "shared_root_identical_query_requires_identical_producer_passages": True,
    "divergent_upstream_bridges_may_induce_arm_specific_producer_passages": True,
    "estimand": "C_MINUS_B_TOTAL_RECURSIVE_TRAJECTORY_EFFECT",
    "per_hop_logical_retrieval_budget_equal": True,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _verify_manifest(
    manifest: Mapping[str, Any], protocol_lock: Mapping[str, Any], *, label: str
) -> None:
    if manifest.get("gold_access") is not False:
        raise ValueError(f"{label} Gold boundary drift")
    recorded = manifest.get("protocol")
    if not isinstance(recorded, Mapping):
        recorded = (manifest.get("artifacts") or {}).get("protocol")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"{label} has no protocol lock")
    if str(recorded.get("sha256") or "") != str(protocol_lock["sha256"]):
        raise ValueError(f"{label} protocol hash drift")


def build_protocol(paths: Mapping[str, Path]) -> dict[str, Any]:
    locks = {name: _file_lock(path) for name, path in paths.items()}
    for name, expected in EXPECTED_SHA256.items():
        if locks[name]["sha256"] != expected:
            raise ValueError(f"frozen parent SHA256 drift: {name}")

    design = _read_object(paths["design_protocol"])
    design_manifest = _read_object(paths["design_manifest"])
    prereg = _read_object(paths["parent_preregistration"])
    prereg_manifest = _read_object(paths["parent_preregistration_manifest"])
    truncation = _read_object(paths["producer_truncation_addendum"])
    truncation_manifest = _read_object(paths["producer_truncation_addendum_manifest"])
    source = _read_object(paths["design_trajectory_addendum"])
    source_manifest = _read_object(paths["design_trajectory_addendum_manifest"])

    gold_policy = design.get("gold_policy") or {}
    if gold_policy.get("freeze_planner_retrieval_subanswer_and_merge_may_read_gold") is not False:
        raise ValueError("design Gold boundary drift")
    _verify_manifest(design_manifest, locks["design_protocol"], label="design manifest")
    if prereg.get("execution_authorization") != (
        "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK"
    ):
        raise ValueError("preregistration execution boundary drift")
    _verify_manifest(
        prereg_manifest, locks["parent_preregistration"], label="prereg manifest"
    )
    if truncation.get("execution_authorization") != (
        "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK"
    ):
        raise ValueError("truncation addendum execution boundary drift")
    _verify_manifest(
        truncation_manifest,
        locks["producer_truncation_addendum"],
        label="truncation manifest",
    )

    if source.get("schema_version") != (
        "subquestion-dependent-retrieval-v7-recursive-trajectory-design-addendum-1"
    ):
        raise ValueError("source trajectory addendum schema drift")
    if source.get("status") != STATUS or source.get("scope") != SCOPE:
        raise ValueError("source trajectory addendum status/scope drift")
    if source.get("effective_invariants") != EXPECTED_INVARIANTS:
        raise ValueError("source trajectory invariants drift")
    if source.get("gold_access") is not False:
        raise ValueError("source trajectory addendum Gold boundary drift")
    source_parents = source.get("parents") or {}
    for name in (
        "design_protocol",
        "design_manifest",
        "parent_preregistration",
        "parent_preregistration_manifest",
        "producer_truncation_addendum",
        "producer_truncation_addendum_manifest",
    ):
        if str((source_parents.get(name) or {}).get("sha256") or "") != str(
            locks[name]["sha256"]
        ):
            raise ValueError(f"source trajectory parent drift: {name}")
    if source_manifest.get("status") != STATUS:
        raise ValueError("source trajectory manifest status drift")
    if str((source_manifest.get("addendum") or {}).get("sha256") or "") != str(
        locks["design_trajectory_addendum"]["sha256"]
    ):
        raise ValueError("source trajectory manifest content-lock drift")

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "scope": SCOPE,
        "parents": locks,
        "effective_invariants": dict(EXPECTED_INVARIANTS),
        "effective_interpretation": dict(source["effective_interpretation"]),
        "supersedes_only": list(source["supersedes_only"]),
        "unchanged": list(source["unchanged"]),
        "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
        "gold_access": False,
        "gpu_calls": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "parent_protocols_mutated": False,
        "scientific_boundary": source["scientific_boundary"],
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_protocol(protocol: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v7 trajectory addendum: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = output_dir / "protocol.json"
    _write_json_exclusive(protocol_path, protocol)
    manifest = {
        "schema_version": (
            "subquestion-dependent-retrieval-v7-recursive-trajectory-addendum-manifest-1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "protocol": _file_lock(protocol_path),
        "gold_access": False,
        "gpu_calls": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "execution_authorization": "BLOCKED_UNTIL_SEPARATE_IMPLEMENTATION_HASH_LOCK",
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    return {"protocol": _file_lock(protocol_path), "manifest": _file_lock(manifest_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: getattr(args, name) for name in DEFAULT_PATHS}
    protocol = build_protocol(paths)
    result = write_protocol(protocol, args.out)
    print(json.dumps({"status": STATUS, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
