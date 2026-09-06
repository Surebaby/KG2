#!/usr/bin/env python3
"""Freeze the implementation/source contract for official-raw unified v3.

The contract is intentionally frozen after the materializer and its tests are
final, but before any release is materialized.  It binds the exact executable
and the upstream policy schemas while leaving result-dependent file identities
to the later release report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_2wiki_official_raw_canonical_retrieval_v1 import (
    POLICY_SCHEMA as RETRIEVAL_POLICY_SCHEMA,
    POLICY_STATUS as RETRIEVAL_POLICY_STATUS,
    RETRIEVAL_STACK,
    SCOPE_SCHEMA as RETRIEVAL_SCOPE_SCHEMA,
    SCOPE_STATUS as RETRIEVAL_SCOPE_STATUS,
)
from scripts.prepare.materialize_2wiki_official_raw_canonical_retrieval_v1 import (
    REPORT_SCHEMA_VERSION as RETRIEVAL_REPORT_SCHEMA,
    SCHEMA_VERSION as RETRIEVAL_CONTEXT_SCHEMA,
    STATUS as RETRIEVAL_RELEASE_STATUS,
)
from scripts.prepare.materialize_2wiki_proofkg_unified_v3 import (
    CANDIDATE_SCHEMA_VERSION,
    CONTRACT_SCHEMA_VERSION,
    CONTRACT_STATUS,
    REQUIRED_OUTPUTS,
    SCHEMA_VERSION as RELEASE_SCHEMA_VERSION,
    SOURCE_RELEASE,
    STATUS as RELEASE_STATUS,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "2WIKI-UNIFIED-PROOFKG-OFFICIAL-RAW-V3-SOURCE-CONTRACT"
DEFAULT_OUT = ROOT / "outputs/audits/2wiki_unified_proofkg_official_raw_v3_contract"
DEFAULT_RETRIEVAL_POLICY = ROOT / (
    "outputs/audits/2wiki_official_raw_canonical_retrieval_v1_scope_policy_"
    "preregistration/protocol.json"
)
DEFAULT_CLOSURE_POLICY = ROOT / (
    "outputs/audits/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_"
    "clean_closure_v3_preregistration/protocol.json"
)
MATERIALIZER = ROOT / "scripts/prepare/materialize_2wiki_proofkg_unified_v3.py"
RETRIEVAL_MATERIALIZER = ROOT / (
    "scripts/prepare/materialize_2wiki_official_raw_canonical_retrieval_v1.py"
)
SOURCE_GATE = ROOT / "kgproweight/reward/trajectory_source_gate.py"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_policy(
    path: Path, *, schema: str, status: str, label: str
) -> dict[str, Any]:
    value = _read_json(path)
    if (
        value.get("schema_version") != schema
        or value.get("status") != status
        or value.get("training_started") is True
    ):
        # Several protocol schemas put the flag only in scientific_boundary.
        boundary = value.get("scientific_boundary") or {}
        if not (
            value.get("schema_version") == schema
            and value.get("status") == status
            and boundary.get("training_started") is False
        ):
            raise ValueError(f"{label} schema/status/training boundary failed")
    return _identity(path)


def freeze_contract(
    *, retrieval_policy: Path, closure_policy: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite unified-v3 contract: {output_dir}")
    retrieval_policy_ref = _validate_policy(
        retrieval_policy,
        schema=RETRIEVAL_POLICY_SCHEMA,
        status=RETRIEVAL_POLICY_STATUS,
        label="retrieval scope policy",
    )
    closure_policy_ref = _validate_policy(
        closure_policy,
        schema="2wiki-official-raw-n1500-clean-closure-policy-v3",
        status="FROZEN_POLICY_WAITING_FOR_ROOT_RESOLUTION",
        label="closure-v3 policy",
    )
    implementation = _identity(MATERIALIZER)
    retrieval_implementation = _identity(RETRIEVAL_MATERIALIZER)
    source_gate = _identity(SOURCE_GATE)
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": CONTRACT_STATUS,
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "release_status": RELEASE_STATUS,
        "candidate_wrapper_schema_version": CANDIDATE_SCHEMA_VERSION,
        "required_outputs": list(REQUIRED_OUTPUTS),
        # The selector-v2 contract validator requires exactly one canonical
        # implementation identity; do not duplicate this under code.materializer.
        "implementation": implementation,
        "source_semantics": {
            "source_release": SOURCE_RELEASE,
            "population": (
                "all and only answer-free strict structural eligible rows from "
                "the frozen official-raw n1500 closure-v3 population"
            ),
            "canonical_passages": {
                "scope_schema": RETRIEVAL_SCOPE_SCHEMA,
                "scope_status": RETRIEVAL_SCOPE_STATUS,
                "context_schema": RETRIEVAL_CONTEXT_SCHEMA,
                "report_schema": RETRIEVAL_REPORT_SCHEMA,
                "release_status": RETRIEVAL_RELEASE_STATUS,
                "stack": RETRIEVAL_STACK,
                "exactly_ten": True,
                "bge_backend_fallback": False,
                "reserve50_schema_or_status_reused": False,
            },
            "proof_graph": {
                "source": "closure-v3 runtime exact subset",
                "historical_cutoff": "2020-12-09T23:59:59Z",
                "m_graph_required": 1,
                "all_source_gate_checks_required": True,
            },
            "gold_boundary": {
                "scope_selection_or_graph_construction_uses_gold": False,
                "source_gold_steps_or_kg_copied": False,
                "official_train_answer_use": "outcome_reward_label_only_after_scope_freeze",
            },
            "final_proof800_selection_performed": False,
        },
        "upstream_contracts": {
            "retrieval_scope_policy": retrieval_policy_ref,
            "closure_v3_policy": closure_policy_ref,
        },
        "code": {
            "retrieval_materializer": retrieval_implementation,
            "source_gate": source_gate,
        },
        "scientific_boundary": {
            "result_dependent_scope_file_not_yet_bound": True,
            "retrieval_not_started": True,
            "unified_release_not_materialized": True,
            "proof800_not_selected": True,
            "training_started": False,
        },
        "training_started": False,
    }
    contract_path = output_dir / "unified_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": f"{CONTRACT_SCHEMA_VERSION}-report",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": CONTRACT_STATUS,
        "contract": _identity(contract_path),
        "implementation": implementation,
        "retrieval_started": False,
        "proof800_selected": False,
        "training_started": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        output_dir,
        status=CONTRACT_STATUS,
        extra={
            "phase": "freeze_2wiki_unified_proofkg_official_raw_v3_contract",
            "experiment_id": EXPERIMENT_ID,
            "contract": _identity(contract_path),
            "report": _identity(report_path),
            "retrieval_started": False,
            "proof800_selected": False,
            "training_started": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-policy", type=Path, default=DEFAULT_RETRIEVAL_POLICY)
    parser.add_argument("--closure-policy", type=Path, default=DEFAULT_CLOSURE_POLICY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_contract(
                retrieval_policy=args.retrieval_policy,
                closure_policy=args.closure_policy,
                output_dir=args.out,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
