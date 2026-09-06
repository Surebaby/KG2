#!/usr/bin/env python
"""Freeze the Gold-free v9.1 rank-first full-chain pilot protocol.

This freezer never reads Gold or the sealed prospective900.  It binds the
already-consumed v8 smoke cohort and the separately frozen fresh train-side
pilot30x3 cohort, plus the exact runtime/code lineage needed by the runner.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pilot import run_dynamic_decomposition_v9_1 as runner  # noqa: E402


SCHEMA_VERSION = "dynamic-decomposition-v9.1-protocol-1"
MANIFEST_SCHEMA_VERSION = "dynamic-decomposition-v9.1-protocol-manifest-1"
STATUS = "AUTHORIZED_SMOKE_THEN_CONDITIONAL_FRESH_PILOT90_GOLD_FREE"

SMOKE_COHORT = Path(
    "outputs/audits/subquestion_decomposition_v8_consumed_smoke4x3_seed20260904_v1/"
    "smoke.identity_only.jsonl"
)
FRESH_PILOT_COHORT = Path(
    "outputs/audits/subquestion_decomposition_v9_canonical_subqa_"
    "pilot30x3_seed20260904_v1/pilot.identity_only.jsonl"
)
V8_IMPLEMENTATION_DIR = Path(
    "outputs/audits/subquestion_decomposition_v8_development90_"
    "implementation_freeze_rrf100_seed42_v2"
)
V9_PHASE0_PROTOCOL_DIR = Path(
    "outputs/audits/subquestion_decomposition_v9_canonical_subqa_phase0_protocol_v1"
)
V9_PHASE0_RUN_DIR = Path(
    "outputs/audits/subquestion_decomposition_v9_canonical_subqa_"
    "phase0_consumed_dev90_seed42_v1"
)
FRESH_PILOT_FREEZE_DIR = Path(
    "outputs/audits/subquestion_decomposition_v9_canonical_subqa_"
    "pilot30x3_seed20260904_v1"
)

EXPECTED_PARENT_SHA256 = {
    "v8_implementation_protocol": (
        "3539d956a893cad119d583bdc875919b2349c47e503af9f05a83c887ebc039be"
    ),
    "v8_implementation_manifest": (
        "289d6ef40db546221a33af2fba31c016e15091a748ae8a85d87b4dd648ffb491"
    ),
    "v9_phase0_protocol": (
        "a22500e7325ea77323063870cc658bc8f4acc809f0bddd604f7dda3d2ebd304c"
    ),
    "v9_phase0_manifest": (
        "23f28d44589982aef20a12fec5e82fb5655501e5b371b34375748145e2f5a401"
    ),
    "v9_phase0_report": (
        "8dc02e7f4ace2559f85fc4b541b8b05d2e51b692b106c46a8b47017df5463495"
    ),
    "fresh_pilot_freeze_protocol": (
        "ce1faef7e5ec310e3cf4b77154fc68bdd1fa72c11563482afd39ae3a1798238c"
    ),
    "fresh_pilot_freeze_report": (
        "111b8593058cf349719c700c49e98e653c4101734bfc140bdfb8ca46aa2dbb00"
    ),
    "fresh_pilot_freeze_manifest": (
        "888287dae4ad112e45790a376f8496769de0ff1fb9a2f082cada320e323a656e"
    ),
}
EXPECTED_COHORT_SHA256 = {
    "consumed_smoke12": (
        "3b2eb4da9abefc09c3df97083aa65462d6e51e7648cc811ae59f8e8266671606"
    ),
    "fresh_pilot90": (
        "7f4c63eb5589ce342a59e9942d986992ce357eb2888a46a0d6b04dc38794a9d8"
    ),
}


class V91FreezeError(RuntimeError):
    """The v9.1 protocol lineage or boundary is invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_lock(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(dict(value)))


def _self_commit(value: Mapping[str, Any]) -> dict[str, Any]:
    if "protocol_body_canonical_sha256" in value:
        raise V91FreezeError("protocol already has a body commitment")
    result = dict(value)
    result["protocol_body_canonical_sha256"] = hashlib.sha256(
        _canonical_json_bytes(dict(value))
    ).hexdigest()
    return result


def _assert_expected_locks(
    locks: Mapping[str, Mapping[str, Any]], expected: Mapping[str, str]
) -> None:
    if set(locks) != set(expected):
        raise V91FreezeError("frozen lock role set mismatch")
    for name, digest in expected.items():
        if locks[name].get("sha256") != digest:
            raise V91FreezeError(f"frozen source drift: {name}")


def freeze() -> dict[str, Any]:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise V91FreezeError(f"run from project root: {PROJECT_ROOT}")
    destination = (PROJECT_ROOT / runner.PROTOCOL_DIR).resolve()
    if destination.exists():
        raise FileExistsError(f"append-only protocol directory exists: {destination}")

    code_paths = {
        "runner": Path("scripts/pilot/run_dynamic_decomposition_v9_1.py"),
        "rank_first_binder": Path("kgproweight/retrieval/canonical_subqa_v9_1.py"),
        "canonical_subqa_v9": Path("kgproweight/retrieval/canonical_subqa_v9.py"),
        "v8_binder_and_policy": Path("kgproweight/retrieval/dynamic_decomposition_v8.py"),
        "v8_materializer": Path("scripts/pilot/materialize_dynamic_decomposition_v8.py"),
        "v8_driver": Path("scripts/pilot/run_dynamic_decomposition_v8.py"),
        "canonical_prompts": Path("kgproweight/data/prompts.py"),
        "canonical_parsers": Path("kgproweight/data/parsers.py"),
    }
    code_locks = {
        name: _file_lock(PROJECT_ROOT / path) for name, path in code_paths.items()
    }

    cohort_locks = {
        "consumed_smoke12": _file_lock(PROJECT_ROOT / SMOKE_COHORT),
        "fresh_pilot90": _file_lock(PROJECT_ROOT / FRESH_PILOT_COHORT),
    }
    _assert_expected_locks(cohort_locks, EXPECTED_COHORT_SHA256)

    parent_paths = {
        "v8_implementation_protocol": V8_IMPLEMENTATION_DIR / "protocol.json",
        "v8_implementation_manifest": V8_IMPLEMENTATION_DIR / "manifest.json",
        "v9_phase0_protocol": V9_PHASE0_PROTOCOL_DIR / "protocol.json",
        "v9_phase0_manifest": V9_PHASE0_PROTOCOL_DIR / "manifest.json",
        "v9_phase0_report": V9_PHASE0_RUN_DIR / "report.json",
        "fresh_pilot_freeze_protocol": FRESH_PILOT_FREEZE_DIR / "protocol.json",
        "fresh_pilot_freeze_report": FRESH_PILOT_FREEZE_DIR / "report.json",
        "fresh_pilot_freeze_manifest": FRESH_PILOT_FREEZE_DIR / "manifest.json",
    }
    parent_locks = {
        name: _file_lock(PROJECT_ROOT / path) for name, path in parent_paths.items()
    }
    _assert_expected_locks(parent_locks, EXPECTED_PARENT_SHA256)

    cohort_specs = {
        "smoke": {
            "path": cohort_locks["consumed_smoke12"]["path"],
            "sha256": cohort_locks["consumed_smoke12"]["sha256"],
            "row_count": 12,
            "per_dataset": 4,
            "gold_access": False,
            "prospective_unlocked": False,
            "scientific_role": "consumed_engineering_only",
        },
        "pilot": {
            "path": cohort_locks["fresh_pilot90"]["path"],
            "sha256": cohort_locks["fresh_pilot90"]["sha256"],
            "row_count": 90,
            "per_dataset": 30,
            "gold_access": False,
            "prospective_unlocked": False,
            "scientific_role": "fresh_train_side_family_disjoint_gold_free_pilot",
        },
    }

    protocol_body = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": runner.PROTOCOL_EXPERIMENT_ID,
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "SMOKE_THEN_FRESH_TRAIN_SIDE_PILOT30X3_GOLD_FREE",
        "gold_access": False,
        "answer_scoring": False,
        "prospective_opened_or_hashed": False,
        "training": False,
        "reward_or_loss_change": False,
        "authorization_basis": (
            "Researcher explicitly approved changing the binding rule on 2026-09-04."
        ),
        "research_question": (
            "Does canonical single-hop sub-QA plus deterministic rank-first lexical "
            "binding make answer-conditioned q2 transitions operational on a fresh "
            "three-dataset train-side pilot?"
        ),
        "single_variable_relative_to_v9_phase0": {
            "old": "reject an answer surface present in more than one q1 document",
            "new": "select the smallest retrieval rank among matching q1 documents",
            "held_fixed": [
                "canonical SFT QA prompt with empty KG",
                "strong legacy SFT checkpoint",
                "greedy decoding and tokenizer chat template",
                "v8 q1/q2 controller protocol",
                "canonical Wiki18 E5100+BM25100+RRF100+BGE10 retrieval",
                "root6+q1novel2+q2novel2 evidence budget",
                "all frozen v8 runtime/mechanism gates",
            ],
        },
        "interpretation_boundary": (
            "The fresh run validates the integrated v9.1 chain. Because canonical sub-QA "
            "was previously checked only posthoc on consumed v8 rows, any future outcome "
            "difference must not be described as a binder-only causal effect. Rank-first "
            "binding establishes lexical locality, not semantic entailment."
        ),
        "runtime_contract": runner.runtime_contract_v91(),
        "gates": runner.expected_gates_v91(),
        "run_registry": runner.expected_run_registry_v91(),
        "cohorts": cohort_specs,
        "code_locks": code_locks,
        "cohort_locks": cohort_locks,
        "parent_locks": parent_locks,
        "execution_sequence": [
            "run consumed smoke12 and require every engineering gate",
            "only then run fresh pilot90 and require every Gold-free mechanism gate",
            "retain any failed run append-only",
            "request separate human approval before attaching Gold or computing EM/F1/IHR",
        ],
        "decision": {
            "pilot_all_gold_free_gates_pass": (
                "freeze predictions and request separate authorization for Gold scoring"
            ),
            "any_smoke_or_pilot_gate_fails": (
                "retain failure and stop; do not open Gold or prospective900"
            ),
        },
        "scientific_boundary": (
            "No Gold, EM/F1/IHR, training, PPO, or generalization claim is authorized. "
            "The sealed prospective900 is neither opened nor hashed by this freezer."
        ),
    }
    protocol = _self_commit(protocol_body)

    destination.mkdir(parents=True)
    protocol_path = destination / "protocol.json"
    _write_json_exclusive(protocol_path, protocol)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": runner.PROTOCOL_EXPERIMENT_ID,
        "status": STATUS,
        "gold_access": False,
        "prospective_opened_or_hashed": False,
        "protocol": _file_lock(protocol_path),
    }
    _write_json_exclusive(destination / "manifest.json", manifest)
    return protocol


def main() -> None:
    protocol = freeze()
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "experiment_id": protocol["experiment_id"],
                "output_dir": str((PROJECT_ROOT / runner.PROTOCOL_DIR).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
