#!/usr/bin/env python
"""Freeze the formal v5 dependent-retrieval materialisation protocol.

This is a *pre-retrieval* lock.  It does not build passages, read scorer Gold,
generate answers, or evaluate utility.  It binds the already frozen v5 design
to the exact consumed same-60 inputs, local Wiki18 settings, code files, model
artifacts, unchanged v4 utility gates, and three distinct downstream
Experiment IDs.

The v5 design intentionally changes two coupled components (typed bridge
admission and guarded passage merging), so the resulting run is registered as
an adaptive development combination experiment.  It cannot isolate either
component without a later fresh 2x2 ablation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kgproweight.utils.logging import artifact_identity, dump_manifest


SCHEMA_VERSION = "subquestion-dependent-retrieval-preregistration-5"
STATUS = "FROZEN_BEFORE_RETRIEVAL"
SCOPE = "ADAPTIVE_DEVELOPMENT_COMBINATION_SAME60_CONSUMED"

EXPERIMENT_IDS = {
    "materialization": "SUBQUESTION-DEPENDENT-RETRIEVAL-PILOT30X2-SEED42-V5-MATERIALIZE",
    "post_materialization_freeze": "SUBQUESTION-DEPENDENT-RETRIEVAL-PILOT30X2-SEED42-V5-FREEZE",
    "answer_evaluation": "SUBQUESTION-DEPENDENT-RETRIEVAL-PILOT30X2-SEED42-V5-EVAL",
}

EXPECTED_DESIGN_SHA256 = "dab90b6204c8e86fdb31bed958c3d7b597fb5500284080a17a05430a6cca9acb"
EXPECTED_V4_EVAL_SHA256 = "66a689a19103fc071b95d046154a99453c5421da259d2a3eb54160dc22322530"
EXPECTED_V4_INPUT_PROTOCOL_SHA256 = (
    "6e6b5313ec8ca3d9e81bebe084837951e173a5277d59102fbacd892d3e2b2a60"
)

DATASETS = {"hotpotqa": 30, "musique": 30}
EXPECTED_DOCUMENTS = 21_015_324
ANSWER_UTILITY_GATES = {
    "pooled_net_correct_gain_min": 3,
    "pooled_delta_f1_gt": 0.0,
    "max_net_correct_loss_per_dataset": 1,
    "parse_count_delta_min": 0,
}
V4_EVAL_PROTOCOL_GATE_SUBSET = {
    "pooled_net_correct_gain_min": 3,
    "max_net_correct_loss_per_dataset": 1,
    "parse_count_delta_min": 0,
}
MECHANISM_GATES = {
    "plan_executable_rate_min_each_dataset": 0.80,
    "dependent_hop_query_nonempty_rate_min_each_dataset": 0.80,
    "retained_new_dependent_document_question_rate_min_each_dataset": 0.50,
}
MATERIALIZATION_GATES = {
    "runtime_errors": 0,
    "fallback_execution_error": 0,
    "identity_join_rate": 1.0,
    "fallback_exact": True,
    "all_rows_top10": True,
    "a_prefix8_exact_when_changed": True,
    "unauthorized_displacement": 0,
}

DEFAULT_INPUTS = {
    "cohort": Path(
        "outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/"
        "pilot.question_only.jsonl"
    ),
    "retrieval_contexts": Path(
        "outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/"
        "retrieval_contexts.jsonl"
    ),
    "musique_plans": Path(
        "outputs/audits/inference_proofkg_v1_pilot30x3_plans_v1/"
        "predictions.question_only.jsonl"
    ),
    "hotpot_plans": Path(
        "outputs/audits/inference_proofkg_hotpot_relation_graph_pilot30_v2_plans/"
        "predictions.question_only.jsonl"
    ),
}

DEFAULT_CODE = {
    "runner": Path("scripts/pilot/audit_plan_once_dependent_retrieval_v5.py"),
    "typed_bridge_selector": Path("kgproweight/retrieval/dependent_v5.py"),
    "guarded_merge": Path("kgproweight/retrieval/dependent_merge_v5.py"),
    "v4_loader_dependency": Path("scripts/pilot/audit_plan_once_dependent_retrieval.py"),
    "retriever_builder": Path("scripts/pilot/audit_iterative_bridge_retrieval.py"),
    "reranker": Path("kgproweight/retrieval/reranker.py"),
    "post_materialization_finalizer": Path(
        "scripts/prepare/finalize_dependent_retrieval_v5.py"
    ),
    "answer_evaluator": Path("scripts/eval/evaluate_dependent_retrieval_pilot.py"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _file_lock(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _expect_hash(lock: Mapping[str, Any], expected: str, *, label: str) -> None:
    actual = str(lock.get("sha256") or "")
    if actual != expected:
        raise ValueError(f"{label} SHA256 drift: expected {expected}, got {actual}")


def _protocol_input_locks(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = protocol.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("frozen v4 input protocol has no inputs mapping")
    locks: dict[str, dict[str, Any]] = {}
    for name in DEFAULT_INPUTS:
        item = inputs.get(name)
        if not isinstance(item, Mapping) or not str(item.get("sha256") or ""):
            raise ValueError(f"frozen v4 input protocol has no hash for {name}")
        locks[name] = dict(item)
    return locks


def _validate_design(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "RULES_FROZEN_BEFORE_IMPLEMENTATION_AND_RETRIEVAL":
        raise ValueError("v5 design is not frozen before implementation/retrieval")
    if protocol.get("scope") != "ADAPTIVE_DEVELOPMENT_COMBINATION_EXPERIMENT":
        raise ValueError("v5 design does not declare the required combination scope")
    population = protocol.get("population")
    if not isinstance(population, Mapping) or population.get("datasets") != DATASETS:
        raise ValueError("v5 design population differs from HotpotQA30 + MuSiQue30")
    if population.get("confirmation_opened") is not False:
        raise ValueError("v5 design unexpectedly opens confirmation")
    gates = ((protocol.get("decision_gates") or {}).get("answer_utility_unchanged_from_v4"))
    if gates != ANSWER_UTILITY_GATES:
        raise ValueError(f"v5 design utility gates drifted: {gates!r}")


def _validate_v4_eval_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("status") != "FROZEN_AFTER_RETRIEVAL_BEFORE_ANSWER_GENERATION":
        raise ValueError("v4 evaluation protocol is not a frozen pre-answer protocol")
    if int(protocol.get("n", -1)) != 60:
        raise ValueError("v4 evaluation protocol is not the frozen same-60 population")
    gates = protocol.get("decision_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("v4 evaluation protocol has no decision_gates")
    for name, expected in V4_EVAL_PROTOCOL_GATE_SUBSET.items():
        if gates.get(name) != expected:
            raise ValueError(
                f"v4 frozen evaluation gate drifted: {name}={gates.get(name)!r}"
            )


def _validate_wiki18_preflight(
    protocol: Mapping[str, Any],
    *,
    corpus: Path,
    dense: Path,
    bm25: Path,
) -> None:
    if protocol.get("status") != "PASS":
        raise ValueError("Wiki18 frozen preflight status is not PASS")
    if int(protocol.get("expected_docs", -1)) != EXPECTED_DOCUMENTS:
        raise ValueError("Wiki18 preflight expected document count drifted")
    counts = protocol.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(name, -1)) != EXPECTED_DOCUMENTS
        for name in ("corpus", "dense", "bm25")
    ):
        raise ValueError("Wiki18 corpus/dense/BM25 counts are not aligned")
    paths = protocol.get("paths")
    expected_paths = {
        "corpus": corpus.expanduser().resolve(),
        "dense": dense.expanduser().resolve(),
        "bm25": bm25.expanduser().resolve(),
    }
    if not isinstance(paths, Mapping):
        raise ValueError("Wiki18 preflight has no paths mapping")
    for name, expected in expected_paths.items():
        if Path(str(paths.get(name) or "")).expanduser().resolve() != expected:
            raise ValueError(f"Wiki18 {name} path differs from its frozen preflight")
        if not expected.exists():
            raise FileNotFoundError(expected)


def _same_artifact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare model identities without making an absolute path the identity."""

    keys = ("exists", "kind", "size_bytes", "md5", "files", "inventory_sha256")
    return all(left.get(key) == right.get(key) for key in keys)


def build_protocol(
    *,
    design_protocol_path: Path,
    v4_eval_protocol_path: Path,
    v4_input_protocol_path: Path,
    wiki18_preflight_path: Path,
    input_paths: Mapping[str, Path],
    code_paths: Mapping[str, Path],
    corpus_path: Path,
    dense_index_path: Path,
    bm25_index_path: Path,
    cross_encoder_path: Path,
    adapter_path: Path,
    base_model_path: Path,
    expected_design_sha256: str,
    expected_v4_eval_sha256: str,
    expected_v4_input_protocol_sha256: str,
) -> dict[str, Any]:
    """Validate the inherited locks and return the complete v5 protocol."""

    if set(input_paths) != set(DEFAULT_INPUTS):
        raise ValueError(f"exactly four input locks are required: {sorted(DEFAULT_INPUTS)}")
    if set(code_paths) != set(DEFAULT_CODE):
        raise ValueError(f"core code lock set differs: {sorted(code_paths)}")
    if len(set(EXPERIMENT_IDS.values())) != len(EXPERIMENT_IDS):
        raise ValueError("v5 downstream Experiment IDs must be pairwise distinct")

    design_lock = _file_lock(design_protocol_path)
    v4_eval_lock = _file_lock(v4_eval_protocol_path)
    v4_input_lock = _file_lock(v4_input_protocol_path)
    wiki18_preflight_lock = _file_lock(wiki18_preflight_path)
    _expect_hash(design_lock, expected_design_sha256, label="v5 design freeze")
    _expect_hash(v4_eval_lock, expected_v4_eval_sha256, label="v4 frozen eval protocol")
    _expect_hash(
        v4_input_lock,
        expected_v4_input_protocol_sha256,
        label="v4 frozen input protocol",
    )

    design = _read_json(design_protocol_path)
    v4_eval = _read_json(v4_eval_protocol_path)
    v4_input = _read_json(v4_input_protocol_path)
    wiki18_preflight = _read_json(wiki18_preflight_path)
    _validate_design(design)
    _validate_v4_eval_protocol(v4_eval)
    _validate_wiki18_preflight(
        wiki18_preflight,
        corpus=corpus_path,
        dense=dense_index_path,
        bm25=bm25_index_path,
    )

    inherited_input_locks = _protocol_input_locks(v4_input)
    inputs = {name: _file_lock(path) for name, path in input_paths.items()}
    for name, lock in inputs.items():
        inherited_sha = str(inherited_input_locks[name]["sha256"])
        if lock["sha256"] != inherited_sha:
            raise ValueError(
                f"same-60 input {name} drifted from the frozen v4 input protocol"
            )

    code = {name: _file_lock(path) for name, path in code_paths.items()}
    freeze_script_lock = _file_lock(Path(__file__))

    cross_encoder_identity = artifact_identity(cross_encoder_path)
    adapter_identity = artifact_identity(adapter_path)
    base_model_identity = artifact_identity(base_model_path)
    for name, identity in {
        "cross_encoder": cross_encoder_identity,
        "strong_sft": adapter_identity,
        "base_model": base_model_identity,
    }.items():
        if identity.get("exists") is not True:
            raise FileNotFoundError(f"{name} artifact is missing: {identity.get('path')}")

    frozen_v4_sft = ((v4_eval.get("models") or {}).get("strong_sft"))
    frozen_v4_base = v4_eval.get("base_model")
    if not isinstance(frozen_v4_sft, Mapping) or not _same_artifact(
        adapter_identity, frozen_v4_sft
    ):
        raise ValueError("strong-SFT artifact differs from the frozen v4 evaluation")
    if not isinstance(frozen_v4_base, Mapping) or not _same_artifact(
        base_model_identity, frozen_v4_base
    ):
        raise ValueError("base-model artifact differs from the frozen v4 evaluation")

    corpus = corpus_path.expanduser().resolve()
    dense = dense_index_path.expanduser().resolve()
    bm25 = bm25_index_path.expanduser().resolve()
    retrieval_assets = {
        "wiki18_preflight": wiki18_preflight_lock,
        "expected_documents": EXPECTED_DOCUMENTS,
        "counts": {name: EXPECTED_DOCUMENTS for name in ("corpus", "dense", "bm25")},
        # Hashing 46 GB of immutable Wiki18 assets again is deliberately avoided;
        # the prior frozen alignment report is content-locked above, while path
        # and byte size guard against accidental asset substitution here.
        "corpus": {"path": str(corpus), "size_bytes": corpus.stat().st_size},
        "dense_index": {"path": str(dense), "size_bytes": dense.stat().st_size},
        "bm25_index": artifact_identity(bm25),
    }

    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "experiment_ids": dict(EXPERIMENT_IDS),
        "research_variable": {
            "type": "ADAPTIVE_DEVELOPMENT_COMBINATION_EXPERIMENT",
            "components_changed_together": [
                "typed_relation_object_v1 bridge admission",
                "dependent_only_a8_union_ce_v1 guarded merge",
            ],
            "component_attribution_allowed": False,
            "later_attribution_requirement": (
                "fresh family/QID-disjoint 2x2 and retrieval-budget-matched ablation"
            ),
        },
        "population": {
            "datasets": dict(DATASETS),
            "n": 60,
            "identity": "exact v4 consumed same-60 qids/questions/order/plans",
            "filtering": "none",
            "confirmation_opened": False,
            "qid_order_sha256": v4_eval.get("qid_order_sha256"),
            "question_key_order_sha256": v4_eval.get("question_key_order_sha256"),
        },
        "inherits": {
            "v5_design_freeze": design_lock,
            "v4_frozen_evaluation_protocol": v4_eval_lock,
            "v4_same60_input_protocol": v4_input_lock,
            "inherited_without_change": [
                "same-60 identity and order",
                "legacy KG bytes",
                "strong SFT and base model",
                "prompt schema and greedy decoding",
                "answer-utility gates",
                "Gold separation",
            ],
        },
        "inputs": inputs,
        "retrieval_assets": retrieval_assets,
        "models": {
            "cross_encoder": cross_encoder_identity,
            "strong_sft": adapter_identity,
            "base_model": base_model_identity,
        },
        "settings": {
            "network_access": False,
            "datasets_in_order": ["hotpotqa", "musique"],
            "n_per_dataset": 30,
            "rrf_candidate_k": 100,
            "step_rerank_topk": 10,
            "cross_encoder_model": str(cross_encoder_path.expanduser().resolve()),
            "max_hops": 4,
            "max_query_variants": 2,
            "bridge_max_docs": 10,
            "bridge_max_candidates": 2,
            "bridge_max_body_chars": 1200,
            "protected_originals": 8,
            "candidates_per_dependent_hop": 2,
            "total_passages": 10,
            "ce_max_chars": 1200,
            "root_hop_injection": False,
            "full_question_union_scoring": "same frozen cross encoder; ties retain Arm A",
            "generation": {
                "seed": 42,
                "decode": "greedy",
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "max_new_tokens": 512,
                "top_k_passages": 10,
            },
        },
        "decision_gates": {
            "materialization": dict(MATERIALIZATION_GATES),
            "mechanism": dict(MECHANISM_GATES),
            "answer_utility_unchanged_from_v4": dict(ANSWER_UTILITY_GATES),
        },
        "gold_policy": {
            "retrieval_may_read_gold": False,
            "forbidden": [
                "answer", "golden_answers", "supporting_facts",
                "question_decomposition", "evidences", "target",
            ],
            "attachment": (
                "only after Gold-free Arm A/v5 passages and report are frozen and hashed"
            ),
        },
        "code": {**code, "preregistration_freezer": freeze_script_lock},
        "anti_p_hacking": list(design.get("anti_p_hacking") or []),
        "scientific_boundary": (
            "Consumed development same-60 only. A pass can authorize only a fresh "
            "family/QID-disjoint confirmation and cannot establish component-level "
            "causality or a paper result. A failed gate is retained append-only."
        ),
    }
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/audits/"
            "subquestion_dependent_retrieval_pilot30x2_seed42_v5_preregistration"
        ),
    )
    parser.add_argument(
        "--design_protocol",
        type=Path,
        default=Path("outputs/audits/subquestion_dependent_retrieval_v5_design_freeze/protocol.json"),
    )
    parser.add_argument(
        "--v4_eval_protocol",
        type=Path,
        default=Path("outputs/audits/plan_once_dependent_retrieval_pilot30x2_freeze_v4/protocol.json"),
    )
    parser.add_argument(
        "--v4_input_protocol",
        type=Path,
        default=Path(
            "outputs/audits/"
            "subquestion_dependent_retrieval_pilot30x2_seed42_v2_preregistration/"
            "protocol.json"
        ),
    )
    parser.add_argument(
        "--wiki18_preflight",
        type=Path,
        default=Path(
            "outputs/audits/"
            "subquestion_dependent_retrieval_pilot30x2_seed42_v1_preflight/"
            "wiki18_assets.json"
        ),
    )
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--corpus_path", type=Path, default=Path("indexes_wiki18/corpus_flashrag.jsonl"))
    parser.add_argument("--dense_index_path", type=Path, default=Path("indexes_wiki18/e5_fp16.dat"))
    parser.add_argument("--bm25_index_path", type=Path, default=Path("indexes_wiki18/bm25"))
    parser.add_argument("--cross_encoder", type=Path, default=Path("models/bge-reranker-v2-m3"))
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"),
    )
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v5 preregistration: {args.out}")
    protocol = build_protocol(
        design_protocol_path=args.design_protocol,
        v4_eval_protocol_path=args.v4_eval_protocol,
        v4_input_protocol_path=args.v4_input_protocol,
        wiki18_preflight_path=args.wiki18_preflight,
        input_paths={name: getattr(args, name) for name in DEFAULT_INPUTS},
        code_paths=dict(DEFAULT_CODE),
        corpus_path=args.corpus_path,
        dense_index_path=args.dense_index_path,
        bm25_index_path=args.bm25_index_path,
        cross_encoder_path=args.cross_encoder,
        adapter_path=args.adapter,
        base_model_path=args.base_model,
        expected_design_sha256=EXPECTED_DESIGN_SHA256,
        expected_v4_eval_sha256=EXPECTED_V4_EVAL_SHA256,
        expected_v4_input_protocol_sha256=EXPECTED_V4_INPUT_PROTOCOL_SHA256,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    protocol_sha256 = _sha256_file(protocol_path)
    dump_manifest(
        args.out,
        status=STATUS,
        extra={
            "phase": "dependent_retrieval_v5_preregistration",
            "scope": SCOPE,
            "experiment_ids": dict(EXPERIMENT_IDS),
            "protocol": {"path": str(protocol_path.resolve()), "sha256": protocol_sha256},
            "gold_access": False,
            "confirmation_opened": False,
        },
    )
    print(json.dumps({
        "status": STATUS,
        "scope": SCOPE,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "experiment_ids": EXPERIMENT_IDS,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
