#!/usr/bin/env python
"""Freeze the v6 dependent-retrieval protocol before any retrieval.

The lock is deliberately expensive and complete: all input/code/model files
and all three Wiki18 assets are content hashed.  This command performs no
retrieval, reads no scorer Gold, and opens no confirmation data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kgproweight.utils.logging import artifact_identity, dump_manifest


SCHEMA_VERSION = "subquestion-dependent-retrieval-preregistration-6"
STATUS = "FROZEN_BEFORE_RETRIEVAL"
SCOPE = "ADAPTIVE_DEVELOPMENT_COMBINATION_SAME60_CONSUMED"
EXPERIMENT_IDS = {
    "materialization": "SUBQUESTION-DEPENDENT-RETRIEVAL-PILOT30X2-SEED42-V6-MATERIALIZE",
    "post_materialization_freeze": "SUBQUESTION-DEPENDENT-RETRIEVAL-PILOT30X2-SEED42-V6-FREEZE",
    "answer_evaluation": "SUBQUESTION-DEPENDENT-RETRIEVAL-PILOT30X2-SEED42-V6-EVAL",
}

EXPECTED_DESIGN_SHA256 = "0bd8a24c5655a4047b4b7831928fa9757f79a781d4461c1794cf1563d3fdd171"
EXPECTED_V4_EVAL_SHA256 = "66a689a19103fc071b95d046154a99453c5421da259d2a3eb54160dc22322530"
EXPECTED_V4_INPUT_SHA256 = "6e6b5313ec8ca3d9e81bebe084837951e173a5277d59102fbacd892d3e2b2a60"
EXPECTED_DOCUMENTS = 21_015_324
DATASETS = {"hotpotqa": 30, "musique": 30}

ANSWER_UTILITY_GATES = {
    "pooled_net_correct_gain_min": 3,
    "pooled_delta_f1_gt": 0.0,
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
    "root_only_injection": 0,
    "duplicate_output_documents": 0,
    "all_dependent_queries_start_with_exact_question": True,
    "max_query_variants_per_logical_hop": 2,
    "all_final_ce_pairs_use_exact_original_question": True,
}

SETTINGS = {
    "network_access": False,
    "datasets_in_order": ["hotpotqa", "musique"],
    "n_per_dataset": 30,
    "rrf_candidate_k": 100,
    "retrieval_query_max_length": 128,
    "step_rerank_topk": 10,
    "max_hops": 4,
    "max_query_variants": 2,
    "bridge_max_docs": 10,
    "bridge_max_hints": 2,
    "bridge_max_body_chars": 1200,
    "protected_originals": 8,
    "candidates_per_query_variant": 2,
    "total_passages": 10,
    "ce_max_chars": 1200,
    "root_hop_injection": False,
    "question_anchor_template": "{original_question}\n{subquery}",
    "no_hint_relation_fallback": True,
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
}

DEFAULT_INPUTS = {
    "cohort": Path("outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/pilot.question_only.jsonl"),
    "retrieval_contexts": Path("outputs/audits/inference_proofkg_v1_n900_seed42_preregistration/retrieval_contexts.jsonl"),
    "musique_plans": Path("outputs/audits/inference_proofkg_v1_pilot30x3_plans_v1/predictions.question_only.jsonl"),
    "hotpot_plans": Path("outputs/audits/inference_proofkg_hotpot_relation_graph_pilot30_v2_plans/predictions.question_only.jsonl"),
}
DEFAULT_CODE = {
    "runner": Path("scripts/pilot/audit_plan_once_dependent_retrieval_v6.py"),
    "query_policy": Path("kgproweight/retrieval/dependent_v6.py"),
    "guarded_merge": Path("kgproweight/retrieval/dependent_merge_v6.py"),
    "v4_loader_dependency": Path("scripts/pilot/audit_plan_once_dependent_retrieval.py"),
    "retriever_builder": Path("scripts/pilot/audit_iterative_bridge_retrieval.py"),
    "reranker": Path("kgproweight/retrieval/reranker.py"),
    "hybrid_retriever": Path("kgproweight/retrieval/hybrid.py"),
    "post_materialization_finalizer": Path("scripts/prepare/finalize_dependent_retrieval_v6.py"),
    "answer_evaluator": Path("scripts/eval/evaluate_dependent_retrieval_pilot.py"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"required non-empty file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def tree_lock(path: Path) -> dict[str, Any]:
    """Hash every file and a canonical inventory digest for a directory."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"required directory is missing: {resolved}")
    files: list[dict[str, Any]] = []
    inventory = hashlib.sha256()
    for child in sorted(value for value in resolved.rglob("*") if value.is_file()):
        relative = child.relative_to(resolved).as_posix()
        digest = sha256_file(child)
        size = child.stat().st_size
        files.append({"path": relative, "size_bytes": size, "sha256": digest})
        inventory.update(relative.encode("utf-8") + b"\0")
        inventory.update(str(size).encode("ascii") + b"\0")
        inventory.update(digest.encode("ascii") + b"\n")
    if not files:
        raise ValueError(f"required directory contains no files: {resolved}")
    return {
        "path": str(resolved),
        "file_count": len(files),
        "size_bytes": sum(int(row["size_bytes"]) for row in files),
        "tree_sha256": inventory.hexdigest(),
        "files": files,
    }


def artifact_lock(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return {"content_lock": file_lock(resolved), "artifact_identity": artifact_identity(resolved)}
    return {"content_lock": tree_lock(resolved), "artifact_identity": artifact_identity(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _expect_hash(lock: Mapping[str, Any], expected: str, label: str) -> None:
    if str(lock.get("sha256") or "") != expected:
        raise ValueError(f"{label} SHA256 drift")


def _validate_design(design: Mapping[str, Any]) -> None:
    if design.get("status") != "RULES_FROZEN_BEFORE_IMPLEMENTATION_AND_RETRIEVAL":
        raise ValueError("v6 design is not frozen before retrieval")
    if design.get("scope") != "ADAPTIVE_DEVELOPMENT_COMBINATION_EXPERIMENT":
        raise ValueError("v6 design scope drifted")
    population = design.get("population") or {}
    if population.get("datasets") != DATASETS or population.get("confirmation_opened") is not False:
        raise ValueError("v6 design population/confirmation boundary drifted")
    gates = design.get("decision_gates") or {}
    if gates.get("materialization") != MATERIALIZATION_GATES:
        raise ValueError("v6 design materialization gates drifted")
    if gates.get("mechanism_unchanged_from_v5") != MECHANISM_GATES:
        raise ValueError("v6 mechanism gates drifted")
    if gates.get("answer_utility_unchanged_from_v4_v5") != ANSWER_UTILITY_GATES:
        raise ValueError("v6 answer utility gates drifted")


def _validate_v4(v4_eval: Mapping[str, Any]) -> None:
    if v4_eval.get("status") != "FROZEN_AFTER_RETRIEVAL_BEFORE_ANSWER_GENERATION":
        raise ValueError("v4 evaluator protocol status drifted")
    if int(v4_eval.get("n", -1)) != 60:
        raise ValueError("v4 evaluator protocol is not same60")
    expected = {
        "pooled_net_correct_gain_min": 3,
        "max_net_correct_loss_per_dataset": 1,
        "parse_count_delta_min": 0,
        "plan_executable_rate_min_each_dataset": 0.8,
        "second_hop_query_nonempty_rate_min_each_dataset": 0.8,
        "new_dependent_candidate_question_rate_min_each_dataset": 0.5,
    }
    if v4_eval.get("decision_gates") != expected:
        raise ValueError("v4 evaluator gates drifted")


def _validate_wiki18_preflight(
    preflight: Mapping[str, Any], corpus: Path, dense: Path, bm25: Path
) -> None:
    if preflight.get("status") != "PASS" or int(preflight.get("expected_docs", -1)) != EXPECTED_DOCUMENTS:
        raise ValueError("Wiki18 preflight is not the frozen PASS/21015324 record")
    counts = preflight.get("counts") or {}
    if any(int(counts.get(name, -1)) != EXPECTED_DOCUMENTS for name in ("corpus", "dense", "bm25")):
        raise ValueError("Wiki18 corpus/index document counts differ")
    actual = preflight.get("paths") or {}
    expected = {"corpus": corpus, "dense": dense, "bm25": bm25}
    for name, path in expected.items():
        if Path(str(actual.get(name) or "")).expanduser().resolve() != path.expanduser().resolve():
            raise ValueError(f"Wiki18 {name} path differs from frozen preflight")


def build_protocol(
    *,
    design_path: Path,
    v4_eval_path: Path,
    v4_input_path: Path,
    wiki18_preflight_path: Path,
    input_paths: Mapping[str, Path],
    code_paths: Mapping[str, Path],
    corpus_path: Path,
    dense_path: Path,
    bm25_path: Path,
    retrieval_encoder_path: Path,
    cross_encoder_path: Path,
    strong_sft_path: Path,
    base_model_path: Path,
    expected_design_sha256: str,
    expected_v4_eval_sha256: str,
    expected_v4_input_sha256: str,
) -> dict[str, Any]:
    if set(input_paths) != set(DEFAULT_INPUTS) or set(code_paths) != set(DEFAULT_CODE):
        raise ValueError("v6 requires the exact frozen input and code lock sets")
    if len(set(EXPERIMENT_IDS.values())) != 3:
        raise ValueError("v6 materialization/freeze/evaluation Experiment IDs must be distinct")

    design_lock, v4_eval_lock, v4_input_lock = (
        file_lock(design_path), file_lock(v4_eval_path), file_lock(v4_input_path)
    )
    _expect_hash(design_lock, expected_design_sha256, "v6 design")
    _expect_hash(v4_eval_lock, expected_v4_eval_sha256, "v4 eval")
    _expect_hash(v4_input_lock, expected_v4_input_sha256, "v4 inputs")
    design, v4_eval, v4_input = read_json(design_path), read_json(v4_eval_path), read_json(v4_input_path)
    _validate_design(design)
    _validate_v4(v4_eval)

    inherited = v4_input.get("inputs") or {}
    inputs = {name: file_lock(path) for name, path in input_paths.items()}
    for name, lock in inputs.items():
        if lock["sha256"] != str((inherited.get(name) or {}).get("sha256") or ""):
            raise ValueError(f"same60 input drifted from frozen v4: {name}")

    preflight_lock = file_lock(wiki18_preflight_path)
    preflight = read_json(wiki18_preflight_path)
    _validate_wiki18_preflight(preflight, corpus_path, dense_path, bm25_path)
    wiki18 = {
        "preflight": preflight_lock,
        "expected_documents": EXPECTED_DOCUMENTS,
        "counts": {name: EXPECTED_DOCUMENTS for name in ("corpus", "dense", "bm25")},
        "corpus": file_lock(corpus_path),
        "dense_index": file_lock(dense_path),
        # Keep the runner-facing shape compatible with artifact_identity.  The
        # complete per-file SHA256 inventory is stored separately below.
        "bm25_index": artifact_identity(bm25_path),
    }
    code = {name: file_lock(path) for name, path in code_paths.items()}
    code["preregistration_freezer"] = file_lock(Path(__file__))

    models = {
        "retrieval_encoder": artifact_identity(retrieval_encoder_path),
        "cross_encoder": artifact_identity(cross_encoder_path),
        "strong_sft": artifact_identity(strong_sft_path),
        "base_model": artifact_identity(base_model_path),
    }
    if models["strong_sft"] != (v4_eval.get("models") or {}).get("strong_sft"):
        raise ValueError("strong SFT differs from frozen v4")
    if models["base_model"] != v4_eval.get("base_model"):
        raise ValueError("base model differs from frozen v4")
    model_content_locks = {
        "retrieval_encoder": artifact_lock(retrieval_encoder_path)["content_lock"],
        "cross_encoder": artifact_lock(cross_encoder_path)["content_lock"],
        "strong_sft": artifact_lock(strong_sft_path)["content_lock"],
        "base_model": artifact_lock(base_model_path)["content_lock"],
    }
    retrieval_asset_content_locks = {
        "corpus": wiki18["corpus"],
        "dense_index": wiki18["dense_index"],
        "bm25_index": tree_lock(bm25_path),
    }

    settings = dict(SETTINGS)
    settings["cross_encoder_model"] = str(cross_encoder_path.expanduser().resolve())
    settings["retrieval_encoder_model"] = str(
        retrieval_encoder_path.expanduser().resolve()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "scope": SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_ids": dict(EXPERIMENT_IDS),
        "population": {
            "datasets": dict(DATASETS),
            "n": 60,
            "identity": "exact v4/v5 consumed same60, no filtering",
            "qid_order_sha256": v4_eval.get("qid_order_sha256"),
            "question_key_order_sha256": v4_eval.get("question_key_order_sha256"),
            "confirmation_opened": False,
        },
        "experimental_design": {
            "type": "ADAPTIVE_DEVELOPMENT_COMBINATION_EXPERIMENT",
            "components_changed_together": [
                "bounded query-hint exploration",
                "exact full-question dependent query anchoring",
                "variant-balanced candidate pooling",
            ],
            "component_attribution_allowed": False,
        },
        "inherits": {
            "v6_design_freeze": design_lock,
            "v4_frozen_eval_protocol": v4_eval_lock,
            "v4_same60_input_protocol": v4_input_lock,
        },
        "inputs": inputs,
        "models": models,
        "model_content_locks": model_content_locks,
        "retrieval_assets": wiki18,
        "retrieval_asset_content_locks": retrieval_asset_content_locks,
        "settings": settings,
        "decision_gates": {
            "materialization": dict(MATERIALIZATION_GATES),
            "mechanism": dict(MECHANISM_GATES),
            "answer_utility": dict(ANSWER_UTILITY_GATES),
        },
        "gold_policy": {
            "retrieval_may_read_gold": False,
            "confirmation_opened": False,
            "attachment": "only after every Gold-free gate and hash lock passes",
        },
        "code": code,
        "required_telemetry": list(design.get("required_telemetry") or []),
        "anti_p_hacking": list(design.get("anti_p_hacking") or []),
        "scientific_boundary": (
            "Consumed development same60 only. A mechanism failure stops before Gold. "
            "A utility pass permits only fresh family/QID-disjoint confirmation with a "
            "retrieval-budget-matched query-expansion arm."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v6_preregistration"))
    parser.add_argument("--design", type=Path, default=Path("outputs/audits/subquestion_dependent_retrieval_v6_design_freeze/protocol.json"))
    parser.add_argument("--v4_eval", type=Path, default=Path("outputs/audits/plan_once_dependent_retrieval_pilot30x2_freeze_v4/protocol.json"))
    parser.add_argument("--v4_inputs", type=Path, default=Path("outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v2_preregistration/protocol.json"))
    parser.add_argument("--wiki18_preflight", type=Path, default=Path("outputs/audits/subquestion_dependent_retrieval_pilot30x2_seed42_v1_preflight/wiki18_assets.json"))
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--corpus", type=Path, default=Path("indexes_wiki18/corpus_flashrag.jsonl"))
    parser.add_argument("--dense", type=Path, default=Path("indexes_wiki18/e5_fp16.dat"))
    parser.add_argument("--bm25", type=Path, default=Path("indexes_wiki18/bm25"))
    parser.add_argument("--retrieval_encoder_path", type=Path, default=Path("models/e5-base-v2"))
    parser.add_argument("--cross_encoder", type=Path, default=Path("models/bge-reranker-v2-m3"))
    parser.add_argument("--strong_sft", type=Path, default=Path("checkpoints/sft_legacy_repaired_v2_quota70_hard_seed42_no_text_head/final"))
    parser.add_argument("--base_model", type=Path, default=Path("models/llama3-8b"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v6 preregistration: {args.out}")
    protocol = build_protocol(
        design_path=args.design, v4_eval_path=args.v4_eval, v4_input_path=args.v4_inputs,
        wiki18_preflight_path=args.wiki18_preflight,
        input_paths={name: getattr(args, name) for name in DEFAULT_INPUTS},
        code_paths=dict(DEFAULT_CODE), corpus_path=args.corpus, dense_path=args.dense,
        bm25_path=args.bm25, retrieval_encoder_path=args.retrieval_encoder_path,
        cross_encoder_path=args.cross_encoder,
        strong_sft_path=args.strong_sft, base_model_path=args.base_model,
        expected_design_sha256=EXPECTED_DESIGN_SHA256,
        expected_v4_eval_sha256=EXPECTED_V4_EVAL_SHA256,
        expected_v4_input_sha256=EXPECTED_V4_INPUT_SHA256,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(protocol_path)
    dump_manifest(args.out, status=STATUS, extra={
        "phase": "dependent_retrieval_v6_preregistration", "scope": SCOPE,
        "experiment_ids": EXPERIMENT_IDS,
        "protocol": {"path": str(protocol_path.resolve()), "sha256": digest},
        "gold_access": False, "confirmation_opened": False,
    })
    print(json.dumps({"status": STATUS, "protocol": str(protocol_path.resolve()), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
