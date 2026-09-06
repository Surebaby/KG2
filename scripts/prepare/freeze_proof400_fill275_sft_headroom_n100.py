#!/usr/bin/env python
"""Freeze a qtype-balanced strong-SFT headroom audit from Proof400 fill275.

This is a train-side development gate.  It selects 25 questions per type from
the 275 newly-added automatic ProofKG rows, excluding all 125 safe-hard rows.
It performs no model inference and never starts an optimiser.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from kgproweight.kg.question_kg import question_sha256, validate_question_kg_record
from kgproweight.reward.proofkg_process import is_automatic_proofkg
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import rank, read_jsonl, ref, sha256_file
from scripts.prepare.freeze_mixed_ppo_three_dataset_v2_proof400 import _choose_max_family
from scripts.prepare.resolve_phase3_ppo_runtime_config import resolve_phase3_ppo_runtime_config
from scripts.pilot.audit_proof400_fill275_sft_headroom import software_environment


QTYPES = ("inference", "comparison", "compositional", "bridge_comparison")
N_PER_QTYPE = 25
N = 100
K = 4
EXPERIMENT_ID = "PROOF400-FILL275-STRONG-SFT-HEADROOM-N100-K4-SEED42-V3-PREREGISTRATION"
STATUS = "FROZEN_NOT_RUN_TRAIN_SIDE_DEVELOPMENT_CONSUMED"
DEFAULT_OUT = Path(
    "outputs/audits/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3_preregistration"
)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _stable_payload_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _by_key(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = str(row.get("question_key") or f"{row.get('dataset')}::{row.get('qid')}")
        if not key or key in result:
            raise ValueError(f"{label}: missing/duplicate question key {key!r}")
        result[key] = row
    return result


def _selected_fill275(proof_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fill = [
        dict(row) for row in proof_rows
        if row.get("proof_source") == "automatic_proofkg_2wiki_train_k4_v1"
    ]
    if len(fill) != 275:
        raise ValueError(f"expected exactly 275 expansion rows, got {len(fill)}")
    if any(row.get("route", "").startswith("2wiki_hard_") for row in fill):
        raise ValueError("safe-hard row entered fill275 candidates")
    selected: list[dict[str, Any]] = []
    for qtype in QTYPES:
        candidates = [row for row in fill if row.get("question_type") == qtype]
        chosen = _choose_max_family(
            candidates,
            n=N_PER_QTYPE,
            label=f"strong-sft-headroom-fill275-{qtype}",
        )
        selected.extend(chosen)
    # Interleave types through a fixed seed-42 hash rather than generating in
    # four type blocks.  Selection never consults Gold labels or model output.
    selected.sort(
        key=lambda row: (
            rank("strong-sft-headroom-fill275-order", row["dataset"], row["qid"]),
            row["qid"],
        )
    )
    if len(selected) != N or len({row["question_key"] for row in selected}) != N:
        raise ValueError("headroom cohort is not 100 unique questions")
    if Counter(row["question_type"] for row in selected) != Counter({q: 25 for q in QTYPES}):
        raise ValueError("headroom cohort is not qtype-balanced 25x4")
    return selected


def _assert_formal_policy_contract(t_cfg: Path, tk_cfg: Path) -> dict[str, Any]:
    t = resolve_phase3_ppo_runtime_config(t_cfg)
    tk = resolve_phase3_ppo_runtime_config(tk_cfg)
    fields = (
        "base_model", "sft_checkpoint", "dtype", "seed", "max_new_tokens",
        "temperature", "top_p", "max_input_length", "max_steps",
        "ppo_max_passages", "rollouts_per_prompt", "min_valid_steps",
        "min_reasoning_chars",
    )
    mismatch = [field for field in fields if t.get(field) != tk.get(field)]
    if mismatch:
        raise ValueError(f"formal T/TK policy or generation drift: {mismatch}")
    if int(t["rollouts_per_prompt"]) != K:
        raise ValueError("formal pair is not K4")
    return {field: t[field] for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proof400_protocol",
        type=Path,
        default=Path(
            "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/protocol.json"
        ),
    )
    parser.add_argument(
        "--mixed_data",
        type=Path,
        default=Path("data/silver_data/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42"),
    )
    parser.add_argument(
        "--formal_t_config",
        type=Path,
        default=Path("configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text7200_seed42.yaml"),
    )
    parser.add_argument(
        "--formal_tk_config",
        type=Path,
        default=Path(
            "configs/training/phase3_ppo_mixed3_rearag_v2_proof400_text_kg_v2_1_7200_seed42.yaml"
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite append-only audit: {args.out}")

    proof400_doc = json.loads(args.proof400_protocol.read_text(encoding="utf-8"))
    if proof400_doc.get("status") != "FROZEN_ANSWER_FREE_LEXICAL_FAMILY_V1_NOT_MATERIALIZED_NOT_TRAINED":
        raise ValueError("unexpected Proof400 source protocol")
    proof400_path = Path(proof400_doc["outputs"]["proof400"]["path"])
    if sha256_file(proof400_path) != proof400_doc["outputs"]["proof400"]["sha256"]:
        raise ValueError("Proof400 source hash drift")
    selected = _selected_fill275(read_jsonl(proof400_path))

    source_silver_path = args.mixed_data / "silver_train.jsonl"
    source_qkg_path = args.mixed_data / "question_kg_records.jsonl"
    source_report_path = args.mixed_data / "report.json"
    sources = [source_silver_path, source_qkg_path, source_report_path,
               args.formal_t_config, args.formal_tk_config]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_silver = _by_key(read_jsonl(source_silver_path), "mixed silver")
    source_qkg = _by_key(read_jsonl(source_qkg_path), "mixed question-KG")
    contract = _assert_formal_policy_contract(args.formal_t_config, args.formal_tk_config)
    adapter = Path(str(contract["sft_checkpoint"]))
    base_model = Path("models/llama3-8b")
    base_index = base_model / "model.safetensors.index.json"
    shard_names = sorted(set(json.loads(base_index.read_text(encoding="utf-8"))["weight_map"].values()))
    if len(shard_names) != 4:
        raise ValueError(f"expected four Llama weight shards, got {shard_names}")
    locked_model_files = [
        base_model / "config.json",
        base_model / "generation_config.json",
        base_index,
        *[base_model / name for name in shard_names],
        base_model / "tokenizer.json",
        base_model / "tokenizer_config.json",
        base_model / "special_tokens_map.json",
        adapter.parent / "manifest.json",
        adapter.parent / "sft_loss.jsonl",
        *sorted((path for path in adapter.iterdir() if path.is_file()), key=lambda path: path.name),
    ]
    for path in locked_model_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    cohort_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    qkg_rows: list[dict[str, Any]] = []
    for order, identity in enumerate(selected, start=1):
        key = str(identity["question_key"])
        silver = source_silver.get(key)
        qkg = source_qkg.get(key)
        if silver is None or qkg is None:
            raise ValueError(f"materialized Proof400 join miss: {key}")
        if (
            silver.get("question") != identity["question"]
            or question_sha256(str(silver["question"])) != identity["question_sha256"]
            or silver.get("metadata", {}).get("proof_source")
            != "automatic_proofkg_2wiki_train_k4_v1"
        ):
            raise ValueError(f"source silver identity/provenance mismatch: {key}")
        validate_question_kg_record(qkg)
        if (
            qkg.get("question_key") != key
            or qkg.get("question_sha256") != identity["question_sha256"]
            or qkg.get("kg_subgraph") != silver.get("kg_subgraph")
            or not is_automatic_proofkg(qkg, qkg.get("kg_subgraph") or [])
        ):
            raise ValueError(f"source ProofKG incomplete or drifted: {key}")
        passages = list(silver.get("retrieved_passages") or [])
        kg = list(silver.get("kg_subgraph") or [])
        if len(passages) != 10 or not kg:
            raise ValueError(f"unexpected original context shape: {key}")
        aliases = [str(value).strip() for value in silver.get("metadata", {}).get("gold_answer_aliases", []) if str(value).strip()]
        primary = str(silver.get("metadata", {}).get("gold_answer") or silver.get("answer") or "").strip()
        if not aliases:
            aliases = [primary]
        if not primary or primary not in aliases:
            aliases = [primary, *[value for value in aliases if value != primary]]

        common = {
            "dataset": identity["dataset"],
            "qid": identity["qid"],
            "question_sha256": identity["question_sha256"],
            "family_sha256": identity["family_sha256"],
            "family_version": identity["family_version"],
            "question_type": identity["question_type"],
            "source_role": "proof400_fill275_expansion",
            "train_side_development_consumed": True,
            "evaluation_eligible": False,
        }
        cohort_rows.append({
            "schema_version": "proof400-fill275-sft-headroom-cohort-v1",
            "cohort_order": order,
            "question": identity["question"],
            "gold_access": False,
            **common,
        })
        prompt_rows.append({
            "schema_version": "proof400-fill275-sft-headroom-prompt-input-v1",
            "cohort_order": order,
            "question": identity["question"],
            "retrieved_passages": passages,
            "kg_subgraph": kg,
            "passages_sha256": _stable_payload_hash(passages),
            "kg_subgraph_sha256": _stable_payload_hash(kg),
            "gold_access": False,
            **common,
        })
        label_rows.append({
            "schema_version": "proof400-fill275-sft-headroom-outcome-label-v1",
            "cohort_order": order,
            "gold_answers": aliases,
            "gold_use": "post_generation_train_outcome_scoring_only",
            "source_split": "train",
            **common,
        })
        qkg_rows.append(dict(qkg))

    # Prompt-facing inputs deliberately carry no explicit answer/Gold fields.
    forbidden_prompt_keys = {"answer", "gold_answer", "gold_answers", "supporting_facts", "decomposition", "steps"}
    if any(forbidden_prompt_keys.intersection(row) for row in prompt_rows):
        raise ValueError("Gold/structure label leaked into prompt-facing rows")

    args.out.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "cohort_question_only": args.out / "cohort.question_only.jsonl",
        "prompt_inputs": args.out / "prompt_inputs.gold_free.jsonl",
        "outcome_labels": args.out / "outcome_labels.train_only.jsonl",
        "question_kg_records": args.out / "question_kg_records.jsonl",
    }
    for name, rows in (
        ("cohort_question_only", cohort_rows),
        ("prompt_inputs", prompt_rows),
        ("outcome_labels", label_rows),
        ("question_kg_records", qkg_rows),
    ):
        write_jsonl(output_paths[name], rows)

    runner_path = Path("scripts/pilot/audit_proof400_fill275_sft_headroom.py")
    freeze_path = Path(__file__)
    code_paths = {
        "freeze": freeze_path,
        "runner": runner_path,
        "launcher": Path("launch_proof400_fill275_sft_headroom_n100_local.sh"),
        "prompt_factory": Path("kgproweight/data/prompts.py"),
        "parser": Path("kgproweight/data/parsers.py"),
        "metrics": Path("kgproweight/eval/metrics.py"),
        "question_kg": Path("kgproweight/kg/question_kg.py"),
        "proofkg_process": Path("kgproweight/reward/proofkg_process.py"),
        "validity": Path("kgproweight/training/reward_function.py"),
        "logging": Path("kgproweight/utils/logging.py"),
    }
    protocol = {
        "schema_version": "proof400-fill275-strong-sft-headroom-protocol-v3",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "cohort": {
            "n": N,
            "source_population": "Proof400 expansion fill275 only; safe-hard125 excluded",
            "selection": "25 per qtype; maximize unique family then seed42 SHA256; no Gold/model output consulted",
            "question_type_counts": dict(sorted(Counter(row["question_type"] for row in cohort_rows).items())),
            "unique_families": len({row["family_sha256"] for row in cohort_rows}),
            "train_side_development_consumed": True,
            "evaluation_eligible": False,
        },
        "model": {
            "policy": "strong quota70 SFT; frozen inference only",
            "base_model_path": str(base_model),
            "adapter_path": str(adapter),
            "locked_files": {str(path): ref(path) for path in locked_model_files},
        },
        "generation": {
            "prompt_factory": "build_rl_messages + tokenizer.apply_chat_template(add_generation_prompt=True)",
            "original_passages_and_proofkg": True,
            "top_k_passages": int(contract["ppo_max_passages"]),
            "max_input_length": int(contract["max_input_length"]),
            "max_new_tokens": int(contract["max_new_tokens"]),
            "greedy": {"do_sample": False},
            "sampled": {
                "do_sample": True,
                "rollouts_per_qid": K,
                "temperature": float(contract["temperature"]),
                "top_p": float(contract["top_p"]),
                "top_k": 0,
            },
            "rollouts_per_qid": K,
            "seed": int(contract["seed"]),
            "dtype": str(contract["dtype"]),
            "min_valid_steps": int(contract["min_valid_steps"]),
            "min_reasoning_chars": int(contract["min_reasoning_chars"]),
            "total_generations": N * (K + 1),
        },
        "decision_gates": {
            "sample_valid_rate_min": 0.90,
            "oracle_at_4_minus_greedy_em_min": 0.05,
            "mixed_outcome_qid_rate_min": 0.20,
            "all_must_pass": True,
            "mixed_outcome_definition": "within the four sampled candidates, min(EM) < max(EM)",
        },
        "failure_decision": {
            "allowed": "freeze a new train-side cohort/quota protocol before any formal PPO run",
            "forbidden": "claim that PPO/reward is ineffective, change the gate post hoc, or cherry-pick rows within this frozen result",
            "this_n100_after_run": "development/consumed and never independent confirmation",
        },
        "runtime": {
            "experiment_id": "PROOF400-FILL275-STRONG-SFT-HEADROOM-N100-K4-SEED42-V3",
            "output_dir": "outputs/validation/proof400_fill275_strong_sft_headroom_n100_k4_seed42_v3",
            "optimizer_updates": 0,
            "reward_or_critic_loaded": False,
        },
        "scientific_boundary": {
            "training_started": False,
            "gpu_generation_started": False,
            "zero_optimizer_updates": True,
            "gold_labels_used_only_after_generation": True,
            "prompt_explicit_gold_fields": 0,
            "reward_rankability_tested": False,
            "ppo_effect_tested": False,
            "cross_dataset_claim_allowed": False,
            "formal_ppo_launch_authorized_by_this_freeze_alone": False,
        },
        "software_environment": software_environment(),
        "inputs": {
            "proof400_protocol": ref(args.proof400_protocol),
            "proof400_question_only": ref(proof400_path),
            "mixed_silver": ref(source_silver_path),
            "mixed_question_kg": ref(source_qkg_path),
            "mixed_materialization_report": ref(source_report_path),
            "formal_t_config": ref(args.formal_t_config),
            "formal_tk_config": ref(args.formal_tk_config),
        },
        "outputs": {name: ref(path) for name, path in output_paths.items()},
        "code_closure": {name: ref(path) for name, path in code_paths.items()},
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, status=STATUS, extra={
        "phase": "proof400_fill275_strong_sft_headroom_preregistration",
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "n": N,
        "qtype_counts": protocol["cohort"]["question_type_counts"],
        "gpu_generation_started": False,
        "training_started": False,
    })
    print(json.dumps({
        "status": STATUS,
        "output": str(args.out),
        "cohort": protocol["cohort"],
        "decision_gates": protocol["decision_gates"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
