"""Offline source evidence/mask for a frozen 96-graph confirmation proposal.

No candidate generation, Gold, network, cache writes, fit or PPO clearance.
The old source verifier and all original training artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kgproweight.kg.versioned_evidence_store import VersionedEvidenceStore
from kgproweight.kg.historical_wikidata_retriever import HistoricalWikidataPropertyRetriever
from kgproweight.kg.store_first_combined_retriever import StoreFirstCombinedRetriever
from kgproweight.reward.source_integrity_v1 import validate_source_integrity_v1
from kgproweight.reward.source_credit_gate_v1 import FrozenSourceCreditMask, MASK_SCHEMA, MASK_VERSION
from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "source-credit-v2-fresh-confirmation-source-preparation-v1"
TYPES = {"bridge_comparison", "comparison", "compositional"}
PARENT_EVIDENCE = ROOT / "outputs/audits/sourcegate_source_disagreement_review_20260905_v1/integrity_clearance_v2/qid_source_evidence.json"
CACHE = ROOT / "data/derived/2wiki_proofkg_official_raw_v2_candidate_pool_n1500_clean_closure_v3/closure_historical_property_cache.jsonl"
ALIASES = ROOT / "data/external/2wikimultihopqa_official_ids/data_ids/id_aliases.json"
STORE = ROOT / "indexes/versioned_2wiki_evidence_store_v6_mixed3_v4_complete_ledger_seed42"
WRAPPER_ALLOWED = {"experiment_id", "payload_sha256", "source_credit_mask", "training_clearance",
    "independent_confirmation_clearance", "ppo_launch_clearance", "confirmation_scope", "confirmation_parent_artifact",
    "confirmation_input_bindings"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def bound(binding, base=ROOT):
    path = Path(binding["path"])
    candidates = [path] if path.is_absolute() else [base / path, ROOT / path]
    for path in candidates:
        if path.is_file() and sha(path) == binding["sha256"]:
            return path.resolve()
    raise ValueError("bound source file missing or changed: " + str(binding["path"]))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_rows(path, data):
    with path.open("x") as stream:
        for row in data:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def require_gold_free(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"gold", "gold_answer", "gold_answers", "golden_answers", "gold_answer_aliases", "gold_target"}:
                raise ValueError("Gold fields are forbidden in source preparation")
            require_gold_free(item)
    elif isinstance(value, list):
        for item in value:
            require_gold_free(item)


def select_graph_inputs(cohort, inputs):
    require_gold_free(cohort)
    require_gold_free(inputs)
    chosen = {r["question_key"]: r for r in cohort if r["proposal_role"] == "graph"}
    all_inputs = {r["question_key"]: r for r in inputs}
    if len(cohort) != 132 or len(inputs) != 132 or len(all_inputs) != 132 or len(chosen) != 96:
        raise ValueError("frozen 132-input/96-graph population mismatch")
    if set(all_inputs) != {r["question_key"] for r in cohort}:
        raise ValueError("input/cohort identities differ")
    if Counter(r["question_type"] for r in chosen.values()) != Counter({name: 32 for name in TYPES}):
        raise ValueError("three-type graph quota changed")
    selected = []
    for key, row in chosen.items():
        actual = all_inputs[key]
        for field in ("dataset", "qid", "question", "question_sha256", "family_sha256"):
            if row[field] != actual[field]:
                raise ValueError("input identity differs from frozen cohort")
        if actual["m_graph"] != 1 or actual["source_record_sha256"] != canonical_sha256(actual["fullsource_record"]):
            raise ValueError("original graph eligibility or record digest mismatch")
        if actual["input_sha256"] != canonical_sha256({k: v for k, v in actual.items() if k != "input_sha256"}):
            raise ValueError("input digest does not reproduce")
        selected.append(actual)
    return selected


def make_wrapper(parent, mask_binding, provenance, variant):
    data = deepcopy(parent)
    data.pop("payload_sha256")
    data.update(experiment_id=parent["experiment_id"] + "-FRESH-CONFIRMATION-SOURCE-ONLY-V1",
        source_credit_mask=deepcopy(mask_binding), training_clearance=False,
        independent_confirmation_clearance=False, ppo_launch_clearance=False,
        confirmation_scope="fresh_three_type_source_preparation_only_no_candidates_no_gold_no_training_clearance",
        confirmation_parent_artifact=provenance["parent_gate"], confirmation_input_bindings=deepcopy(provenance))
    data["payload_sha256"] = canonical_sha256(data)
    if {k: v for k, v in data.items() if k not in WRAPPER_ALLOWED} != {
            k: v for k, v in parent.items() if k not in WRAPPER_ALLOWED}:
        raise ValueError("confirmation wrapper changed frozen numerical/research fields")
    return data


def construct_evidence(inputs, parent_source_bindings):
    official = {r["Q_id"]: r for r in read_rows(ALIASES)}
    historical = {r["qid"]: r for r in read_rows(CACHE) if r.get("entity")}
    bind = lambda path: {str(path.relative_to(ROOT)): parent_source_bindings[str(path.relative_to(ROOT))]}
    names_binding, hist_binding = bind(ALIASES), bind(CACHE)
    store_binding = {**bind(STORE / "edges.jsonl"), **bind(STORE / "aliases.jsonl")}
    implementation_binding = {name: digest for name, digest in parent_source_bindings.items() if name.endswith(".py")}
    store = VersionedEvidenceStore(STORE)
    historical_backend = HistoricalWikidataPropertyRetriever(cache_path=CACHE, cutoff="2020-12-09T23:59:59Z",
                                                             offline=True, label_resolver=store)
    retriever = StoreFirstCombinedRetriever(store, historical_backend)
    requests, qids = defaultdict(set), set()
    for row in inputs:
        execution = row["fullsource_record"].get("execution") or {}
        qids.update(entity["qid"] for entity in execution.get("anchor_entities", {}).values() if entity.get("qid"))
        for hop in execution.get("hops", []):
            for entity in hop.get("input_entities", []):
                if entity.get("qid"):
                    qids.add(entity["qid"])
                    requests[entity["qid"]].update(hop.get("pids", []))
    replayed = {}
    # Tripwires prove offline cache misses cannot silently request or persist.
    with patch.object(historical_backend, "_request_entity", side_effect=AssertionError("network forbidden")), \
         patch.object(historical_backend, "_persist", side_effect=AssertionError("cache writes forbidden")):
        for qid, pids in sorted(requests.items()):
            replayed[qid] = retriever.fetch_edges(qid, sorted(pids))
            for edge in replayed[qid]:
                qids.add(edge["head_qid"])
                if edge.get("tail_qid"):
                    qids.add(edge["tail_qid"])
                edge["bindings"] = {**store_binding, **implementation_binding,
                                    **(hist_binding if edge["source"] == "historical_fallback" else {})}
    entities = {}
    for qid in sorted(qids):
        official_row, historical_row = official.get(qid, {}), historical.get(qid, {})
        entity = historical_row.get("entity") or {}
        label = (entity.get("labels") or {}).get("en", {}).get("value")
        aliases = [item["value"] for item in (entity.get("aliases") or {}).get("en", [])]
        entities[qid] = {"labels": [label] if label else [],
            "aliases": sorted(set(official_row.get("aliases", []) + aliases)),
            "demonyms": official_row.get("demonyms", []),
            "bindings": {**(names_binding if official_row else {}), **(hist_binding if historical_row else {})},
            "typed_edges": replayed.get(qid, [])}
    return {"schema_version": "qid-source-evidence-v1", "bindings": dict(parent_source_bindings),
            "entities": entities, "gold_used": False,
            "provenance_note": "Same frozen offline n1500 cache/store and source checker; QID names only from official aliases/demonyms and historical labels, never store votes."}


def run(input_manifest_path, cohort_manifest_path, calibration_manifest_path, output):
    if output.exists():
        raise FileExistsError("source preparation refuses to overwrite prior results")
    observed = {}
    def track(path):
        value = identity(path)
        observed[value["path"]] = value
        return Path(value["path"])
    def check(value, base=ROOT):
        return track(bound(value, base))
    input_manifest = json.loads(track(input_manifest_path).read_text())
    cohort_manifest = json.loads(track(cohort_manifest_path).read_text())
    calibration_manifest = json.loads(track(calibration_manifest_path).read_text())
    input_path = check(input_manifest["outputs"]["inputs.jsonl"], input_manifest_path.parent)
    cohort_path = check(cohort_manifest["outputs"]["candidate_cohort.question_only.jsonl"], cohort_manifest_path.parent)
    inputs = select_graph_inputs(read_rows(cohort_path), read_rows(input_path))
    parent_mask_path = check(calibration_manifest["source_credit_mask"])
    parent_mask = json.loads(parent_mask_path.read_text())
    evidence_path = check(parent_mask["source_evidence"])
    if evidence_path != PARENT_EVIDENCE.resolve():
        raise ValueError("unexpected original bound source evidence")
    parent_evidence = json.loads(evidence_path.read_text())
    source_bindings = parent_evidence["bindings"]
    for name, digest in source_bindings.items():
        check({"path": name, "sha256": digest})
    check(parent_mask["verifier_code"])
    track(Path(__file__))
    track(ROOT / "kgproweight/reward/source_credit_gate_v1.py")
    track(ROOT / "kgproweight/reward/source_credit_gate_v2.py")
    parent_gates = {variant: check(calibration_manifest["outputs"][variant]["gate.json"])
                    for variant in ("norm_only", "features_v2")}
    output.mkdir(parents=True)
    evidence = construct_evidence(inputs, source_bindings)
    checks = [{**validate_source_integrity_v1(row["fullsource_record"], evidence),
               "question_key": row["question_key"], "original_m_graph": row["m_graph"], "input_sha256": row["input_sha256"]}
              for row in inputs]
    write_rows(output / "graph_inputs.jsonl", inputs)
    write_json(output / "qid_source_evidence.json", evidence)
    write_rows(output / "question_checks.jsonl", checks)
    mask_data = {"schema_version": MASK_SCHEMA, "mask_version": MASK_VERSION,
        "experiment_id": "SOURCE-CREDIT-V2-FRESH-CONFIRMATION-MASK-20260906-V1",
        "inputs": identity(output / "graph_inputs.jsonl"), "question_checks": identity(output / "question_checks.jsonl"),
        "source_evidence": identity(output / "qid_source_evidence.json"), "verifier_code": parent_mask["verifier_code"],
        "scope": "reward_credit_only_input_unchanged", "confirmation_only": True,
        "source_integrity_clearance": False, "gold_used": False,
        "source_credit_rule": "Original m_graph AND exact frozen record identity AND unchanged source verifier PASS; keep every FAIL/UNVERIFIED with zero credit."}
    mask_data["payload_sha256"] = canonical_sha256(mask_data)
    write_json(output / "mask_manifest.json", mask_data)
    mask = FrozenSourceCreditMask.load(output / "mask_manifest.json")
    mask_binding = {**identity(output / "mask_manifest.json"), "payload_sha256": mask.payload_sha256}
    wrapper_checks = {}
    by_type = defaultdict(Counter)
    type_index = {row["question_key"]: row["question_type"] for row in read_rows(cohort_path)}
    for check_row in checks:
        by_type[type_index[check_row["question_key"]]][check_row["status"]] += 1
    for variant, parent_path in parent_gates.items():
        parent = json.loads(parent_path.read_text())
        provenance = {"parent_gate": identity(parent_path), "inputs_manifest": identity(input_manifest_path),
                      "cohort_manifest": identity(cohort_manifest_path), "source_mask": mask_binding}
        wrapper = make_wrapper(parent, mask_binding, provenance, variant)
        folder = output / variant
        folder.mkdir()
        write_json(folder / "gate.json", wrapper)
        gate = SourceCreditGateV2.load(folder / "gate.json", allow_unvalidated=True)
        try:
            SourceCreditGateV2(wrapper, mask=gate.mask)
        except ValueError as exc:
            if "fresh confirmation" not in str(exc):
                raise
        else:
            raise AssertionError("source-only wrapper accepted for production")
        decisions = []
        for row, checked_row in zip(inputs, checks):
            spec = SimpleNamespace(**deepcopy(row["spec"]))
            features = gate.mask_features(spec, gate.compute_features(spec, [], {}))
            marker = features["source_credit_mask"]
            if marker["status"] != checked_row["status"] or marker["status"] == "MISSING":
                raise ValueError("new confirmation identity mask did not reproduce source decision")
            if features["m_graph"] != int(checked_row["clearance"]):
                raise ValueError("non-PASS acquired Graph credit or PASS lost original eligibility")
            alpha = gate.predict(features)
            if features["m_graph"] == 0 and alpha != 0:
                raise ValueError("masked new question acquired nonzero alpha")
            decisions.append({"question_key": row["question_key"], "status": marker["status"],
                              "m_graph": features["m_graph"], "empty_trajectory_diagnostic_alpha": alpha})
        write_rows(folder / "identity_mask_checks.jsonl", decisions)
        wrapper_checks[variant] = {"frozen_fields_exactly_unchanged": True, "identity_count": len(decisions),
            "missing": 0, "source_decision_mismatches": 0, "production_load_rejected": True,
            "candidate_generated": False, "alpha_scope": "empty trajectory mask wiring only, not candidate reward or gate utility"}
    for path, binding in observed.items():
        if identity(path) != binding:
            raise ValueError("source or code changed during offline preparation")
    report = {"schema_version": SCHEMA, "experiment_id": "SOURCE-CREDIT-V2-FRESH-CONFIRMATION-SOURCE-20260906-V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "SOURCE_PREPARED_NOT_CONFIRMATION_RESULT_NOT_PPO_CLEARANCE",
        "questions": len(inputs), "source_status_counts": dict(Counter(row["status"] for row in checks)),
        "by_graph_type": {key: dict(value) for key, value in by_type.items()}, "inference_type_covered": False,
        "replacement_count": 0, "cache_or_network_writes": 0, "network_used": False,
        "gold_access": False, "gpu_used": False, "candidate_generation_started": False,
        "optimizer_updates": 0, "independent_confirmation_clearance": False, "training_clearance": False,
        "ppo_launch_clearance": False, "source_inputs_repaired": False,
        "all_source_bytes_unchanged": True, "mask_load_reverified": True, "wrapper_checks": wrapper_checks,
        "source_evidence_entities": len(evidence["entities"]), "source_bindings": observed}
    write_json(output / "report.json", report)
    with (output / "source_preparation.executed.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    names = ["graph_inputs.jsonl", "qid_source_evidence.json", "question_checks.jsonl", "mask_manifest.json",
             "report.json", "source_preparation.executed.py", "norm_only/gate.json", "features_v2/gate.json",
             "norm_only/identity_mask_checks.jsonl", "features_v2/identity_mask_checks.jsonl"]
    write_json(output / "manifest.json", {"schema_version": SCHEMA, "experiment_id": report["experiment_id"],
        "status": report["status"], "source_bindings": observed,
        "outputs": {name: identity(output / name) for name in names}, "independent_confirmation_clearance": False,
        "training_clearance": False, "ppo_launch_clearance": False, "gold_access": False, "optimizer_updates": 0})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output_dir.exists()
    try:
        result = run(args.inputs_manifest.resolve(), args.cohort_manifest.resolve(), args.calibration_manifest.resolve(), args.output_dir.resolve())
    except Exception as exc:
        if not existed and args.output_dir.exists() and not (args.output_dir / "FAILED.json").exists():
            write_json(args.output_dir / "FAILED.json", {"status": "FAILED", "type": type(exc).__name__,
                "message": str(exc), "gpu_used": False, "optimizer_updates": 0})
        raise
    print(json.dumps({key: result[key] for key in ("status", "questions", "source_status_counts", "by_graph_type")}))
