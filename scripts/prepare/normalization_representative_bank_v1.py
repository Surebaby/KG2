"""Small append-only train-only source-normalization supplement, never PPO.

Identity-only reuse-first hash sampling: H40, M40, W32graph+8ordinary.
Old train candidates are reused exactly. Source labels are projected away;
neither answer quality nor trajectory validity influences selection or retries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.prepare import source_quality_candidate_bank_v1 as bank

ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
MODULE = "scripts.prepare.normalization_representative_bank_v1"
VERSION = "normalization-representative-bank-v1"
DEFAULT = ROOT / "outputs/audits/normalization_representative_bank_v1_seed42_20260906_r2"
OLD_INPUT = ROOT / "outputs/audits/source_quality_candidate_bank_v1_inputs_seed42_tensorboard_v1"
OLD_GENERATED = ROOT / "outputs/audits/source_quality_candidate_bank_v1_generated_seed42_local4090_v1"
OLD_SCORED = ROOT / "outputs/audits/source_quality_candidates_format_v2_scored_local_seed42"
OLD_FIT = ROOT / "outputs/calibration/source_credit_gate_v1_local_seed42"
QUOTAS = [("hotpotqa", 0, 40), ("musique", 0, 40), ("2wikimultihopqa", 1, 32), ("2wikimultihopqa", 0, 8)]


def status(directory, state, **extra):
    value = {"time_utc": datetime.now(timezone.utc).isoformat(), "status": state,
             "policy_optimizer_updates": 0, "pid": os.getpid(), **extra}
    encoded = bank.canonical_json(value) + "\n"
    with (directory / "events.jsonl").open("a") as handle:
        handle.write(encoded)
    tmp = directory / "status.json.tmp"
    tmp.write_text(encoded)
    tmp.replace(directory / "status.json")
    print(encoded.strip(), flush=True)


def select(groups, old_inputs, assignments, protected):
    consumed = {row["family_sha256"] for row in assignments if row["split"] != "train"}
    train = {(row["dataset"], row["qid"]) for row in assignments if row["split"] == "train"}
    old = {bank.key(row): row for row in old_inputs}
    protected_qids = {row["qid"] for row in protected}
    protected_questions = {bank.row_identity(row)["question_sha256"] for row in protected}
    protected_families = {bank.row_identity(row)["family_sha256"] for row in protected}
    available = [row for row in groups if row["family_sha256"] not in consumed | protected_families
                 and row["qid"] not in protected_qids and row["question_sha256"] not in protected_questions]
    selected, used_families = [], set()
    for dataset, graph, count in QUOTAS:
        pool = [row for row in available if row["dataset"] == dataset and int(row["process_reward_eligible"]) == graph]
        def reusable(row):
            return bank.key(row) in old and (dataset, row["qid"]) in train
        pool.sort(key=lambda row: (not reusable(row), bank.digest([
            "normalization-representative-bank-v1", 42, row["family_sha256"], bank.key(row)])))
        picked = []
        for row in pool:
            if row["family_sha256"] in used_families:
                continue
            picked.append({**row, "reuse": reusable(row)})
            used_families.add(row["family_sha256"])
            if len(picked) == count:
                break
        if len(picked) != count:
            raise ValueError("insufficient isolated normalization questions")
        selected.extend(picked)
    return sorted(selected, key=bank.key), {
        "available_by_dataset_graph": dict(Counter(f"{row['dataset']}::graph{int(row['process_reward_eligible'])}" for row in available)),
        "consumed_family_count_excluded": len(consumed),
        "selected_questions": len(selected), "reused_questions": sum(row["reuse"] for row in selected),
        "new_questions": sum(not row["reuse"] for row in selected),
        "new_by_dataset": dict(Counter(row["dataset"] for row in selected if not row["reuse"])),
        "selection_uses_gold_scores_or_validity": False,
    }


def _read_safe_evidence(path, selected):
    """Whitelist source evidence after identity selection; discard label keys."""
    allowed = {"dataset", "qid", "question", "retrieved_passages", "metadata"}
    result = {}
    def remove_labels(value):
        return {key: item for key, item in value.items() if key not in bank.FORBIDDEN}
    with path.open() as handle:
        for line in handle:
            projected = json.loads(line, object_hook=remove_labels)
            key = bank.key(projected)
            if key in selected:
                row = {name: projected[name] for name in allowed}
                if row["metadata"].get("source_split") != "train":
                    raise ValueError("normalization evidence is not train-only")
                result[key] = row
    return result


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    status(directory, "PREPARING_IDENTITY_ONLY_SUPPLEMENT")
    data = ROOT / bank.DEFAULT_DATA
    final = json.loads((data / "report.json").read_text())
    if final["status"] != "COMPLETE_DATA_NOT_TRAINED" or not all(final["gates"].values()):
        raise ValueError("final training release is not frozen/pass")
    sources = {name: data / filename for name, filename in (
        ("groups", "prompt_groups.jsonl"), ("questionkg", "question_kg_records.jsonl"),
        ("gate", "source_gate_records.jsonl"), ("silver", "silver_train.jsonl"))}
    output_keys = {"groups": "prompt_groups", "questionkg": "question_kg_records", "gate": "source_gate_records", "silver": "silver_train"}
    for name, path in sources.items():
        if bank.file_sha(path) != final["outputs"][output_keys[name]]["sha256"]:
            raise ValueError("frozen final-data source hash mismatch")
    original = bank.load_release(OLD_INPUT, bank.PREPARE_VERSION)
    original_rows = bank.validate_inputs(OLD_INPUT, original)
    assignments = bank.read_rows(OLD_FIT / "assignments.jsonl")
    fit_manifest = json.loads((OLD_FIT / "manifest.json").read_text())
    if bank.file_sha(OLD_FIT / "assignments.jsonl") != fit_manifest["outputs"]["assignments.jsonl"]["sha256"]:
        raise ValueError("old split assignment hash mismatch")
    ledger = bank.resolve(original["source_bindings"]["protected_ledger"], OLD_INPUT)
    groups = bank.read_rows(sources["groups"])
    selected, summary = select(groups, original_rows, assignments, bank.read_rows(ledger))
    if summary["reused_questions"] != 47 or summary["new_questions"] != 73:
        raise ValueError(f"expected frozen 47/73 reuse/new allocation: {summary}")
    bank.write_rows(directory / "selection.question_only.jsonl", selected)
    selected_keys = {bank.key(row) for row in selected}
    evidence = _read_safe_evidence(sources["silver"], selected_keys)
    graph_index = {bank.key(row): row for row in bank.read_rows(sources["questionkg"])}
    gate_index = {bank.key(row): row for row in bank.read_rows(sources["gate"])}
    old_index = {bank.key(row): row for row in original_rows}
    render = bank.make_renderer(ROOT / original["policy_path"])
    source_hashes = {name: bank.file_sha(path) for name, path in sources.items()}
    rows = []
    for selection in selected:
        key = bank.key(selection)
        if selection["reuse"]:
            row = deepcopy(old_index[key])
        else:
            source, record, decision = evidence[key], graph_index[key], gate_index[key]
            identity = bank.row_identity(selection)
            if bank.row_identity(source) != identity or bank.row_identity(record) != identity:
                raise ValueError("new evidence source identity mismatch")
            passages = source["retrieved_passages"]
            if not bank._ten_safe_passages(passages):
                raise ValueError("new question does not have exactly ten safe passages")
            messages = bank.build_rl_messages(selection["question"], passages, record["kg_subgraph"], top_k=10, max_kg_triples=12)
            prompt, tokens = render(messages)
            if tokens > 6144:
                raise ValueError("no question dropping or prompt truncation permitted")
            row = {**identity, "question_key": key, "source_split": "train", "m_graph": decision["m_graph"],
                "retrieved_passages": passages, "kg_subgraph": record["kg_subgraph"], "source_quality_record": record,
                "messages": messages, "prompt": prompt, "prompt_tokens": tokens,
                "source_record_sha256": bank.digest(record), "fullsource_record": record,
                "spec": {"query": selection["question"], "retrieved_passages": passages, "kg_subgraph": record["kg_subgraph"],
                         "metadata": {"dataset": selection["dataset"], "qid": selection["qid"], "source_quality_record": record}},
                "source_bindings": {"questionkg": {"file_sha256": source_hashes["questionkg"], "row_sha256": bank.digest(record)},
                                    "gate": {"file_sha256": source_hashes["gate"], "row_sha256": bank.digest(decision)},
                                    "groups": {"file_sha256": source_hashes["groups"], "row_sha256": bank.digest({k:v for k,v in selection.items() if k != 'reuse'})}}}
            row["input_sha256"] = bank.input_hash(row)
        bank.assert_gold_free(row)
        rows.append(row)
    bank.write_rows(directory / "inputs.all.jsonl", rows)
    overlap = bank.isolation(rows, bank.read_rows(ledger))
    # Freeze importable copies so concurrent authorized work cannot alter this run.
    snapshot = directory / "runtime_code"
    code_bindings = {}
    for tree in ("kgproweight", "scripts"):
        for source in sorted((ROOT / tree).rglob("*.py")):
            relative = source.relative_to(ROOT)
            dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            before = bank.file_sha(source)
            dest.write_bytes(source.read_bytes())
            if bank.file_sha(dest) != before:
                raise ValueError("source changed while snapshotting")
            code_bindings[str(relative)] = bank.identity(dest)
    pending = directory / "new_inputs"
    pending.mkdir()
    fresh_keys = {bank.key(row) for row in selected if not row["reuse"]}
    new_rows = [row for row in rows if bank.key(row) in fresh_keys]
    bank.write_rows(pending / "inputs.jsonl", new_rows)
    bindings = {name: {**info, "path": str(bank.resolve(info, OLD_INPUT))}
                for name, info in original["source_bindings"].items() if not name.startswith("code:")}
    bindings.update({"code:" + name: info for name, info in code_bindings.items()})
    pending_report = {key: value for key, value in original.items()
                      if key not in {"outputs", "controls_per_dataset", "binding_revision", "binding_validation"}}
    bank.finish(pending, {**pending_report, "experiment_id": "NORMALIZATION-REPRESENTATIVE-V1-NEW73-INPUTS-20260906-R2",
        "status": "NORMALIZATION_ONLY_TRAIN_INPUTS_FROZEN", "source_bindings": bindings,
        "n_questions": len(new_rows), "n_candidates": 2 * len(new_rows),
        "by_dataset": dict(Counter(row["dataset"] for row in new_rows)),
        "graph_eligible": sum(row["m_graph"] for row in new_rows),
        "max_input_tokens_observed": max(row["prompt_tokens"] for row in new_rows),
        "parent_preparation_manifest": bank.identity(OLD_INPUT / "manifest.json"),
        "boundary": "73 new normalization-only train questions; no gate fitting, no PPO, no independent confirmation",
        "qid_order": [bank.key(row) for row in new_rows], "training_started": False}, ["inputs.jsonl"])
    protocol = {"schema_version": VERSION, "experiment_id": "NORMALIZATION-REPRESENTATIVE-V1-20260906-R2",
        "selection": summary, "quotas": QUOTAS, "seed": 42, "K": 2,
        "selection_contract": "reuse frozen train candidates first, then fixed hash of family+qid; unique selected families; no scores/validity/Gold selection",
        "membership": "normalization_train_only_not_independent_confirmation",
        "generation": original["generation"], "scoring": {**original["scoring"], "format_contract": "source-gate-runtime-v2-format-v2"},
        "existing_candidate_reuse": "exact frozen generation and ReaRAG raw score bytes/values; no quality-based substitution",
        "models": {key: original[key] for key in ("base_model", "policy_tokenizer", "rearag_model")},
        "policy_path": str(ROOT / original["policy_path"]), "project_root": str(ROOT),
        "protected_overlap": overlap, "source_bindings": {name: bank.identity(path) for name, path in sources.items()},
        "parent_bindings": {"inputs": bank.identity(OLD_INPUT / "manifest.json"),
                            "generation": bank.identity(OLD_GENERATED / "manifest.json"),
                            "scored": bank.identity(OLD_SCORED / "manifest.json"),
                            "assignments": bank.identity(OLD_FIT / "assignments.jsonl")},
        "code_bindings": code_bindings, "input_sha256": bank.file_sha(directory / "inputs.all.jsonl"),
        "selection_sha256": bank.file_sha(directory / "selection.question_only.jsonl"),
        "new_bank_manifest_sha256": bank.file_sha(pending / "manifest.json"),
        "budget_estimate": "146 new SFT generations about 15-25 min, then batch1 ReaRAG about 3-6 min; 24GB4090 model stages serialized",
        "gold_used": False, "label_values_inspected_or_em_computed": False,
        "label_source_projection": "Only dataset/qid/question/frozen passages/source_split; label keys removed at JSON object projection, after identity-only selection",
        "policy_optimizer_updates": 0, "no_failed_candidate_replacement": True}
    bank.write_json(directory / "protocol.json", protocol)
    status(directory, "READY_FOR_SERIAL_GENERATE_SCORE", **summary)
    return protocol


def verify(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    for name, info in protocol["code_bindings"].items():
        if bank.file_sha(Path(info["path"])) != info["sha256"]:
            raise ValueError(f"runtime snapshot changed: {name}")
    for name, sha in (("inputs.all.jsonl", protocol["input_sha256"]),
                      ("selection.question_only.jsonl", protocol["selection_sha256"]),
                      ("new_inputs/manifest.json", protocol["new_bank_manifest_sha256"])):
        if bank.file_sha(directory / name) != sha:
            raise ValueError("frozen supplement input changed")
    return protocol


def score(directory):
    protocol = verify(directory)
    old_generated = bank.load_release(OLD_GENERATED, bank.GENERATION_VERSION)
    bank.load_release(OLD_SCORED, bank.VERSION)
    old_gens = {row["candidate_id"]: row for row in bank.read_rows(OLD_GENERATED / "generations.jsonl")}
    old_scores = {row["candidate_id"]: row for row in bank.read_rows(OLD_SCORED / "candidates.scored.jsonl")}
    bank.load_release(directory / "generated", bank.GENERATION_VERSION)
    generated = {row["candidate_id"]: row for row in bank.read_rows(directory / "generated/generations.jsonl")}
    selection = {bank.key(row): row for row in bank.read_rows(directory / "selection.question_only.jsonl")}
    model = ROOT / protocol["models"]["rearag_model"]["path"]
    bank.validate_model(model, protocol["models"]["rearag_model"])
    torch = bank.require_cuda("cuda:0")
    from kgproweight.reward.text_reward_model import RearagPromptScorer
    from scripts.prepare.score_sourcegate_candidates_format_v2 import score_candidate_v2
    scorer = RearagPromptScorer.from_pretrained(str(model), device="cuda:0", dtype="bf16")
    torch.cuda.reset_peak_memory_stats()
    n, fresh, reused, valid, steps = 0, 0, 0, 0, 0
    started = time.monotonic()
    with (directory / "normalization_rows.jsonl").open("x") as merged, (directory / "new_scored.full.jsonl").open("x") as full:
        for row in bank.read_rows(directory / "inputs.all.jsonl"):
            key = bank.key(row)
            for k in range(2):
                cid = f"{key}::k{k}"
                if selection[key]["reuse"]:
                    scored, prediction = old_scores[cid], old_gens[cid]
                    if scored["generation"] != prediction["generation"] or scored["input_sha256"] != row["input_sha256"]:
                        raise ValueError("old candidate reuse identity mismatch")
                    reused += 1
                    origin = "exact_frozen_train_candidate_reuse"
                else:
                    prediction = generated[cid]
                    scored = score_candidate_v2(row, prediction, scorer)
                    full.write(bank.canonical_json(scored) + "\n"); full.flush()
                    fresh += 1
                    origin = "new_frozen_sft_generation_original_bf16_rearag"
                valid += int(scored["trajectory_valid"]); steps += len(scored["raw_text"])
                compact = {"schema_version": "normalization-only-train-score-row-v1", "candidate_id": cid,
                    "dataset": row["dataset"], "qid": row["qid"], "question_sha256": row["question_sha256"],
                    "family_sha256": row["family_sha256"], "split": "train", "normalization_train_only": True,
                    "trajectory_valid": scored["trajectory_valid"], "raw_text": scored["raw_text"],
                    "format_validation": scored["format_validation"], "input_sha256": row["input_sha256"],
                    "generation_sha256": bank.digest(prediction), "score_row_sha256": bank.digest(scored),
                    "origin": origin, "gold_used": False, "selection_uses_validity": False}
                merged.write(bank.canonical_json(compact) + "\n"); merged.flush()
                n += 1
                if n % 10 == 0:
                    status(directory, "SCORING_SUPPLEMENT", completed=n, expected=240, reused=reused, new=fresh,
                        valid=valid, text_steps=steps, elapsed_seconds=round(time.monotonic() - started, 1))
    if (n, fresh, reused) != (240, 146, 94):
        raise ValueError("normalization supplement population mismatch")
    report = {"schema_version": "normalization-only-train-score-bank-v1", "experiment_id": protocol["experiment_id"],
        "status": "COMPLETE_NORMALIZATION_TRAIN_ONLY_NOT_GATE_OR_PPO_CLEARANCE", "questions": 120,
        "candidates": n, "new_candidates": fresh, "reused_candidates": reused, "valid_candidates": valid,
        "text_steps": steps, "gold_used": False, "policy_optimizer_updates": 0,
        "membership": protocol["membership"], "selection": protocol["selection"],
        "protocol": bank.identity(directory / "protocol.json"),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    bank.finish(directory, report, ["normalization_rows.jsonl", "new_scored.full.jsonl", "protocol.json", "inputs.all.jsonl", "selection.question_only.jsonl"])
    status(directory, report["status"], completed=n, expected=n, valid=valid, text_steps=steps)


def run(directory):
    protocol = verify(directory)
    snapshot = directory / "runtime_code"
    env = {**os.environ, "KGPW_PROJECT_ROOT": str(ROOT), "PYTHONPATH": str(snapshot) + ":" + str(ROOT / "flashrag_src"),
           "HF_MODULES_CACHE": str(ROOT / "outputs/runtime/hf_modules_cache"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    commands = [
        ("GENERATING_SUPPLEMENT", [sys.executable, "-m", "scripts.prepare.source_quality_candidate_bank_v1", "generate",
            "--bank-dir", str(directory / "new_inputs"), "--output-dir", str(directory / "generated"),
            "--experiment-id", protocol["experiment_id"] + "-GENERATE146", "--project-root", str(ROOT),
            "--base-model", str(ROOT / protocol["models"]["base_model"]["path"]), "--policy", protocol["policy_path"]]),
        ("SCORING_SUPPLEMENT", [sys.executable, "-m", MODULE, "score", "--bank-dir", str(directory)])]
    for phase, command in commands:
        log = directory / ("generate.log" if phase.startswith("GENERATING") else "score.log")
        with log.open("x") as handle:
            child = subprocess.Popen(command, cwd=snapshot, env=env, stdout=handle, stderr=subprocess.STDOUT)
            status(directory, phase, child_pid=child.pid, log=str(log))
            while child.poll() is None:
                time.sleep(10)
                if phase.startswith("GENERATING"):
                    path = directory / "generated/generations.jsonl"
                    count = sum(1 for _ in path.open()) if path.exists() else 0
                    status(directory, phase, child_pid=child.pid, completed=count, expected=146)
            if child.returncode:
                status(directory, "FAILED_APPEND_ONLY_OUTPUTS_RETAINED", stage=phase, exit_code=child.returncode)
                raise RuntimeError(f"{phase} child failed ({child.returncode})")
    status(directory, "COMPLETE_NORMALIZATION_TRAIN_ONLY_NOT_GATE_OR_PPO_CLEARANCE", completed=240, expected=240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "score"))
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.command](args.bank_dir.resolve())
