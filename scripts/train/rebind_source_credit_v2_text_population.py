"""Replace only text fit-population statistics after a frozen small-bank run.

Alpha weights, source masks, Graph statistics and the original diagnostic
candidate bank remain byte-identical or value-identical as explicitly checked.
The balanced bank contains train-only questions; it is not fresh confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from kgproweight.reward.source_credit_gate_v2 import SourceCreditGateV2
from kgproweight.reward.source_quality_gate_v1 import canonical_sha256
from kgproweight.reward.source_reward_normalization_v2 import fit_text_normalization_v2
from scripts.train.calibrate_source_credit_gate_v1 import write_json
from scripts.train.calibrate_source_credit_gate_v2 import CODE_FILES as CALIBRATION_CODE_FILES, ROOT, identity
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


SCHEMA = "source-credit-v2-representative-text-population-rebind-v1"
DATASETS = {"hotpotqa", "2wikimultihopqa", "musique"}
CODE_FILES = tuple(dict.fromkeys((*CALIBRATION_CODE_FILES,
    "scripts/prepare/freeze_qpeg_v1_protocol.py", "scripts/train/rebind_source_credit_v2_text_population.py")))
PARENT_FILES = {"gate.json", "report.json", "candidates.jsonl", "assignments.jsonl"}
BANK_FILES = {"normalization_rows.jsonl", "new_scored.full.jsonl", "protocol.json",
              "inputs.all.jsonl", "selection.question_only.jsonl", "report.json"}
SELECTION_QUOTAS = [["hotpotqa", 0, 40], ["musique", 0, 40],
                    ["2wikimultihopqa", 1, 32], ["2wikimultihopqa", 0, 8]]


def bound_file(binding, base, *, expected=None, observed=None):
    """Resolve in the owning release, and verify the exact file subsequently read."""
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        raise ValueError("a path/hash file binding is required")
    path = Path(binding["path"])
    path = (path if path.is_absolute() else base / path).resolve()
    if expected is not None and path != expected.resolve():
        raise ValueError("release binding path differs from the exact file to be read")
    if binding.get("origin_path") and Path(binding["origin_path"]).resolve() != path:
        raise ValueError("release origin_path contradicts its bound path")
    actual = identity(path)
    if actual["sha256"] != binding.get("sha256"):
        raise ValueError(f"bound bytes changed: {path}")
    if observed is not None:
        previous = observed.setdefault(str(path), actual)
        if previous != actual:
            raise ValueError("a bound input changed while it was being validated")
    return path


def release_outputs(manifest, directory, *, required=None, observed=None):
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or (required is not None and set(outputs) != set(required)):
        raise ValueError("release must bind exactly its registered output files")
    result = {}
    for name, binding in outputs.items():
        if Path(name).name != name:
            raise ValueError("release output names must be plain filenames")
        result[name] = bound_file(binding, directory, expected=directory / name, observed=observed)
    return result


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_rows(rows, field):
    result = {}
    for row in rows:
        value = field(row) if callable(field) else row.get(field)
        if value is None or value in result:
            raise ValueError("source records must have complete unique identities")
        result[value] = row
    return result


def question_key(row):
    if row.get("dataset") not in DATASETS or not isinstance(row.get("qid"), str) or not row["qid"]:
        raise ValueError("source question requires a supported dataset and string qid")
    return f"{row['dataset']}::{row['qid']}"


def question_identity(row):
    question = row.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("bound question text is required to replay identity hashes")
    result = {"question_sha256": hashlib.sha256(question.encode()).hexdigest(),
              "family_sha256": family_sha256(question), "family_version": FAMILY_VERSION}
    if any(row.get(key) != value for key, value in result.items()):
        raise ValueError("question/family identity does not reproduce from frozen text")
    return result


def verify_normalization_sources(directory, bank, paths, rows, assignments, observed):
    """Replay compact rows against bound inputs, selection, generations and scores."""
    protocol = json.loads(paths["protocol.json"].read_text())
    if (protocol.get("schema_version") != "normalization-representative-bank-v1"
            or protocol.get("K") != 2 or protocol.get("seed") != 42
            or protocol.get("membership") != "normalization_train_only_not_independent_confirmation"
            or protocol.get("policy_optimizer_updates") != 0
            or isinstance(protocol.get("policy_optimizer_updates"), bool)
            or protocol.get("gold_used") is not False
            or protocol.get("no_failed_candidate_replacement") is not True
            or protocol.get("quotas") != SELECTION_QUOTAS
            or protocol.get("selection", {}).get("selection_uses_gold_scores_or_validity") is not False):
        raise ValueError("normalization source protocol lacks the frozen train/SFT contract")
    bound_file(bank["protocol"], directory, expected=paths["protocol.json"], observed=observed)
    for name, key in (("inputs.all.jsonl", "input_sha256"), ("selection.question_only.jsonl", "selection_sha256")):
        if identity(paths[name])["sha256"] != protocol.get(key):
            raise ValueError("normalization source inputs/selection disagree with its protocol")
    for name, binding in protocol.get("code_bindings", {}).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("producer source snapshot names must be safe relative paths")
        bound_file(binding, directory, expected=directory / "runtime_code" / name, observed=observed)
    if not protocol.get("code_bindings"):
        raise ValueError("normalization producer requires its frozen source snapshots")
    parents = protocol.get("parent_bindings") or {}
    parent_paths = {name: bound_file(parents[name], ROOT, observed=observed)
                    for name in ("inputs", "generation", "scored", "assignments")}
    old_assignments = read_rows(parent_paths["assignments"])
    fields = ("candidate_id", "dataset", "qid", "family_sha256", "split")
    if {tuple(row.get(k) for k in fields) for row in old_assignments} != {
            tuple(row.get(k) for k in fields) for row in assignments}:
        raise ValueError("normalization selection and gate parent use different frozen family assignments")
    old_inputs_manifest = json.loads(parent_paths["inputs"].read_text())
    if old_inputs_manifest.get("schema_version") != "source-quality-candidate-preparation-v1":
        raise ValueError("unknown parent SFT input release")
    old_input_files = release_outputs(old_inputs_manifest, parent_paths["inputs"].parent, observed=observed)
    old_inputs = unique_rows(read_rows(old_input_files["inputs.jsonl"]), question_key)
    if any(protocol.get("models", {}).get(k) != old_inputs_manifest.get(k)
           for k in ("base_model", "policy_tokenizer", "rearag_model")):
        raise ValueError("normalization source changed a frozen model identity")
    if protocol.get("generation") != old_inputs_manifest.get("generation"):
        raise ValueError("normalization source changed the frozen generation contract")
    policy_path = Path(protocol["policy_path"]).resolve()
    old_policy_path = Path(old_inputs_manifest["policy_path"])
    old_policy_path = (old_policy_path if old_policy_path.is_absolute() else ROOT / old_policy_path).resolve()
    if policy_path != old_policy_path:
        raise ValueError("normalization source is not the original frozen SFT policy")
    for name in ("policy", "policy_config"):
        bound_file(old_inputs_manifest["source_bindings"][name], ROOT, observed=observed)
    new_input_manifest_path = directory / "new_inputs/manifest.json"
    bound_file({"path": str(new_input_manifest_path), "sha256": protocol["new_bank_manifest_sha256"]},
               directory, expected=new_input_manifest_path, observed=observed)
    new_input_manifest = json.loads(new_input_manifest_path.read_text())
    new_input_files = release_outputs(new_input_manifest, new_input_manifest_path.parent, observed=observed)
    if (new_input_manifest.get("training_started") is not False
            or new_input_manifest.get("base_model") != old_inputs_manifest.get("base_model")
            or new_input_manifest.get("generation") != old_inputs_manifest.get("generation")
            or new_input_manifest.get("source_bindings", {}).get("policy", {}).get("sha256")
                != old_inputs_manifest["source_bindings"]["policy"]["sha256"]):
        raise ValueError("new normalization inputs do not bind the original frozen SFT generator")
    new_inputs = unique_rows(read_rows(new_input_files["inputs.jsonl"]), question_key)
    generation_sets = {}
    for label, manifest_path, input_manifest_path in (
        ("reuse", parent_paths["generation"], parent_paths["inputs"]),
        ("new", directory / "generated/manifest.json", new_input_manifest_path),
    ):
        manifest = json.loads(manifest_path.read_text())
        observed[str(manifest_path.resolve())] = identity(manifest_path)
        if (manifest.get("schema_version") != "source-quality-candidate-generation-v1"
                or manifest.get("training_started") is not False
                or manifest.get("bank_manifest_sha256") != identity(input_manifest_path)["sha256"]
                or manifest.get("generation_contract_sha256") != canonical_sha256(protocol["generation"])):
            raise ValueError("generation release does not bind the frozen SFT inputs/contract")
        files = release_outputs(manifest, manifest_path.parent, observed=observed)
        generation_sets[label] = unique_rows(read_rows(files["generations.jsonl"]), "candidate_id")
    old_scores_manifest = json.loads(parent_paths["scored"].read_text())
    if old_scores_manifest.get("schema_version") != "source-quality-candidate-bank-v1":
        raise ValueError("unknown original score release")
    old_score_files = release_outputs(old_scores_manifest, parent_paths["scored"].parent, observed=observed)
    score_sets = {"reuse": unique_rows(read_rows(old_score_files["candidates.scored.jsonl"]), "candidate_id"),
                  "new": unique_rows(read_rows(paths["new_scored.full.jsonl"]), "candidate_id")}
    inputs = unique_rows(read_rows(paths["inputs.all.jsonl"]), question_key)
    selection = unique_rows(read_rows(paths["selection.question_only.jsonl"]), question_key)
    if len(inputs) != 120 or set(inputs) != set(selection) or {question_key(r) for r in rows} != set(inputs):
        raise ValueError("normalization row/input/selection populations differ")
    if len({row["family_sha256"] for row in selection.values()}) != 120:
        raise ValueError("representative selection must keep unique frozen families")
    groups_path = bound_file(protocol["source_bindings"]["groups"], ROOT, observed=observed)
    groups = unique_rows(read_rows(groups_path), question_key)
    ledger_path = bound_file(old_inputs_manifest["source_bindings"]["protected_ledger"], ROOT, observed=observed)
    ledger = read_rows(ledger_path)
    protected_qids = {row["qid"] for row in ledger}
    protected_qhash = {hashlib.sha256(row["question"].encode()).hexdigest() for row in ledger}
    protected_family = {family_sha256(row["question"]) for row in ledger}
    old_train_keys = {question_key(row) for row in old_assignments if row["split"] == "train"}
    consumed_families = {row["family_sha256"] for row in old_assignments if row["split"] != "train"}
    available = [row for row in groups.values() if row["family_sha256"] not in consumed_families | protected_family
                 and row["qid"] not in protected_qids and row["question_sha256"] not in protected_qhash]
    replayed_selection, used_families = {}, set()
    for dataset, graph, count in SELECTION_QUOTAS:
        pool = [row for row in available if row["dataset"] == dataset and int(row["process_reward_eligible"]) == graph]
        def reusable(row):
            return question_key(row) in old_inputs and question_key(row) in old_train_keys
        pool.sort(key=lambda row: (not reusable(row), canonical_sha256([
            "normalization-representative-bank-v1", 42, row["family_sha256"], question_key(row)])))
        picked = 0
        for row in pool:
            if row["family_sha256"] in used_families:
                continue
            used_families.add(row["family_sha256"])
            replayed_selection[question_key(row)] = {**row, "reuse": reusable(row)}
            picked += 1
            if picked == count:
                break
        if picked != count:
            raise ValueError("frozen identity-only selection cannot reproduce its quota")
    if replayed_selection != selection:
        raise ValueError("normalization selection differs from frozen reuse-first hash sampling")
    origins = Counter()
    for key, source in inputs.items():
        question_identity(source)
        chosen = selection[key]
        question_identity(chosen)
        if source.get("source_split") != "train" or chosen.get("gold_access") is not False:
            raise ValueError("bound source questions are not explicitly Gold-free train data")
        if {k:v for k,v in chosen.items() if k != "reuse"} != groups.get(key):
            raise ValueError("selection row differs from its bound original train question record")
        if (source["qid"] in protected_qids or source["question_sha256"] in protected_qhash
                or source["family_sha256"] in protected_family):
            raise ValueError("representative source overlaps a protected identity/family")
        if source["input_sha256"] != canonical_sha256({k:v for k,v in source.items() if k != "input_sha256"}):
            raise ValueError("normalization input hash does not reproduce")
        if not isinstance(chosen.get("reuse"), bool):
            raise ValueError("source selection must explicitly distinguish reuse and new inputs")
        if chosen["reuse"]:
            if key not in old_train_keys or source != old_inputs.get(key):
                raise ValueError("reused normalization input is not the exact frozen train input")
        elif source != new_inputs.get(key):
            raise ValueError("new normalization input differs from its generation input release")
    if set(new_inputs) != {key for key,row in selection.items() if row["reuse"] is False}:
        raise ValueError("new generation input population differs from frozen selection")
    for row in rows:
        key, cid = question_key(row), row["candidate_id"]
        source, chosen = inputs[key], selection[key]
        origin = "reuse" if chosen["reuse"] else "new"
        expected_origin = "exact_frozen_train_candidate_reuse" if origin == "reuse" else "new_frozen_sft_generation_original_bf16_rearag"
        if (row.get("schema_version") != "normalization-only-train-score-row-v1"
                or row.get("normalization_train_only") is not True or row.get("gold_used") is not False
                or row.get("selection_uses_validity") is not False or row.get("origin") != expected_origin):
            raise ValueError("compact normalization row contradicts its frozen source/selection contract")
        for field in ("question_sha256", "family_sha256", "input_sha256"):
            if row.get(field) != source.get(field):
                raise ValueError("normalization row identity differs from its bound input")
        if cid not in {f"{key}::k0", f"{key}::k1"}:
            raise ValueError("normalization candidate ID does not match its question/K2 identity")
        prediction, scored = generation_sets[origin].get(cid), score_sets[origin].get(cid)
        if not prediction or not scored:
            raise ValueError("normalization row lacks its bound generation/score record")
        if row.get("generation_sha256") != canonical_sha256(prediction) or row.get("score_row_sha256") != canonical_sha256(scored):
            raise ValueError("normalization row generation/score record digest mismatch")
        for value in (prediction, scored):
            if value.get("input_sha256") != source["input_sha256"] or question_key(value) != key:
                raise ValueError("generation/score record belongs to another frozen input")
        if (prediction.get("candidate_index") != int(cid[-1])
                or prediction.get("policy_sha256") != old_inputs_manifest["source_bindings"]["policy"]["sha256"]
                or prediction.get("base_model_identity_sha256") != canonical_sha256(protocol["models"]["base_model"])
                or prediction.get("generation_contract_sha256") != canonical_sha256(protocol["generation"])
                or prediction.get("bank_manifest_sha256") != identity(parent_paths["inputs"] if origin == "reuse" else new_input_manifest_path)["sha256"]):
            raise ValueError("candidate generation does not bind the frozen policy/model/contract")
        if scored.get("generation") != prediction.get("generation") or any(
                row.get(field) != scored.get(field) for field in ("raw_text", "trajectory_valid", "format_validation")):
            raise ValueError("compact normalization observations differ from the bound original score row")
        origins[origin] += 1
    if set(generation_sets["new"]) != {r["candidate_id"] for r in rows if not selection[question_key(r)]["reuse"]} or set(score_sets["new"]) != set(generation_sets["new"]):
        raise ValueError("new scored/generated candidates were added, dropped or replaced")
    return {"replayed_candidates": len(rows), "origin_counts": dict(origins),
            "input_selection_generation_score_identity_reproduced": True,
            "identity_only_selection_reproduced": True,
            "protected_overlap": 0, "policy_identity": old_inputs_manifest["source_bindings"]["policy"],
            "source_split_basis": "bound train inputs, original question records, frozen family assignments and protected ledger"}


def validate_normalization_rows(rows, old_assignments):
    if len(rows) != 240 or len({row["candidate_id"] for row in rows}) != 240:
        raise ValueError("representative normalization bank must retain exactly 240 unique K2 candidates")
    consumed = {row["family_sha256"] for row in old_assignments if row["split"] != "train"}
    by_question = defaultdict(list)
    for row in rows:
        question_key(row)
        if row.get("split") != "train" or row.get("family_split", "train") != "train":
            raise ValueError("normalization replacement requires explicitly train-only rows")
        if any(key in row for key in ("gold", "gold_answer", "gold_answers", "gold_target", "gold_answer_aliases",
                                     "answer", "answers", "answer_aliases", "golden_answers", "target", "labels")):
            raise ValueError("normalization cannot consume Gold labels")
        for key in ("family_sha256", "question_sha256"):
            value = row.get(key)
            if not isinstance(value,str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("normalization rows require frozen family/question identities")
        if row["family_sha256"] in consumed:
            raise ValueError("normalization population overlaps consumed non-train families")
        by_question[row["dataset"],row["qid"]].append(row)
    if set(dataset for dataset,_ in by_question) != DATASETS:
        raise ValueError("normalization bank must cover the three frozen datasets")
    if Counter(dataset for dataset,_ in by_question) != Counter({key:40 for key in DATASETS}):
        raise ValueError("normalization bank must contain exactly 40 questions per dataset")
    for candidates in by_question.values():
        if len(candidates) != 2 or len({(r["family_sha256"],r["question_sha256"]) for r in candidates}) != 1:
            raise ValueError("normalization bank must preserve K2 and consistent question identities")
        key = question_key(candidates[0])
        if {row["candidate_id"] for row in candidates} != {f"{key}::k0", f"{key}::k1"}:
            raise ValueError("normalization candidate IDs must preserve their question and K2 indices")
    if len({candidates[0]["family_sha256"] for candidates in by_question.values()}) != 120:
        raise ValueError("normalization selection must preserve 120 unique families")
    return {"questions":120, "candidates":240, "by_dataset_questions":{key:40 for key in sorted(DATASETS)},
        "consumed_nontrain_family_overlap":0, "gold_used":False}


def rebind(calibration_dir: Path, normalization_manifest: Path, output_dir: Path):
    if output_dir.exists(): raise FileExistsError("refusing to overwrite representative normalization successor")
    parent_path = calibration_dir / "manifest.json"
    parent_identity, bank_identity = identity(parent_path), identity(normalization_manifest)
    observed = {str(parent_path.resolve()): parent_identity,
                str(normalization_manifest.resolve()): bank_identity}
    parent = json.loads(parent_path.read_text())
    if (parent.get("schema_version") != "source-credit-v2-development-calibration-manifest-v1"
            or parent.get("status") != "V2_ENGINEERING_REPAIR_FITTED_NOT_INDEPENDENT_CONFIRMATION"
            or parent.get("training_clearance") is not False or parent.get("ppo_launch_clearance") is not False):
        raise ValueError("expected original frozen v2 development calibration")
    if set(parent.get("outputs",{})) != {"norm_only","features_v2"}:
        raise ValueError("both fixed experimental variants are required")
    parent_files = {variant: release_outputs({"outputs": outputs}, calibration_dir / variant,
                                            required=PARENT_FILES, observed=observed)
                    for variant, outputs in parent["outputs"].items()}
    code_bindings = {}
    if not set(CALIBRATION_CODE_FILES).issubset(parent.get("code_bindings") or {}):
        raise ValueError("calibration parent lacks the required live source bindings")
    for name in CODE_FILES:
        actual = identity(ROOT / name)
        if name in parent["code_bindings"]:
            bound_file(parent["code_bindings"][name], ROOT, expected=ROOT / name, observed=observed)
        observed[str((ROOT / name).resolve())] = actual
        code_bindings[name] = actual
    bank = json.loads(normalization_manifest.read_text())
    if (bank.get("schema_version") != "normalization-only-train-score-bank-v1"
            or bank.get("status") != "COMPLETE_NORMALIZATION_TRAIN_ONLY_NOT_GATE_OR_PPO_CLEARANCE"):
        raise ValueError("normalization source bank has not completed")
    if (bank.get("policy_optimizer_updates") != 0 or isinstance(bank.get("policy_optimizer_updates"), bool)
            or bank.get("ppo_started",False) is not False or bank.get("gold_used") is not False):
        raise ValueError("normalization source must come from frozen SFT with zero PPO updates")
    if any(normalization_manifest.parent.glob("FAILED*.json")):
        raise ValueError("failed normalization banks cannot be promoted")
    bank_files = release_outputs(bank, normalization_manifest.parent, required=BANK_FILES, observed=observed)
    source = bank_files["normalization_rows.jsonl"]
    rows = read_rows(source)
    assignments = read_rows(parent_files["norm_only"]["assignments.jsonl"])
    population = validate_normalization_rows(rows,assignments)
    source_review = verify_normalization_sources(normalization_manifest.parent, bank, bank_files, rows, assignments, observed)
    stats = fit_text_normalization_v2(rows)
    if set(stats["by_dataset"]) != DATASETS:
        raise ValueError("every dataset must retain valid step observations without resampling")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "started.json",{"parent":parent_identity,"normalization_bank":bank_identity,"policy_optimizer_updates":0})
    try:
        outputs={}
        for variant in ("norm_only","features_v2"):
            before_path=parent_files[variant]["gate.json"]
            before=json.loads(before_path.read_text())
            if before.get("training_clearance") is not False or before.get("independent_confirmation_clearance") is not False:
                raise ValueError("normalization-only replacement cannot inherit a claimed training clearance")
            artifact=deepcopy(before)
            artifact.pop("payload_sha256")
            artifact.update(experiment_id=before["experiment_id"]+"-REPRESENTATIVE-TEXT-V1",
                parent_artifact=identity(before_path), text_normalization_bank=bank_identity,
                text_normalization_rows=identity(source),
                normalization_population_revision=SCHEMA)
            norm=artifact["normalization"]
            norm.update(text_v2=stats,text_center=stats["text_center"],text_scale=stats["text_scale"],
                text_fit_population="frozen_representative_H40_W40_M40_train_question_bank")
            artifact["payload_sha256"]=canonical_sha256(artifact)
            # Only the documented normalization and provenance fields may vary.
            old_state=deepcopy(before);new_state=deepcopy(artifact)
            for state in (old_state,new_state):
                for key in ("payload_sha256","experiment_id","parent_artifact","text_normalization_bank","text_normalization_rows","normalization_population_revision"):
                    state.pop(key,None)
                for key in ("text_v2","text_center","text_scale","text_fit_population"):
                    state["normalization"].pop(key,None)
            if old_state != new_state:
                raise ValueError("text population revision changed another research variable")
            old_gate=SourceCreditGateV2.load(before_path,allow_unvalidated=True)
            gate=SourceCreditGateV2(artifact,mask=old_gate.mask,allow_unvalidated=True)
            candidates=read_rows(parent_files[variant]["candidates.jsonl"])
            if any(gate.predict(row["features"]) != old_gate.predict(row["features"]) for row in candidates):
                raise ValueError("text-only population revision changed alpha predictions")
            dest=output_dir / variant
            dest.mkdir()
            write_json(dest / "gate.json",artifact)
            report=json.loads(parent_files[variant]["report.json"].read_text())
            report.update(experiment_id=artifact["experiment_id"],normalization=norm,
                representative_text_population=population,normalization_bank=bank_identity,
                normalization_source_replay=source_review,
                alpha_predictions_graph_statistics_and_source_mask_exactly_unchanged=True)
            write_json(dest / "report.json",report)
            for name in ("candidates.jsonl","assignments.jsonl"):
                shutil.copyfile(parent_files[variant][name],dest / name)
                if identity(dest / name)["sha256"] != observed[str(parent_files[variant][name])]["sha256"]:
                    raise ValueError("diagnostic bank changed during population-only revision")
            outputs[variant]={name:identity(dest / name) for name in ("gate.json","report.json","candidates.jsonl","assignments.jsonl")}
        for path, binding in observed.items():
            if identity(Path(path)) != binding:
                raise ValueError("bound source/dependency changed during normalization replacement")
        for name, binding in code_bindings.items():
            snapshot = output_dir / "runtime_code" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, snapshot)
            if identity(snapshot)["sha256"] != binding["sha256"]:
                raise ValueError("executed source snapshot changed during publication")
        manifest={"schema_version":SCHEMA,"experiment_id":"SOURCE-CREDIT-V2-REPRESENTATIVE-TEXT-20260906-V1",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "status":"V2_REPRESENTATIVE_TEXT_REPAIRED_NOT_INDEPENDENT_CONFIRMATION",
            "parent_manifest":parent_identity,"normalization_bank":bank_identity,
            "source_credit_mask":parent["source_credit_mask"],"outputs":outputs,
            "normalization_population":population,"normalization":stats,"normalization_source_replay":source_review,
            "code_bindings":code_bindings,"source_bindings":observed,
            "training_clearance":False,"independent_confirmation_clearance":False,
            "ppo_launch_clearance":False,"policy_optimizer_updates":0,
            "alpha_predictions_graph_statistics_and_source_mask_exactly_unchanged":True}
        shutil.copyfile(output_dir / "runtime_code/scripts/train/rebind_source_credit_v2_text_population.py",output_dir / "rebind_script.executed.py")
        write_json(output_dir / "manifest.json",manifest)
        return manifest
    except Exception as exc:
        write_json(output_dir / "FAILED.json",{"error_type":type(exc).__name__,"error":str(exc),"policy_optimizer_updates":0})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir",type=Path,required=True)
    parser.add_argument("--normalization-manifest",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=rebind(args.calibration_dir.resolve(),args.normalization_manifest.resolve(),args.output_dir.resolve())
    print(json.dumps({key:result[key] for key in ("experiment_id","status","normalization_population","normalization")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
