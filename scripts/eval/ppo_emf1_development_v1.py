#!/usr/bin/env python
"""Freeze, score and select on the existing 150-question development cohort.

This is a fixed-context development proxy, not the canonical Scheme-A main
table.  The optional generate command performs greedy CUDA inference only;
no retrieval, training or confirmation evaluation is performed.  Gold aliases
are written separately from generation inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any, Callable

# Support both ``python -m scripts.eval...`` and direct script execution from a
# remote checkout without relying on an editable installation's ``scripts``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kgproweight.utils.flashrag_bootstrap import setup_flashrag

setup_flashrag()

from kgproweight.data.prompts import build_inference_messages
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.reward.proofkg_process import canonical_exact_match, canonical_token_f1
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


VERSION = "ppo-emf1-development-v1"
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
VIEWS = ("legacy", "no_graph")
ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "development": "outputs/audits/saeg_v1_evaluation_protocol_v1/development.question_only.jsonl",
    "source": "outputs/audits/qpeg_v4_schema_adaptation_development_inputs_v1/arm_no_graph.jsonl",
    "confirmation": "outputs/audits/saeg_v1_evaluation_protocol_v1/confirmation.question_only.jsonl",
    "canonical": "outputs/audits/saeg_v1_evaluation_protocol_v1/canonical_reporting.question_only.jsonl",
    "rollout": "data/silver_data/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42/prompt_groups.jsonl",
    "replay": "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/selection_records.jsonl",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def key(row: dict[str, Any]) -> str:
    dataset, qid = row.get("dataset"), row.get("qid")
    if dataset not in DATASETS or not isinstance(qid, str) or not qid:
        raise ValueError("every row requires a supported dataset and a nonempty string qid")
    return f"{dataset}::{qid}"


def identities(rows: list[dict[str, Any]]) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = {name: set() for name in ("qid", "question", "family")}
    for row in rows:
        key(row)
        question = row.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError("identity rows require a nonempty question")
        dataset = row["dataset"]
        result["qid"].add((dataset, row["qid"]))
        result["question"].add((dataset, hashlib.sha256(question.encode()).hexdigest()))
        result["family"].add((dataset, family_sha256(question)))
    return result


def input_hash(row: dict[str, Any]) -> str:
    return digest({k: v for k, v in row.items() if k != "input_sha256"})


def bind_base_model(base_model: Path, project_root: Path) -> dict[str, Any]:
    """Hash the exact local base weights/tokenizer; no model is loaded."""
    required = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"]
    for name in required:
        if not (base_model / name).is_file():
            raise ValueError(f"required local base-model file missing: {base_model / name}")
    index = json.loads((base_model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(Path(name).name != name for name in shards):
        raise ValueError("invalid local base-model shard index")
    optional = [name for name in ("generation_config.json", "special_tokens_map.json", "added_tokens.json")
                if (base_model / name).is_file()]
    files = {}
    for name in sorted(set(required + optional + shards)):
        bound = identity(base_model / name)
        files[name] = {**bound, "path": name, "origin_path": bound["path"]}
    return {"path": logical_path(base_model, project_root), "full_weight_files_hashed": True, "files": files}


def bind_tokenizer(directory: Path, project_root: Path = ROOT) -> dict[str, Any]:
    required = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    files = {}
    for name in required + [name for name in ("added_tokens.json",) if (directory / name).is_file()]:
        bound = identity(directory / name)
        files[name] = {**bound, "path": name, "origin_path": bound["path"]}
    return {"path": logical_path(directory, project_root), "files": files}


def make_renderer(base_model: Path) -> Callable:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    def render(messages):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
        return prompt, len(encoded["input_ids"])
    return render


def _new_directory(path: Path) -> None:
    # Even an empty pre-existing directory is refused: releases are append-only.
    path.mkdir(parents=True, exist_ok=False)


def _finish(path: Path, report: dict[str, Any], artifacts: list[str]) -> dict[str, Any]:
    write_json(path / "report.json", report)
    manifest = {
        "schema_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": report["status"],
        "experiment_id": report["experiment_id"],
        "outputs": {name: {**identity(path / name), "path": name,
                           "origin_path": str((path / name).resolve())}
                    for name in [*artifacts, "report.json"]},
        "implementation": identity(Path(__file__)),
        "training_started": False,
    }
    write_json(path / "manifest.json", manifest)
    return report


def _load_release(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != VERSION:
        raise ValueError("unexpected development release schema")
    for name, bound in manifest["outputs"].items():
        if Path(name).name != name or file_sha(directory / name) != bound["sha256"]:
            raise ValueError(f"release artifact hash mismatch: {name}")
    return json.loads((directory / "report.json").read_text()), manifest


def build_legacy_view(rows: list[dict[str, Any]], project_root: Path) -> tuple[dict[str, list], dict]:
    """Use the real canonical CPU KG method, including its offline fallback.

    Bypassing the pipeline constructor avoids generator/retriever creation.
    Only fields consumed by _build_legacy_kg_context are initialized below.
    """
    from kgproweight.kg.entity_linker import EntityLinker
    from kgproweight.kg.wikidata_retriever import _QA_RELATION_FILTER, WikidataSubgraphRetriever
    from kgproweight.pipeline.kg_proweight_pipeline import KGProWeightPipeline

    cache_dir = project_root / "indexes/kg_cache"
    index_path = cache_dir / "question_kg_index_v2.json"
    description_path = project_root / "indexes/entity_desc_index.json"
    cache_path = cache_dir / "kg_subgraph_cache.jsonl"
    for path in (index_path, description_path, cache_path):
        if not path.is_file():
            raise ValueError(f"required frozen offline KG asset missing: {path}")
    raw = json.loads(index_path.read_text())
    descriptions = json.loads(description_path.read_text())
    if not isinstance(raw, list) or not raw or not isinstance(descriptions, dict) or not descriptions:
        raise ValueError("invalid frozen legacy index/description schema")
    if any("builder_version" not in r for r in raw):
        raise ValueError("legacy view requires the canonical versioned v2 question index")
    pipe = KGProWeightPipeline.__new__(KGProWeightPipeline)
    pipe.max_kg_triples = 12
    pipe.max_mentions = 5
    pipe._kg_source_counts = {"index": 0, "fallback": 0, "empty": 0}
    pipe._q_kg_index = {
        r.get("question", r.get("q", "")): [(t["h"], t["r"], t["t"]) for t in r["triples"]]
        for r in raw
    }
    pipe.entity_linker = EntityLinker(cache_path=None, offline=True, entity_index_path=str(description_path))
    if pipe.entity_linker._entity_index != descriptions:
        raise ValueError("offline entity-description consumer did not load the frozen asset")
    pipe.kg_retriever = WikidataSubgraphRetriever(
        max_hops=2, max_neighbors=30, cache_dir=str(cache_dir), offline=True,
        relation_filter=_QA_RELATION_FILTER,
    )
    before = {str(p): identity(p) for p in (index_path, description_path, cache_path)}
    graphs = {}
    for row in rows:
        item = SimpleNamespace(id=row["qid"], question=row["question"], metadata={})
        graphs[key(row)] = [list(t) for t in pipe._build_legacy_kg_context(item, row["retrieved_passages"])]
    if any(file_sha(Path(p)) != value["sha256"] for p, value in before.items()):
        raise ValueError("offline graph construction unexpectedly changed a frozen asset")
    return graphs, {
        "method": "KGProWeightPipeline._build_legacy_kg_context",
        "offline": True, "entity_cache_path": None,
        "max_hops": 2, "max_neighbors": 30, "max_mentions": 5, "max_kg_triples": 12,
        "source_counts": pipe._kg_source_counts,
        "assets": before,
        "source_files": {
            str(p.relative_to(project_root)): identity(p)
            for p in [project_root / "kgproweight/pipeline/kg_proweight_pipeline.py",
                      project_root / "kgproweight/kg/entity_linker.py",
                      project_root / "kgproweight/kg/wikidata_retriever.py",
                      project_root / "kgproweight/kg/kg_filter.py"]
        },
    }


def _registry(path: Path, project_root: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text())
    candidates = document.get("candidates", [])
    seen = set()
    normalized = []
    for row in candidates:
        model_id = row.get("model_id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            raise ValueError("candidate registry requires unique nonempty model_id values")
        seen.add(model_id)
        checkpoint = Path(row["checkpoint_path"])
        checkpoint = (checkpoint if checkpoint.is_absolute() else project_root / checkpoint).resolve()
        step = row["training_step"]
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("training_step must be a nonnegative integer")
        is_sft = row.get("is_sft", False)
        if not isinstance(is_sft, bool) or (is_sft and step != 0):
            raise ValueError("SFT registration requires is_sft=true and training_step=0")
        model_file = checkpoint / "adapter_model.safetensors"
        config_file = checkpoint / "adapter_config.json"
        if is_sft and (not model_file.is_file() or not config_file.is_file()):
            raise ValueError("the registered SFT adapter must already exist")
        normalized.append({
            "model_id": model_id, "checkpoint_path": logical_path(checkpoint, project_root),
            "training_step": step, "is_sft": is_sft,
            "adapter_sha256_at_freeze": file_sha(model_file) if model_file.is_file() else None,
            "adapter_config_sha256_at_freeze": file_sha(config_file) if config_file.is_file() else None,
        })
    if len(normalized) < 2 or sum(c["is_sft"] for c in normalized) != 1:
        raise ValueError("registry must freeze exactly one SFT and at least one PPO candidate")
    if len({c["checkpoint_path"] for c in normalized}) != len(normalized):
        raise ValueError("candidate checkpoint paths must be distinct")
    return normalized


def logical_path(path: Path, project_root: Path) -> str:
    """Keep registered project paths portable across local/remote checkouts."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def prepare_bank(
    *, output_dir: Path, candidate_registry: Path, experiment_id: str,
    project_root: Path = ROOT, paths: dict[str, Path] | None = None,
    n_per_dataset: int = 50,
    legacy_builder: Callable | None = None,
    base_model: Path | None = None, tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("refusing to overwrite development release")
    paths = paths or {name: project_root / value for name, value in DEFAULTS.items()}
    documents = {name: read_rows(path) for name, path in paths.items()}
    development, source = documents["development"], documents["source"]
    required = {dataset: n_per_dataset for dataset in DATASETS}
    if dict(Counter(r["dataset"] for r in development)) != required:
        raise ValueError("development cohort must have the frozen equal per-dataset count")
    if any(r.get("role") != "development" for r in development + source):
        raise ValueError("only development-role identities and inputs can enter this bank")
    dev_identity = identities(development)
    if any(len(value) != len(development) for value in dev_identity.values()):
        raise ValueError("development qid/question/current-family identities must be unique")
    if len(source) != len(development) or identities(source) != dev_identity:
        raise ValueError("source/development qid/question/current-family join must be exact")
    overlaps = {}
    for name in ("confirmation", "canonical", "rollout", "replay"):
        other = identities(documents[name])
        overlaps[name] = {field: len(dev_identity[field] & other[field]) for field in other}
        if any(overlaps[name].values()):
            raise ValueError(f"development isolation failure against {name}: {overlaps[name]}")
    source = sorted(source, key=lambda r: (DATASETS.index(r["dataset"]), key(r)))
    labels = []
    for row in source:
        if len(row.get("retrieved_passages", [])) != 10 or row.get("kg_subgraph") != []:
            raise ValueError("parent input must contain exactly 10 passages and an empty graph")
        aliases = row.get("gold_answers")
        if not isinstance(aliases, list) or not aliases or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ValueError("source requires nonempty string gold aliases; never invent or filter labels")
        labels.append({"dataset": row["dataset"], "qid": row["qid"], "gold_answers": aliases})
    candidates = _registry(candidate_registry, project_root)
    base_model = base_model or project_root / "models/llama3-8b"
    base_identity = bind_base_model(base_model, project_root)
    tokenizer_identity = bind_tokenizer(tokenizer_path, project_root) if tokenizer_path else None
    render = make_renderer(tokenizer_path or base_model)
    graphs, legacy_provenance = (legacy_builder or build_legacy_view)(source, project_root)
    if set(graphs) != {key(r) for r in source}:
        raise ValueError("legacy graph identity join must be exact")
    prepared = {}
    for view in VIEWS:
        prepared[view] = []
        for row in source:
            graph = graphs[key(row)] if view == "legacy" else []
            messages = build_inference_messages(
                question=row["question"], retrieved_passages=row["retrieved_passages"],
                kg_triples=graph, top_k=10, max_kg_triples=12,
            )
            prompt, prompt_tokens = render(messages)
            if prompt_tokens > 6144:
                raise ValueError("frozen development prompt exceeds canonical 6144-token input cap; no silent truncation")
            entry = {
                "schema_version": VERSION, "role": "development", "view": view,
                "dataset": row["dataset"], "qid": row["qid"], "question_key": key(row),
                "question": row["question"],
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "current_family_sha256": family_sha256(row["question"]),
                "retrieved_passages": row["retrieved_passages"], "kg_subgraph": graph,
                "messages": messages, "prompt": prompt, "prompt_tokens": prompt_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            entry["input_sha256"] = input_hash(entry)
            prepared[view].append(entry)
    _new_directory(output_dir)
    for view in VIEWS:
        write_rows(output_dir / f"{view}.inputs.jsonl", prepared[view])
    write_rows(output_dir / "labels.jsonl", labels)
    write_json(output_dir / "candidate_registry.json", {"schema_version": VERSION, "candidates": candidates})
    report = {
        "schema_version": VERSION, "experiment_id": experiment_id,
        "status": "FROZEN_DEVELOPMENT_BANK_NOT_EVALUATED", "n": len(source),
        "by_dataset": required, "unique_current_families": len(dev_identity["family"]),
        "qid_order": [key(r) for r in source], "qid_order_sha256": digest([key(r) for r in source]),
        "isolation": overlaps, "sources": {name: identity(path) for name, path in paths.items()},
        "candidate_registry_source": identity(candidate_registry),
        "views": {view: {"file": f"{view}.inputs.jsonl", "n": len(prepared[view])} for view in VIEWS},
        "base_model": base_identity,
        **({"tokenizer": tokenizer_identity} if tokenizer_identity else {}),
        "generation": {"do_sample": False, "max_new_tokens": 512, "seed": 42,
                       "max_input_tokens": 6144, "batch_size": 1,
                       "temperature": None, "top_p": None, "top_k": 0,
                       "dtype": "bfloat16", "chat_template": ("frozen explicit tokenizer; add_generation_prompt=true" if tokenizer_identity else "frozen base tokenizer; add_generation_prompt=true")},
        "selection_rule": "legacy macro EM descending, macro F1 descending, SFT before PPO on exact ties, then earlier training_step and model_id",
        "required_selection_views": list(VIEWS), "primary_selection_view": "legacy",
        "legacy_runtime": legacy_provenance,
        "source_code": {"prompts": identity(project_root / "kgproweight/data/prompts.py"),
                        "family_function": hashlib.sha256(inspect.getsource(family_sha256).encode()).hexdigest()},
        "boundary": "Consumed development only; fixed canonical Step inputs, legacy primary and passage-only retention secondary. Frozen-context proxy, never canonical Scheme-A main-table results. No QPEG/SAEG graph/schema. Gold aliases are scorer-only; no training/confirmation authorization.",
    }
    return _finish(output_dir, report, ["legacy.inputs.jsonl", "no_graph.inputs.jsonl", "labels.jsonl", "candidate_registry.json"])


def score_predictions(
    *, bank_dir: Path, predictions: Path, model_id: str, checkpoint: Path,
    view: str, output_dir: Path, experiment_id: str, project_root: Path = ROOT,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("refusing to overwrite a development score release")
    bank, _ = _load_release(bank_dir)
    if bank["status"] != "FROZEN_DEVELOPMENT_BANK_NOT_EVALUATED" or view not in VIEWS:
        raise ValueError("unexpected bank/view")
    registry = json.loads((bank_dir / "candidate_registry.json").read_text())["candidates"]
    matches = [r for r in registry if r["model_id"] == model_id]
    if len(matches) != 1:
        raise ValueError("model is not in the frozen candidate registry")
    candidate = matches[0]
    checkpoint = checkpoint.resolve()
    if logical_path(checkpoint, project_root) != candidate["checkpoint_path"]:
        raise ValueError("checkpoint path differs from the frozen candidate registration")
    adapter_sha = file_sha(checkpoint / "adapter_model.safetensors")
    adapter_config_sha = file_sha(checkpoint / "adapter_config.json")
    if candidate["adapter_sha256_at_freeze"] not in (None, adapter_sha):
        raise ValueError("registered checkpoint bytes changed after freeze")
    if candidate["adapter_config_sha256_at_freeze"] not in (None, adapter_config_sha):
        raise ValueError("registered adapter config bytes changed after freeze")
    inputs = read_rows(bank_dir / f"{view}.inputs.jsonl")
    labels = read_rows(bank_dir / "labels.jsonl")
    rows = read_rows(predictions)
    if len(rows) != len(inputs) or [key(r) for r in rows] != bank["qid_order"]:
        raise ValueError("predictions qid/order/count mismatch")
    if [key(r) for r in inputs] != bank["qid_order"] or [key(r) for r in labels] != bank["qid_order"]:
        raise ValueError("bank input/label order differs from frozen order")
    scored = []
    for row, item, label in zip(rows, inputs, labels):
        if item.get("input_sha256") != input_hash(item):
            raise ValueError("bank row input hash is inconsistent")
        if row.get("input_sha256") != item["input_sha256"] or row.get("view") != view:
            raise ValueError("prediction model-input hash/view mismatch")
        if row.get("model_id") != model_id or row.get("adapter_sha256") != adapter_sha:
            raise ValueError("prediction model identity/hash mismatch")
        if row.get("adapter_config_sha256") != adapter_config_sha:
            raise ValueError("prediction adapter configuration hash mismatch")
        if row.get("bank_manifest_sha256") != file_sha(bank_dir / "manifest.json"):
            raise ValueError("prediction bank manifest hash mismatch")
        if row.get("generation_contract_sha256") != digest(bank["generation"]):
            raise ValueError("prediction generation contract hash mismatch")
        if row.get("base_model_identity_sha256") != digest(bank["base_model"]):
            raise ValueError("prediction base model/tokenizer identity hash mismatch")
        raw = row.get("raw_output")
        if not isinstance(raw, str):
            raise ValueError("predictions require a raw_output string (empty output remains a scored failure)")
        answer = extract_kg_proweight_answer(raw)
        # The canonical evaluator makes the same defensive second extraction.
        answer = extract_kg_proweight_answer(answer)
        aliases = label["gold_answers"]
        scored.append({
            "dataset": item["dataset"], "qid": item["qid"], "input_sha256": item["input_sha256"],
            "answer": answer, "em": max(canonical_exact_match(answer, gold) for gold in aliases),
            "f1": max(canonical_token_f1(answer, gold) for gold in aliases),
            "empty_output": not raw.strip(),
        })
    by_dataset = {}
    for dataset in DATASETS:
        subset = [r for r in scored if r["dataset"] == dataset]
        if len(subset) != bank["by_dataset"][dataset]:
            raise ValueError("scored dataset count differs from frozen bank")
        by_dataset[dataset] = {"n": len(subset), **{
            metric: sum(r[metric] for r in subset) / len(subset) for metric in ("em", "f1")}}
    report = {
        "schema_version": VERSION, "experiment_id": experiment_id,
        "status": "COMPLETE_DEVELOPMENT_SCORE_NOT_MAIN_TABLE", "model_id": model_id,
        "candidate": candidate, "adapter_sha256": adapter_sha,
        "adapter_config_sha256": adapter_config_sha, "view": view,
        "bank_manifest_sha256": file_sha(bank_dir / "manifest.json"),
        "prediction_source": identity(predictions), "n": len(scored), "by_dataset": by_dataset,
        "macro": {metric: sum(by_dataset[d][metric] for d in DATASETS) / len(DATASETS) for metric in ("em", "f1")},
        "scorer_source": {name: identity(ROOT / name) for name in
                          ("kgproweight/eval/pred_processing.py", "kgproweight/data/parsers.py", "kgproweight/reward/proofkg_process.py")},
        "boundary": "Development fixed-context proxy only; all registered questions remain in the denominator, including empty/incorrect outputs.",
    }
    _new_directory(output_dir)
    write_rows(output_dir / "scored.jsonl", scored)
    return _finish(output_dir, report, ["scored.jsonl"])


def generate_predictions(
    *, bank_dir: Path, model_id: str, checkpoint: Path, view: str,
    output_dir: Path, experiment_id: str, base_model: Path | None = None,
    project_root: Path = ROOT, device: str = "cuda:0", tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """Run a registered adapter on frozen gold-free inputs, never on labels."""
    if output_dir.exists():
        raise ValueError("refusing to overwrite generation output")
    bank, _ = _load_release(bank_dir)
    if bank["status"] != "FROZEN_DEVELOPMENT_BANK_NOT_EVALUATED" or view not in VIEWS:
        raise ValueError("unexpected development bank/view")
    candidates = json.loads((bank_dir / "candidate_registry.json").read_text())["candidates"]
    candidate = next((c for c in candidates if c["model_id"] == model_id), None)
    if candidate is None or logical_path(checkpoint, project_root) != candidate["checkpoint_path"]:
        raise ValueError("generation model/path is not a frozen candidate")
    adapter_sha = file_sha(checkpoint / "adapter_model.safetensors")
    adapter_config_sha = file_sha(checkpoint / "adapter_config.json")
    if candidate["adapter_sha256_at_freeze"] not in (None, adapter_sha):
        raise ValueError("registered adapter bytes changed")
    if candidate["adapter_config_sha256_at_freeze"] not in (None, adapter_config_sha):
        raise ValueError("registered adapter config changed")
    base_model = base_model or project_root / bank["base_model"]["path"]
    # An explicit local deployment path may live outside this checkout (or be
    # a symlink). Content hashes, not the machine-specific absolute path, are
    # authoritative for the base model and tokenizer.
    for name, bound in bank["base_model"]["files"].items():
        if file_sha(base_model / name) != bound["sha256"]:
            raise ValueError(f"generation base model/tokenizer hash differs: {name}")
    frozen_tokenizer = bank.get("tokenizer")
    tokenizer_path = tokenizer_path or (project_root / frozen_tokenizer["path"] if frozen_tokenizer else base_model)
    for name, bound in (frozen_tokenizer or bank["base_model"])["files"].items():
        if not frozen_tokenizer and not name.startswith(("tokenizer", "special_tokens", "added_tokens")):
            continue
        if file_sha(tokenizer_path / name) != bound["sha256"]:
            raise ValueError(f"generation tokenizer hash differs: {name}")
    import torch
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("development generation requires an available CUDA GPU; no automatic CPU fallback")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = read_rows(bank_dir / f"{view}.inputs.jsonl")
    if [key(row) for row in rows] != bank["qid_order"]:
        raise ValueError("generation bank qid/order mismatch")
    if any(row.get("input_sha256") != input_hash(row) for row in rows):
        raise ValueError("generation input hashes inconsistent")
    forbidden = {"answer", "answers", "gold_answers", "golden_answers", "gold_answer", "target", "labels"}
    if any(forbidden.intersection(row) for row in rows):
        raise ValueError("gold/target field found in generation input")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _new_directory(output_dir)
    start = {"schema_version": VERSION, "experiment_id": experiment_id,
             "status": "STARTED_DEVELOPMENT_GENERATION", "model_id": model_id, "view": view,
             "bank_manifest_sha256": file_sha(bank_dir / "manifest.json"), "adapter_sha256": adapter_sha,
             "adapter_config_sha256": adapter_config_sha}
    write_json(output_dir / "started.json", start)
    try:
        random.seed(bank["generation"]["seed"])
        torch.manual_seed(bank["generation"]["seed"])
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device)
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
        model.eval()
        prediction_path = output_dir / "predictions.jsonl"
        with prediction_path.open("x", encoding="utf-8") as handle:
            for index, row in enumerate(rows, 1):
                prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
                if prompt != row["prompt"] or hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]:
                    raise ValueError("runtime tokenizer/chat template differs from frozen prompt")
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                    truncation=False, return_attention_mask=True)
                if encoded["input_ids"].shape[1] != row["prompt_tokens"] or encoded["input_ids"].shape[1] > 6144:
                    raise ValueError("runtime input token count differs from frozen bank")
                encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
                with torch.inference_mode():
                    output = model.generate(
                        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                        max_new_tokens=512, do_sample=False, temperature=None, top_p=None, top_k=0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = output[0][encoded["input_ids"].shape[1]:]
                prediction = {
                    "dataset": row["dataset"], "qid": row["qid"], "view": view,
                    "input_sha256": row["input_sha256"], "model_id": model_id,
                    "adapter_sha256": adapter_sha, "bank_manifest_sha256": start["bank_manifest_sha256"],
                    "adapter_config_sha256": adapter_config_sha,
                    "generation_contract_sha256": digest(bank["generation"]),
                    "base_model_identity_sha256": digest(bank["base_model"]),
                    "raw_output": tokenizer.decode(response, skip_special_tokens=True),
                    "response_token_ids": response.detach().cpu().tolist(),
                    "prompt_tokens": row["prompt_tokens"], "response_tokens": int(response.shape[0]),
                }
                handle.write(canonical_json(prediction) + "\n")
                handle.flush()
                print(f"{model_id} {view} development {index}/{len(rows)}", flush=True)
        if (file_sha(checkpoint / "adapter_model.safetensors") != adapter_sha or
                file_sha(checkpoint / "adapter_config.json") != adapter_config_sha):
            raise ValueError("adapter changed during generation")
        report = {**start, "status": "COMPLETE_DEVELOPMENT_GENERATION_NOT_MAIN_TABLE", "n": len(rows),
                  "generation": bank["generation"], "base_model": bank["base_model"],
                  "device": device, "torch_version": torch.__version__,
                  "boundary": "Greedy development inference only; generator never reads labels.jsonl contents; no optimization."}
        return _finish(output_dir, report, ["predictions.jsonl", "started.json"])
    except Exception as exc:
        write_json(output_dir / "FAILED.json", {**start, "status": "FAILED_DEVELOPMENT_GENERATION_RETAINED",
                   "error_type": type(exc).__name__, "error": str(exc)})
        raise


def select_checkpoint(
    *, bank_dir: Path, score_dirs: list[Path], output_dir: Path, experiment_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("refusing to overwrite checkpoint selection")
    bank, _ = _load_release(bank_dir)
    registry = json.loads((bank_dir / "candidate_registry.json").read_text())["candidates"]
    bank_hash = file_sha(bank_dir / "manifest.json")
    reports = {}
    bindings = []
    for directory in score_dirs:
        report, _ = _load_release(directory)
        pair = (report.get("model_id"), report.get("view"))
        if report.get("status") != "COMPLETE_DEVELOPMENT_SCORE_NOT_MAIN_TABLE" or report.get("bank_manifest_sha256") != bank_hash:
            raise ValueError("selection accepts only scores from this frozen development bank")
        if pair in reports:
            raise ValueError("duplicate candidate/view scores")
        reports[pair] = report
        bindings.append(identity(directory / "manifest.json"))
    expected = {(r["model_id"], view) for r in registry for view in VIEWS}
    if set(reports) != expected:
        raise ValueError("selection requires exactly every frozen candidate in both development views")
    for candidate in registry:
        a, b = (reports[(candidate["model_id"], view)] for view in VIEWS)
        if a["candidate"] != candidate or b["candidate"] != candidate or a["adapter_sha256"] != b["adapter_sha256"]:
            raise ValueError("candidate registration/model bytes differ across development views")
        if a["adapter_config_sha256"] != b["adapter_config_sha256"]:
            raise ValueError("candidate adapter config differs across development views")
    ranked = sorted(registry, key=lambda c: (
        -reports[(c["model_id"], "legacy")]["macro"]["em"],
        -reports[(c["model_id"], "legacy")]["macro"]["f1"],
        not c["is_sft"], c["training_step"], c["model_id"],
    ))
    selected = ranked[0]
    sft = next(c for c in registry if c["is_sft"])
    report = {
        "schema_version": VERSION, "experiment_id": experiment_id,
        "status": "SELECTED_ON_DEVELOPMENT_ONLY", "selected": selected,
        "selected_adapter_sha256": reports[(selected["model_id"], "legacy")]["adapter_sha256"],
        "bank_manifest_sha256": bank_hash, "score_manifests": bindings,
        "selection_rule": bank["selection_rule"], "ranking": [c["model_id"] for c in ranked],
        "results": {c["model_id"]: {view: reports[(c["model_id"], view)]["macro"] for view in VIEWS} for c in registry},
        "selected_minus_sft": {view: {
            metric: reports[(selected["model_id"], view)]["macro"][metric] - reports[(sft["model_id"], view)]["macro"][metric]
            for metric in ("em", "f1")} for view in VIEWS},
        "boundary": "SFT can win. Selection is exploratory on previously consumed development data; no final/canonical/confirmation score was used. This does not authorize further training or establish a paper improvement.",
    }
    _new_directory(output_dir)
    return _finish(output_dir, report, [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--candidate-registry", type=Path, required=True)
    prepare.add_argument("--project-root", type=Path, default=ROOT)
    prepare.add_argument("--base-model", type=Path, default=None)
    prepare.add_argument("--tokenizer-path", type=Path, default=None)
    prepare.add_argument("--experiment-id", required=True)
    score = sub.add_parser("score")
    score.add_argument("--bank-dir", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--model-id", required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--view", choices=VIEWS, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--experiment-id", required=True)
    score.add_argument("--project-root", type=Path, default=ROOT)
    generate = sub.add_parser("generate")
    generate.add_argument("--bank-dir", type=Path, required=True)
    generate.add_argument("--model-id", required=True)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--view", choices=VIEWS, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--experiment-id", required=True)
    generate.add_argument("--base-model", type=Path, default=None)
    generate.add_argument("--tokenizer-path", type=Path, default=None)
    generate.add_argument("--project-root", type=Path, default=ROOT)
    generate.add_argument("--device", default="cuda:0")
    select = sub.add_parser("select")
    select.add_argument("--bank-dir", type=Path, required=True)
    select.add_argument("--score-dir", type=Path, action="append", required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--experiment-id", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        result = prepare_bank(**args)
    elif command == "score":
        result = score_predictions(**args)
    elif command == "generate":
        result = generate_predictions(**args)
    else:
        args["score_dirs"] = args.pop("score_dir")
        result = select_checkpoint(**args)
    print(json.dumps({"status": result["status"], "experiment_id": result["experiment_id"]}))


if __name__ == "__main__":
    main()
