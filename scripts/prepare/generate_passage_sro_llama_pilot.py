#!/usr/bin/env python
"""Generate and fail-closed validate local-model passage SRO edges."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import torch

from kgproweight.kg.passage_sro import (
    ALLOWED_RELATIONS,
    PASSAGE_SRO_VALIDATOR_VERSION,
    parse_extraction_json,
    triples_from_edges,
    validate_extracted_edges,
)
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.utils.logging import dump_manifest, prepare_new_run_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(row: Mapping[str, Any]) -> list[dict[str, str]]:
    passages = []
    for index, passage in enumerate(row["passages"], start=1):
        text = str(passage.get("contents") or passage.get("text") or "").strip()
        passages.append(f"[Passage {index}]\n{text}")
    relations = ", ".join(sorted(ALLOWED_RELATIONS))
    instruction = f"""Extract at most 4 question-relevant factual edges explicitly stated in the passages.

Return JSON only: {{"edges":[{{"head":"...","relation":"...","tail":"...","passage_rank":1,"evidence_quote":"...","relation_trigger":"..."}}]}}.

Rules:
- A triple is (head entity, canonical relation, tail entity/literal), never a whole sentence in the relation field.
- relation must be exactly one of: {relations}
- evidence_quote must be a verbatim contiguous quote from the numbered passage.
- tail and relation_trigger must be verbatim spans inside evidence_quote.
- head must be a verbatim span inside evidence_quote, or exactly the title on the first line of that passage when the quote uses a pronoun.
- Extract only explicit facts. Do not infer, use outside knowledge, answer the question, or invent a relation.
- Prefer edges that form the shortest useful multi-hop chain. If none are explicit, return {{"edges":[]}}.

Question: {row['question']}

""" + "\n\n".join(passages)
    return [
        {"role": "system", "content": "You are a conservative information extraction system. Output valid JSON only."},
        {"role": "user", "content": instruction},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cohort_path = Path(protocol["inputs"]["cohort"]["path"])
    if _sha256(cohort_path) != protocol["inputs"]["cohort"]["sha256"]:
        raise SystemExit("cohort hash differs from frozen protocol")
    model_path = Path(protocol["extractor"]["base_model"])
    if _sha256(model_path / "config.json") != protocol["extractor"]["config_sha256"]:
        raise SystemExit("extractor config hash mismatch")
    if _sha256(model_path / "model.safetensors.index.json") != protocol["extractor"]["model_index_sha256"]:
        raise SystemExit("extractor weight index hash mismatch")

    run_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=protocol["experiment_id"] + "-EXTRACTION",
        extra={"protocol_sha256": _sha256(args.protocol), "phase": "passage_sro_llama_extraction"},
    )
    raw_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed = int(protocol["extractor"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        counters: dict[str, Counter[str]] = {
            dataset: Counter() for dataset in protocol["datasets"]
        }
        rows = _read_jsonl(cohort_path)
        for index, row in enumerate(rows, start=1):
            messages = _prompt(row)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            runtime_error = None
            generation = ""
            parsed: list[dict[str, Any]] = []
            accepted: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            try:
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=int(protocol["extractor"]["max_new_tokens"]),
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generation = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
                )
                parsed = parse_extraction_json(generation)
                accepted, rejected = validate_extracted_edges(
                    parsed, row["passages"], max_edges=int(protocol["extractor"]["max_edges"])
                )
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"

            triples = triples_from_edges(accepted)
            record = make_question_kg_record(
                dataset=str(row["dataset"]), qid=str(row["qid"]), question=str(row["question"]),
                triples=triples,
                provenance={
                    "builder_version": "passage-sro-llama-1",
                    "validator_version": PASSAGE_SRO_VALIDATOR_VERSION,
                    "source": "frozen_retrieved_passages",
                    "gold_access": False,
                    "network_access": False,
                    "all_accepted_edges_exact_span_validated": True,
                },
            )
            raw_rows.append({
                "dataset": row["dataset"], "qid": row["qid"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": int(encoded["input_ids"].shape[1]),
                "generation": generation,
            })
            details.append({
                **record, "accepted_edges": accepted, "rejected_edges": rejected,
                "raw_edge_count": len(parsed), "runtime_error": runtime_error,
            })
            records.append(record)
            count = counters[str(row["dataset"])]
            count["n"] += 1
            count["parse_valid"] += int(runtime_error is None)
            count["nonempty"] += int(bool(triples))
            count["accepted_edges"] += len(accepted)
            count["rejected_edges"] += len(rejected)
            count["runtime_errors"] += int(runtime_error is not None)
            print(f"passage SRO extraction {index}/{len(rows)} {row['dataset']}::{row['qid']} accepted={len(accepted)}", flush=True)

        raw_path = run_dir / "raw_generations.jsonl"
        record_path = run_dir / "question_kg_records.jsonl"
        detail_path = run_dir / "runtime_details.jsonl"
        _write_jsonl(raw_path, raw_rows)
        _write_jsonl(record_path, records)
        _write_jsonl(detail_path, details)
        by_dataset: dict[str, Any] = {}
        structural_pass = True
        for dataset, count in counters.items():
            n = count["n"]
            metrics = {
                **dict(count),
                "parse_rate": count["parse_valid"] / max(1, n),
                "nonempty_rate": count["nonempty"] / max(1, n),
            }
            metrics["gates"] = {
                "parse": metrics["parse_rate"] >= protocol["structural_gates"]["per_dataset_parse_rate_min"],
                "nonempty": metrics["nonempty_rate"] >= protocol["structural_gates"]["per_dataset_nonempty_rate_min"],
                "runtime": count["runtime_errors"] == 0,
            }
            structural_pass = structural_pass and all(metrics["gates"].values())
            by_dataset[dataset] = metrics
        report = {
            "schema_version": "passage-sro-llama-extraction-report-1",
            "experiment_id": experiment_id,
            "status": "PASS_STRUCTURE_READY_FOR_UTILITY" if structural_pass else "FAIL_STOP_STRUCTURE",
            "development_only": True,
            "gold_access": False,
            "network_access": False,
            "protocol": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
            "by_dataset": by_dataset,
            "structural_gate_pass": structural_pass,
            "outputs": {
                "raw_generations": {"path": str(raw_path), "sha256": _sha256(raw_path)},
                "question_kg_records": {"path": str(record_path), "sha256": _sha256(record_path)},
                "runtime_details": {"path": str(detail_path), "sha256": _sha256(detail_path)},
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dump_manifest(run_dir, status=report["status"], extra={
            "experiment_id": experiment_id, "phase": "passage_sro_llama_extraction",
            "report_sha256": _sha256(report_path),
        })
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        dump_manifest(run_dir, status="FAILED_RUNTIME", extra={
            "experiment_id": experiment_id,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        })
        raise


if __name__ == "__main__":
    main()
