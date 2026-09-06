#!/usr/bin/env python
"""Build train-only automatic Passage-QPEG quality labels and SFT candidates.

Retrieval and sentence selection are answer-free.  Raw-train Gold support is
joined only afterwards to classify selected evidence and build supervised
targets.  The output is a candidate pool, not a final sampled SFT schedule.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import joblib

from kgproweight.data.saeg_parsers import parse_saeg_steps
from kgproweight.kg.qpeg import passage_sentences
from kgproweight.kg.qpeg_sentence_selector import build_selected_sentence_record, sentence_candidates
from kgproweight.kg.question_kg import question_sha256
from kgproweight.kg.source_adaptive_evidence_graph import passages_sha256
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.build_qpeg_v4_schema_adaptation_data import _best_answer_sentence, _clean, _norm


EXPECTED_PROTOCOL_STATUS = "FROZEN_BEFORE_TRAIN_RETRIEVAL_DATA_BUILD_OR_MODEL_UPDATE"
EXPECTED_ADDENDUM_STATUS = "FROZEN_CORRECTION_BEFORE_DATA_CLASSIFICATION_OR_MODEL_UPDATE"
EXPERIMENT_ID = "SAEG-P-ALIGNMENT-V2-TRAIN1781-CANDIDATES-SEED42"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_key(title: object, sentence: object) -> tuple[str, str]:
    return _norm(title), _norm(sentence)


def required_support_units(dataset: str, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return ordered, de-duplicated raw-train support units."""
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if dataset in {"hotpotqa", "2wikimultihopqa"}:
        metadata = raw.get("metadata") or {}
        context = metadata.get("context") or {}
        titles = list(context.get("title") or [])
        sentence_key = "sentences" if dataset == "hotpotqa" else "content"
        contents = list(context.get(sentence_key) or [])
        by_title = {
            _norm(title): (str(title), [str(value) for value in sentences])
            for title, sentences in zip(titles, contents)
        }
        support = metadata.get("supporting_facts") or {}
        for hop_index, (title, sentence_index) in enumerate(
            zip(support.get("title") or [], support.get("sent_id") or []), start=1
        ):
            source = by_title.get(_norm(title))
            if source is None:
                continue
            canonical_title, sentences = source
            index = int(sentence_index)
            if not 0 <= index < len(sentences):
                continue
            sentence = _clean(sentences[index])
            key = support_key(canonical_title, sentence)
            if not all(key) or key in seen:
                continue
            seen.add(key)
            output.append({
                "hop_index": hop_index,
                "title": _clean(canonical_title),
                "sentence": sentence,
                "key": list(key),
            })
        return output

    decomposition = (((raw.get("metadata") or {}).get("metadata") or {}).get("question_decomposition") or [])
    for hop_index, step in enumerate(decomposition, start=1):
        support = step.get("support_paragraph") or {}
        title = _clean(support.get("title") or f"support-{hop_index}")
        text = _clean(support.get("paragraph_text"))
        if not text:
            continue
        sentence = _best_answer_sentence(text, _clean(step.get("answer")))
        key = support_key(title, sentence)
        if not all(key) or key in seen:
            continue
        seen.add(key)
        output.append({
            "hop_index": hop_index,
            "title": title,
            "sentence": sentence,
            "key": list(key),
        })
    return output


def quality_class(required_count: int, selected_count: int, matched_required_count: int) -> str:
    if required_count < 2:
        return "unresolved_gold"
    if selected_count == 0:
        return "empty"
    if matched_required_count == 0:
        return "misleading"
    if matched_required_count == required_count:
        return "complete"
    return "partial"


def automatic_passage_items(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = []
    for ordinal, edge in enumerate(graph.get("edges") or [], start=1):
        items.append({
            "passage_id": f"P{ordinal}",
            "title": str(edge["head_surface"]),
            "sentence": str(edge["tail_surface"]),
            "source_passage_id": str(edge["passage_id"]),
            "passage_rank": int(edge["passage_rank"]),
            "sentence_index": int(edge["sentence_index"]),
            "sentence_sha256": str(edge["sentence_sha256"]),
            "selector_score": float(edge["relevance_score"]),
            "construction_gold_access": False,
        })
    return items


def _replace_citations(text: str, passage_ids: Sequence[str]) -> str:
    knowledge_replaced, count = re.subn(
        r"(?im)^[ \t]*Knowledge Used\s*:[^\r\n]*$",
        "Knowledge Used: []",
        str(text).strip(),
        count=1,
    )
    if count != 1:
        raise ValueError("source step must contain exactly one Knowledge Used field")
    passage_field = f"Passage Used: [{', '.join(passage_ids)}]" if passage_ids else "Passage Used: []"
    existing, existing_count = re.subn(
        r"(?im)^[ \t]*Passage Used\s*:[^\r\n]*$",
        passage_field,
        knowledge_replaced,
        count=1,
    )
    if existing_count:
        return existing
    return knowledge_replaced.replace("Knowledge Used: []", f"Knowledge Used: []\n{passage_field}", 1)


def build_aligned_steps(
    source_steps: Sequence[Mapping[str, Any]], selected_id_by_key: Mapping[tuple[str, str], str]
) -> list[dict[str, Any]]:
    output = []
    for raw in source_steps:
        passage_ids = []
        for triple in raw.get("cited_triples") or []:
            if len(triple) != 3:
                continue
            passage_id = selected_id_by_key.get(support_key(triple[0], triple[2]))
            if passage_id:
                passage_ids.append(passage_id)
        passage_ids = list(dict.fromkeys(passage_ids))
        row = {
            "index": int(raw["index"]),
            "text": _replace_citations(str(raw["text"]), passage_ids),
            "label": 1.0 if passage_ids else 0.0,
            "cited_triples": [],
        }
        if passage_ids:
            row["cited_edge_ids"] = passage_ids
            row["cited_passage_ids"] = passage_ids
        output.append(row)
    return output


def assistant_trace(steps: Sequence[Mapping[str, Any]], answer: str) -> str:
    chunks = [f"[Step {index}]\n{step['text']}" for index, step in enumerate(steps, start=1)]
    chunks.append(f"[Final Answer]\n{answer}")
    return "\n\n".join(chunks)


def raw_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("id") or row.get("qid") or "")
            if qid in wanted:
                output[qid] = row
    if set(output) != wanted:
        raise ValueError(f"{path}: missing {len(wanted - set(output))} frozen qids")
    return output


def index(rows: Sequence[Mapping[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        key = str(raw[field])
        if key in output:
            raise ValueError(f"duplicate {label}: {key}")
        output[key] = dict(raw)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(
        "outputs/audits/saeg_p_hard_negative_alignment_v2_protocol/protocol.json"))
    parser.add_argument("--isolation_addendum", type=Path, default=Path(
        "outputs/audits/saeg_p_hard_negative_alignment_v2_isolation_addendum/isolation_addendum.json"))
    parser.add_argument("--effective_cohort", type=Path, default=Path(
        "outputs/audits/saeg_p_hard_negative_alignment_v2_isolation_addendum/effective_train.question_only.jsonl"))
    parser.add_argument("--retrieval", type=Path, default=Path(
        "outputs/audits/saeg_p_alignment_v2_train1800_retrieval/retrieval_contexts.jsonl"))
    parser.add_argument("--selector", type=Path, default=Path(
        "outputs/training/qpeg_v3_sentence_selector_n1000x3_seed42/selector.joblib"))
    parser.add_argument("--qpeg_silver", type=Path, default=Path(
        "data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2/silver_curriculum.jsonl"))
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path(
        "data/silver_data/saeg_p_alignment_v2_train1781_candidates_seed42"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite candidate data: {args.out}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    addendum = json.loads(args.isolation_addendum.read_text(encoding="utf-8"))
    if protocol.get("status") != EXPECTED_PROTOCOL_STATUS:
        raise ValueError("unexpected protocol status")
    if addendum.get("status") != EXPECTED_ADDENDUM_STATUS:
        raise ValueError("unexpected isolation addendum status")
    selector_sha256 = sha256_file(args.selector)
    if selector_sha256 != protocol["automatic_input_path"]["selector"]["sha256"]:
        raise ValueError("selector hash differs from frozen protocol")
    if sha256_file(args.effective_cohort) != addendum["outputs"]["effective_train"]["sha256"]:
        raise ValueError("effective cohort hash differs from isolation addendum")

    cohort = read_jsonl(args.effective_cohort)
    contexts_all = index(read_jsonl(args.retrieval), "question_key", "retrieval context")
    qpeg_rows = read_jsonl(args.qpeg_silver)
    source_by_key = {}
    for row in qpeg_rows:
        metadata = row.get("metadata") or {}
        if metadata.get("curriculum_variant") != "qpeg":
            continue
        key = f"{row['dataset']}::{metadata['source_qid']}"
        source_by_key[key] = row
    wanted: dict[str, set[str]] = defaultdict(set)
    for row in cohort:
        wanted[str(row["dataset"])].add(str(row["qid"]))
    raw_by_dataset = {
        dataset: raw_rows(args.data_root / dataset / "train.jsonl", qids)
        for dataset, qids in wanted.items()
    }
    model = joblib.load(args.selector)
    threshold = float(model["threshold"])
    max_edges = int(model["max_selected_edges"])
    if threshold != float(protocol["automatic_input_path"]["threshold"]) or max_edges != int(
        protocol["automatic_input_path"]["max_selected_edges"]
    ):
        raise ValueError("selector runtime parameters differ from frozen protocol")

    candidates_out: list[dict[str, Any]] = []
    quality_out: list[dict[str, Any]] = []
    graphs_out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    citation_errors = 0
    for frozen in cohort:
        dataset, qid = str(frozen["dataset"]), str(frozen["qid"])
        question_key_value = str(frozen["question_key"])
        context = contexts_all.get(question_key_value)
        source = source_by_key.get(question_key_value)
        raw = raw_by_dataset[dataset][qid]
        if context is None or source is None:
            raise ValueError(f"{question_key_value}: missing retrieval or Gold-source trajectory")
        if any(str(value["question_sha256"]) != str(frozen["question_sha256"]) for value in (context,)):
            raise ValueError(f"{question_key_value}: question hash mismatch")
        if question_sha256(str(raw["question"]).strip()) != frozen["question_sha256"]:
            raise ValueError(f"{question_key_value}: raw question hash mismatch")
        passages = list(context.get("passages") or [])
        if len(passages) != 10:
            raise ValueError(f"{question_key_value}: expected ten retrieved passages")
        graph = build_selected_sentence_record(
            dataset=dataset,
            qid=qid,
            question=str(frozen["question"]),
            passages=passages,
            vectorizer=model["vectorizer"],
            classifier=model["classifier"],
            threshold=threshold,
            max_edges=max_edges,
        )
        graph["role"] = "train_alignment_v2"
        graph["family_sha256"] = frozen["family_sha256"]
        graph["selector"]["model_sha256"] = selector_sha256
        units = required_support_units(dataset, raw)
        required_keys = {tuple(unit["key"]) for unit in units}
        all_context_keys = {
            support_key(candidate["head_surface"], candidate["tail_surface"])
            for candidate in sentence_candidates(
                dataset=dataset, question=str(frozen["question"]), passages=passages
            )
        }
        selected_items = automatic_passage_items(graph)
        selected_id_by_key = {
            support_key(item["title"], item["sentence"]): str(item["passage_id"])
            for item in selected_items
        }
        matched_required = required_keys & set(selected_id_by_key)
        context_matched_required = required_keys & all_context_keys
        qclass = quality_class(len(required_keys), len(selected_items), len(matched_required))
        selected_supported = sum(key in required_keys for key in selected_id_by_key)
        steps = build_aligned_steps(source.get("steps") or [], selected_id_by_key)
        answer = str(source["answer"]).strip()
        teacher_output = assistant_trace(steps, answer)
        parsed = parse_saeg_steps(
            teacher_output,
            known_kg=[],
            known_passage_ids=[str(item["passage_id"]) for item in selected_items],
        )
        valid_citations = bool(parsed) and all(step.citation_contract_valid for step in parsed)
        cited_ids = [value for step in parsed for value in step.cited_passage_ids]
        matched_ids = {selected_id_by_key[key] for key in matched_required}
        unexpected_citations = sorted(set(cited_ids) - matched_ids)
        if not valid_citations or unexpected_citations:
            citation_errors += 1
        trajectory = {
            "schema_version": "saeg-silver-trajectory-v2",
            "qid": f"{question_key_value}::P_AUTO_ALIGN_V2",
            "source_qid": qid,
            "question_key": question_key_value,
            "question": str(frozen["question"]),
            "answer": answer,
            "dataset": dataset,
            "evidence_mode": "P_ONLY" if selected_items else "N_REPLAY",
            "steps": steps,
            "kg_subgraph": [],
            "passage_evidence": selected_items,
            "retrieved_passages": passages,
            "accepted": qclass != "unresolved_gold",
            "metadata": {
                "experiment_id": EXPERIMENT_ID,
                "source_question_sha256": frozen["question_sha256"],
                "source_family_sha256": frozen["family_sha256"],
                "passages_sha256": passages_sha256(passages),
                "passage_quality_class": qclass,
                "required_support_units": len(required_keys),
                "required_units_in_full_context": len(context_matched_required),
                "required_units_selected": len(matched_required),
                "selected_edges": len(selected_items),
                "selected_supported_edges": selected_supported,
                "selected_edge_precision": selected_supported / max(1, len(selected_items)),
                "selector_support_recall": len(matched_required) / max(1, len(required_keys)),
                "full_context_support_recall": len(context_matched_required) / max(1, len(required_keys)),
                "gold_train_only": True,
                "evaluation_eligible": False,
                "teacher_api_used": False,
                "automatic_retrieval_gold_access": False,
                "automatic_selector_gold_access": False,
            },
            "teacher_output": teacher_output,
            "teacher_model": "deterministic_train_gold_selective_citation_adapter_v2",
        }
        quality_row = {
            "schema_version": "saeg-p-alignment-quality-record-v2",
            "question_key": question_key_value,
            "dataset": dataset,
            "qid": qid,
            "question_sha256": frozen["question_sha256"],
            "family_sha256": frozen["family_sha256"],
            "quality_class": qclass,
            "required_support_units": units,
            "required_units_in_full_context": sorted(list(key) for key in context_matched_required),
            "required_units_selected": sorted(list(key) for key in matched_required),
            "selected_edges": [
                {
                    "passage_id": item["passage_id"],
                    "title": item["title"],
                    "sentence": item["sentence"],
                    "selector_score": item["selector_score"],
                    "matches_required_support": support_key(item["title"], item["sentence"]) in required_keys,
                }
                for item in selected_items
            ],
            "evaluation_eligible": False,
        }
        candidates_out.append(trajectory)
        quality_out.append(quality_row)
        graphs_out.append(graph)
        counts[f"dataset::{dataset}"] += 1
        counts[f"class::{dataset}::{qclass}"] += 1
        counts[f"class::all::{qclass}"] += 1
        counts[f"selected::{dataset}"] += len(selected_items)
        counts[f"selected_supported::{dataset}"] += selected_supported
        counts[f"required::{dataset}"] += len(required_keys)
        counts[f"required_selected::{dataset}"] += len(matched_required)
        counts[f"required_in_context::{dataset}"] += len(context_matched_required)

    args.out.mkdir(parents=True, exist_ok=False)
    candidate_path = args.out / "silver_candidates.jsonl"
    quality_path = args.out / "evidence_quality.train_gold_only.jsonl"
    graph_path = args.out / "automatic_qpeg.answer_free.jsonl"
    write_jsonl(candidate_path, candidates_out)
    write_jsonl(quality_path, quality_out)
    write_jsonl(graph_path, graphs_out)
    per_dataset: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        classes = {
            name: counts[f"class::{dataset}::{name}"]
            for name in ("complete", "partial", "misleading", "empty", "unresolved_gold")
        }
        selected = counts[f"selected::{dataset}"]
        per_dataset[dataset] = {
            "rows": counts[f"dataset::{dataset}"],
            "classes": classes,
            "selected_edge_precision": counts[f"selected_supported::{dataset}"] / max(1, selected),
            "selector_required_unit_recall": counts[f"required_selected::{dataset}"] / max(
                1, counts[f"required::{dataset}"]
            ),
            "full_context_required_unit_recall": counts[f"required_in_context::{dataset}"] / max(
                1, counts[f"required::{dataset}"]
            ),
        }
        gates[f"{dataset}_complete_ge_20"] = classes["complete"] >= 20
        gates[f"{dataset}_partial_plus_misleading_ge_100"] = (
            classes["partial"] + classes["misleading"] >= 100
        )
    gates.update({
        "candidate_rows_exact_1781": len(candidates_out) == 1781,
        "unique_question_key": len({row["question_key"] for row in candidates_out}) == 1781,
        "quality_classification_rate_1": sum(
            counts[f"class::all::{name}"]
            for name in ("complete", "partial", "misleading", "empty", "unresolved_gold")
        ) == 1781,
        "selected_edge_target_citation_errors_0": citation_errors == 0,
        "all_evaluation_ineligible": all(row["metadata"]["evaluation_eligible"] is False for row in candidates_out),
    })
    status = "PASS_DATA_GATES_NOT_SAMPLED_NOT_TRAINED" if all(gates.values()) else "FAIL_STOP_DATA_GATES"
    report = {
        "schema_version": "saeg-p-alignment-v2-candidate-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "counts": dict(sorted(counts.items())),
        "per_dataset": per_dataset,
        "gates": {"checks": gates, "all_pass": all(gates.values())},
        "integrity": {
            "automatic_retrieval_and_selection_gold_access": False,
            "quality_labels_and_targets_use_train_gold": True,
            "citation_errors": citation_errors,
            "teacher_api_calls": 0,
            "final_schedule_materialized": False,
            "model_updates": 0,
        },
        "inputs": {
            "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
            "isolation_addendum": {"path": str(args.isolation_addendum), "sha256": sha256_file(args.isolation_addendum)},
            "effective_cohort": {"path": str(args.effective_cohort), "sha256": sha256_file(args.effective_cohort)},
            "retrieval": {"path": str(args.retrieval), "sha256": sha256_file(args.retrieval)},
            "selector": {"path": str(args.selector), "sha256": selector_sha256},
            "qpeg_silver": {"path": str(args.qpeg_silver), "sha256": sha256_file(args.qpeg_silver)},
        },
        "outputs": {
            "silver_candidates": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "quality_records": {"path": str(quality_path), "sha256": sha256_file(quality_path)},
            "automatic_qpeg": {"path": str(graph_path), "sha256": sha256_file(graph_path)},
        },
        "scientific_boundary": (
            "Candidate pool only. Passing data gates permits freezing a sampled continued-SFT schedule; it "
            "does not establish model utility or authorize training."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "saeg_p_alignment_v2_candidates", **report}, status=status)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
