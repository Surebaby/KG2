#!/usr/bin/env python
"""Build deterministic train-only QPEG sentence-schema adaptation data.

Gold supporting/decomposition annotations are used only for raw-train rows.
They select the exact evidence sentences and construct supervised citations;
evaluation graphs remain answer-free and are built separately.
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

from kgproweight.data.silver_dataset import SilverDatasetReader
from kgproweight.kg.qpeg import passage_sentences
from kgproweight.kg.question_kg import question_sha256
from kgproweight.utils.logging import dump_manifest


EXPERIMENT_ID = "QPEG-V4-TRAINONLY-SCHEMA-ADAPT-DATA-N2400-SEED42"
RELATION = "evidence sentence"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def _passage(passage_id: str, title: str, sentences: Sequence[str]) -> dict[str, Any]:
    clean_title = _clean(title)
    clean_sentences = [_clean(sentence) for sentence in sentences if _clean(sentence)]
    return {
        "id": passage_id,
        "title": clean_title,
        "contents": f"{clean_title}\n{' '.join(clean_sentences)}",
        "source": "raw_train_context",
    }


def _hotpot_or_2wiki(
    dataset: str, row: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]], list[str]]:
    metadata = row.get("metadata") or {}
    context = metadata.get("context") or {}
    titles = list(context.get("title") or [])
    sentence_key = "sentences" if dataset == "hotpotqa" else "content"
    contents = list(context.get(sentence_key) or [])
    passages = [
        _passage(f"train-context-{rank}", str(title), [str(value) for value in sentences])
        for rank, (title, sentences) in enumerate(zip(titles, contents))
    ]
    by_title = {
        _norm(title): (str(title), [str(value) for value in sentences])
        for title, sentences in zip(titles, contents)
    }
    support = metadata.get("supporting_facts") or {}
    edges: list[tuple[str, str, str]] = []
    conclusions: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for title, sentence_index in zip(support.get("title") or [], support.get("sent_id") or []):
        source = by_title.get(_norm(title))
        if source is None:
            continue
        canonical_title, sentences = source
        index = int(sentence_index)
        if not 0 <= index < len(sentences):
            continue
        sentence = _clean(sentences[index])
        edge = (_clean(canonical_title), RELATION, sentence)
        if not all(edge) or edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
        conclusions.append(sentence)
        if len(edges) == 4:
            break
    return passages, edges, conclusions


def _best_answer_sentence(text: str, answer: str) -> str:
    passage = {"contents": f"support\n{_clean(text)}"}
    sentences = passage_sentences(passage) or [_clean(text)]
    answer_norm = _norm(answer)
    for sentence in sentences:
        if answer_norm and answer_norm in _norm(sentence):
            return _clean(sentence)
    answer_tokens = set(answer_norm.split())
    return _clean(max(
        sentences,
        key=lambda sentence: (len(set(_norm(sentence).split()) & answer_tokens), -len(sentence)),
    ))


def _musique(
    row: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]], list[str]]:
    decomposition = (((row.get("metadata") or {}).get("metadata") or {}).get("question_decomposition") or [])
    passages: list[dict[str, Any]] = []
    edges: list[tuple[str, str, str]] = []
    conclusions: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for rank, step in enumerate(decomposition):
        support = step.get("support_paragraph") or {}
        title = _clean(support.get("title") or f"support-{rank}")
        text = _clean(support.get("paragraph_text"))
        if not text:
            continue
        passages.append(_passage(f"train-support-{support.get('idx', rank)}", title, [text]))
        sentence = _best_answer_sentence(text, _clean(step.get("answer")))
        edge = (title, RELATION, sentence)
        if not all(edge) or edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
        conclusions.append(_clean(step.get("answer")) or sentence)
        if len(edges) == 4:
            break
    return passages, edges, conclusions


def _citation(edge: Sequence[str]) -> str:
    return f"({edge[0]}, {edge[1]}, {edge[2]})"


def _steps(
    edges: Sequence[tuple[str, str, str]], conclusions: Sequence[str], answer: str, *, cite: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (edge, conclusion) in enumerate(zip(edges, conclusions), start=1):
        used = f"[{_citation(edge)}]" if cite else "[]"
        text = (
            f"Reasoning: The passage {edge[0]} provides the following evidence: {edge[2]}\n"
            f"Knowledge Used: {used}\n"
            f"Conclusion: {conclusion}"
        )
        rows.append({
            "index": index,
            "text": text,
            "label": 1.0 if cite else 0.0,
            "cited_triples": [list(edge)] if cite else [],
        })
    rows.append({
        "index": len(rows) + 1,
        "text": (
            "Reasoning: Combining the preceding passage evidence resolves the multi-hop question.\n"
            "Knowledge Used: []\n"
            f"Conclusion: {answer}"
        ),
        "label": 0.0,
        "cited_triples": [],
    })
    return rows


def build_trajectory(
    dataset: str, raw: Mapping[str, Any], *, graph: bool
) -> dict[str, Any]:
    answer_values = raw.get("golden_answers") or []
    answer = _clean(answer_values[0] if answer_values else "")
    if not answer:
        raise ValueError(f"{dataset}/{raw.get('id')}: missing train answer")
    if dataset in {"hotpotqa", "2wikimultihopqa"}:
        passages, edges, conclusions = _hotpot_or_2wiki(dataset, raw)
    else:
        passages, edges, conclusions = _musique(raw)
    if len(edges) < 2:
        raise ValueError(f"{dataset}/{raw.get('id')}: fewer than two usable support edges")
    qid = str(raw["id"])
    question = str(raw["question"]).strip()
    variant = "qpeg" if graph else "no_graph_replay"
    return {
        "qid": f"{qid}::{variant}",
        "question": question,
        "answer": answer,
        "dataset": dataset,
        "steps": _steps(edges, conclusions, answer, cite=graph),
        "kg_subgraph": [list(edge) for edge in edges] if graph else [],
        "retrieved_passages": passages,
        "accepted": True,
        "metadata": {
            "source_qid": qid,
            "source_question_sha256": question_sha256(question),
            "curriculum_variant": variant,
            "gold_train_only": True,
            "evidence_source": "raw_train_supporting_facts_or_decomposition",
            "teacher_api_used": False,
            "gold_answer": answer,
        },
        "teacher_model": "deterministic_train_gold_schema_adapter_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol_dir", type=Path,
        default=Path("outputs/audits/qpeg_v4_schema_adaptation_protocol_v1"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/silver_data/qpeg_v4_schema_adaptation_n2400_seed42_v2"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite adaptation data: {args.out}")
    args.out.mkdir(parents=True)
    protocol_path = args.protocol_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING_DATA_BUILD_OR_EVALUATION_RETRIEVAL":
        raise ValueError("unexpected QPEG-v4 protocol status")
    cohort_path = args.protocol_dir / "train.question_only.jsonl"
    frozen = _read_jsonl(cohort_path)
    if len(frozen) != 1800:
        raise ValueError(f"expected 1800 frozen train qids, got {len(frozen)}")

    by_dataset_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        by_dataset_qid[dataset] = {
            str(row["id"]): row for row in _read_jsonl(args.data_root / dataset / "train.jsonl")
        }

    graph_rows: list[dict[str, Any]] = []
    replay_candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in frozen:
        dataset = str(row["dataset"])
        raw = by_dataset_qid[dataset].get(str(row["qid"]))
        if raw is None:
            raise ValueError(f"missing raw train row: {row['question_key']}")
        if question_sha256(str(raw["question"]).strip()) != row["question_sha256"]:
            raise ValueError(f"question hash mismatch: {row['question_key']}")
        trajectory = build_trajectory(dataset, raw, graph=True)
        graph_rows.append(trajectory)
        rank = hashlib.sha256(f"42\0{row['question_key']}".encode()).hexdigest()
        replay_candidates[dataset].append((rank, raw))
        counters[dataset]["graph_rows"] += 1
        counters[dataset]["graph_edges"] += len(trajectory["kg_subgraph"])

    replay_rows: list[dict[str, Any]] = []
    for dataset, values in replay_candidates.items():
        selected = sorted(values, key=lambda item: item[0])[:200]
        replay_rows.extend(build_trajectory(dataset, raw, graph=False) for _, raw in selected)
        counters[dataset]["no_graph_replay_rows"] += len(selected)
    rows = graph_rows + replay_rows
    if len(rows) != 2400 or len({row["qid"] for row in rows}) != 2400:
        raise RuntimeError("adaptation curriculum row count or identity failed")
    if any(not 3 <= len(row["steps"]) <= 5 for row in rows):
        raise RuntimeError("adaptation trace step count outside [3,5]")

    silver_path = args.out / "silver_curriculum.jsonl"
    _write_jsonl(silver_path, rows)
    reader = SilverDatasetReader(silver_path, split=None)
    if len(reader.accepted()) != 2400:
        raise RuntimeError("SilverDatasetReader did not accept all adaptation rows")
    report = {
        "schema_version": "qpeg-v4-schema-adaptation-data-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_NOT_TRAINED",
        "counts": {
            "total": len(rows),
            "graph": len(graph_rows),
            "no_graph_replay": len(replay_rows),
            "by_dataset": {dataset: dict(counter) for dataset, counter in counters.items()},
        },
        "integrity": {
            "unique_training_row_id": True,
            "all_accepted": True,
            "steps_between_3_and_5": True,
            "graph_rows_cite_only_prompt_edges": all(
                set(tuple(value) for step in row["steps"] for value in step["cited_triples"])
                <= set(tuple(value) for value in row["kg_subgraph"])
                for row in graph_rows
            ),
            "no_graph_rows_have_empty_kg_and_citations": all(
                not row["kg_subgraph"] and not any(step["cited_triples"] for step in row["steps"])
                for row in replay_rows
            ),
            "gold_use": "raw train only; final answer and support/decomposition annotations",
            "teacher_api_calls": 0,
        },
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": _sha256(protocol_path)},
            "cohort": {"path": str(cohort_path), "sha256": _sha256(cohort_path)},
            "raw_train_sha256": protocol["raw_sha256"],
        },
        "outputs": {"silver": {"path": str(silver_path), "sha256": _sha256(silver_path)}},
        "scientific_boundary": "This is gold-derived raw-train schema supervision, not answer-free ProofKG generation and not a Teacher-generated silver corpus.",
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v4_schema_adaptation_data", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
