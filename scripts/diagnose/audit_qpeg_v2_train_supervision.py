#!/usr/bin/env python
"""Audit whether train-only support annotations can supervise a QPEG edge selector."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from kgproweight.kg.qpeg import build_qpeg_record, passage_sentences, passage_title
from kgproweight.utils.logging import dump_manifest


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
EXPERIMENT_ID = "QPEG-V2-TRAIN-SUPERVISION-AUDIT-N1000X3-SEED42"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(value: object) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _hash_sample(path: Path, dataset: str, n: int) -> list[dict[str, Any]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("id") or "")
            rank = hashlib.sha256(f"42::{dataset}::{qid}".encode()).hexdigest()
            rows.append((rank, row))
    rows.sort(key=lambda value: value[0])
    return [row for _, row in rows[:n]]


def _context_and_supports(dataset: str, row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    metadata = row.get("metadata") or {}
    if dataset == "hotpotqa":
        context = metadata.get("context") or {}
        titles = list(context.get("title") or [])
        contents = list(context.get("sentences") or [])
        sf = metadata.get("supporting_facts") or {}
        supports = {
            (_norm(title), int(sent_id))
            for title, sent_id in zip(sf.get("title") or [], sf.get("sent_id") or [])
        }
        passages = [
            {"id": f"train-context-{index}", "contents": f'"{title}"\n' + " ".join(sentences)}
            for index, (title, sentences) in enumerate(zip(titles, contents))
        ]
        return passages, supports
    if dataset == "2wikimultihopqa":
        context = metadata.get("context") or {}
        titles = list(context.get("title") or [])
        contents = list(context.get("content") or [])
        sf = metadata.get("supporting_facts") or {}
        supports = {
            (_norm(title), int(sent_id))
            for title, sent_id in zip(sf.get("title") or [], sf.get("sent_id") or [])
        }
        passages = [
            {"id": f"train-context-{index}", "contents": f'"{title}"\n' + " ".join(sentences)}
            for index, (title, sentences) in enumerate(zip(titles, contents))
        ]
        return passages, supports

    decomposition = ((metadata.get("metadata") or {}).get("question_decomposition") or [])
    passages = []
    answers: dict[int, str] = {}
    seen: set[int] = set()
    for step in decomposition:
        support = step.get("support_paragraph") or {}
        paragraph_index = int(support.get("idx", step.get("paragraph_support_idx", len(passages))))
        if paragraph_index in seen:
            continue
        seen.add(paragraph_index)
        passages.append({
            "id": f"train-support-{paragraph_index}",
            "contents": f'"{support.get("title", "")}"\n{support.get("paragraph_text", "")}',
            "_paragraph_index": paragraph_index,
        })
        answers[len(passages) - 1] = _norm(step.get("answer"))
    return passages, answers


def _is_positive(dataset: str, edge: Mapping[str, Any], passages: list[Mapping[str, Any]], support: Any) -> bool:
    rank = int(edge["passage_rank"])
    sentence_index = int(edge["sentence_index"])
    if dataset in {"hotpotqa", "2wikimultihopqa"}:
        return (_norm(passage_title(passages[rank])), sentence_index) in support
    answer = support.get(rank, "")
    if not answer:
        return False
    sentences = passage_sentences(passages[rank])
    sentence = _norm(sentences[sentence_index]) if sentence_index < len(sentences) else ""
    return answer in sentence or answer in _norm(edge.get("tail_surface"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--n_per_dataset", type=int, default=1000)
    parser.add_argument(
        "--out", type=Path,
        default=Path("outputs/audits/qpeg_v2_train_supervision_n1000x3_seed42"),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite audit: {args.out}")
    args.out.mkdir(parents=True)

    detail_rows: list[dict[str, Any]] = []
    dataset_reports: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        path = args.data_root / dataset / "train.jsonl"
        input_hashes[str(path)] = _sha256(path)
        rows = _hash_sample(path, dataset, args.n_per_dataset)
        counters: Counter[str] = Counter()
        for row in rows:
            passages, support = _context_and_supports(dataset, row)
            if not passages:
                counters["no_context"] += 1
                continue
            graph = build_qpeg_record(
                dataset=dataset,
                qid=str(row["id"]),
                question=str(row["question"]),
                passages=passages,
            )
            labels = [_is_positive(dataset, edge, passages, support) for edge in graph["edges"]]
            positive = sum(labels)
            negative = len(labels) - positive
            counters["qid"] += 1
            counters["nonempty_qid"] += bool(graph["edges"])
            counters["positive_qid"] += positive > 0
            counters["edges"] += len(labels)
            counters["positive_edges"] += positive
            counters["negative_edges"] += negative
            detail_rows.append({
                "dataset": dataset,
                "qid": str(row["id"]),
                "question_sha256": hashlib.sha256(str(row["question"]).strip().encode()).hexdigest(),
                "edge_count": len(labels),
                "positive_edge_count": positive,
                "negative_edge_count": negative,
                "has_positive": positive > 0,
                "label_source": (
                    "train_supporting_fact_title_sentence"
                    if dataset != "musique" else "train_decomposition_support_answer_sentence"
                ),
            })
        n_qid = counters["qid"]
        n_edges = counters["edges"]
        dataset_reports[dataset] = {
            **dict(counters),
            "nonempty_rate": counters["nonempty_qid"] / max(1, n_qid),
            "qid_with_positive_rate": counters["positive_qid"] / max(1, n_qid),
            "edge_positive_rate": counters["positive_edges"] / max(1, n_edges),
        }
    _write_jsonl(args.out / "per_question.jsonl", detail_rows)
    total_positive = sum(value.get("positive_edges", 0) for value in dataset_reports.values())
    total_negative = sum(value.get("negative_edges", 0) for value in dataset_reports.values())
    gates = {
        "hotpot_positive_qid_rate_ge_0_50": dataset_reports["hotpotqa"]["qid_with_positive_rate"] >= 0.50,
        "2wiki_positive_qid_rate_ge_0_50": dataset_reports["2wikimultihopqa"]["qid_with_positive_rate"] >= 0.50,
        "musique_positive_qid_rate_ge_0_30": dataset_reports["musique"]["qid_with_positive_rate"] >= 0.30,
        "positive_edges_ge_1000": total_positive >= 1000,
        "negative_edges_ge_1000": total_negative >= 1000,
    }
    report = {
        "schema_version": "qpeg-v2-train-supervision-audit-v1",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SELECTOR_SUPERVISION_AVAILABLE" if all(gates.values()) else "FAIL_SELECTOR_SUPERVISION_INSUFFICIENT",
        "scope": "train-only support/decomposition labels; no dev pilot/confirmation/final labels",
        "selection": "sha256(42::dataset::qid), first n_per_dataset",
        "n_per_dataset": args.n_per_dataset,
        "datasets": dataset_reports,
        "totals": {"positive_edges": total_positive, "negative_edges": total_negative},
        "gates": {"checks": gates, "all_pass": all(gates.values())},
        "inputs": input_hashes,
        "scientific_boundary": "This audits label availability only; it does not establish selector accuracy or QPEG utility.",
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(args.out, extra={"phase": "qpeg_v2_train_supervision_audit", **report}, status=report["status"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
