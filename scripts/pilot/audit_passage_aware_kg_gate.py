#!/usr/bin/env python
"""Audit structural and answer-coverage gates for fixed KG override arms.

Gold answers are used only for post-build evaluation.  They are never read by
the KG builders.  The audit is descriptive and does not modify any input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Tuple

from kgproweight.kg.kg_filter import filter_by_passage_support


Triple = Tuple[str, str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _triples(row: Dict[str, Any]) -> List[Triple]:
    result = []
    for value in row.get("kg_subgraph") or []:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            result.append(tuple(str(part) for part in value))
    return result


def _answer_hit(answer: str, triples: Iterable[Triple]) -> bool:
    target = _norm(answer)
    if not target or target in {"yes", "no"}:
        return False
    return any(target in _norm(part) for triple in triples for part in triple)


def _has_connected_pair(triples: List[Triple]) -> bool:
    nodes = [({_norm(triple[0]), _norm(triple[2])} - {""}) for triple in triples]
    return any(nodes[i] & nodes[j] for i in range(len(nodes)) for j in range(i + 1, len(nodes)))


def _read_arm(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("arm must be LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("arm label cannot be empty")
    return label.strip(), Path(raw_path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--arm", action="append", type=_read_arm, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--prompt_passages", type=int, default=15)
    parser.add_argument("--selection_jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    silver_path = Path(args.silver).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_path}")
    arms: Dict[str, Tuple[Path, List[Dict[str, Any]]]] = {}
    selected_qids = None
    if args.selection_jsonl:
        selection_path = Path(args.selection_jsonl).resolve()
        selection_rows = [
            json.loads(line) for line in selection_path.open(encoding="utf-8") if line.strip()
        ]
        selected_qids = [str(row.get("qid") or row.get("id") or "") for row in selection_rows]
        if not selected_qids or "" in selected_qids or len(set(selected_qids)) != len(selected_qids):
            raise SystemExit("selection JSONL requires unique non-empty qids")
    qids: set[str] | None = None
    for label, path in args.arm:
        if label in arms:
            raise SystemExit(f"duplicate arm label: {label}")
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if selected_qids is not None:
            all_rows = {str(row.get("qid") or ""): row for row in rows}
            missing = sorted(set(selected_qids) - set(all_rows))
            if missing:
                raise SystemExit(f"selected qids absent from arm {label}: {missing}")
            rows = [all_rows[qid] for qid in selected_qids]
        row_qids = {str(row.get("qid") or "") for row in rows}
        if "" in row_qids or len(row_qids) != len(rows):
            raise SystemExit(f"invalid or duplicate qids in arm {label}")
        if qids is None:
            qids = row_qids
        elif row_qids != qids:
            raise SystemExit(f"qid mismatch in arm {label}")
        arms[label] = (path, rows)
    assert qids is not None

    silver_by_qid: Dict[str, Dict[str, Any]] = {}
    with silver_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid") or row.get("id") or "")
            if qid in qids:
                silver_by_qid[qid] = row
    if set(silver_by_qid) != qids:
        raise SystemExit("some arm qids are absent from silver")

    summaries: Dict[str, Dict[str, Any]] = {}
    details: Dict[str, List[Dict[str, Any]]] = {}
    for label, (_, rows) in arms.items():
        counts: Counter[str] = Counter()
        arm_details = []
        for row in rows:
            qid = str(row["qid"])
            # A passage-only override intentionally omits ``kg_subgraph``;
            # under validate_sft semantics that means "keep stored silver KG".
            # An explicit empty list remains empty and is never replaced.
            kg_source = row if "kg_subgraph" in row else silver_by_qid[qid]
            triples = _triples(kg_source)
            passages = list(row.get("retrieved_passages") or [])[: args.prompt_passages]
            answer = str(silver_by_qid[qid].get("answer") or "")
            supported = filter_by_passage_support(triples, passages, min_entities=1)
            answer_hit = _answer_hit(answer, triples)
            connected = _has_connected_pair(triples)
            counts["questions"] += 1
            counts["empty"] += int(not triples)
            counts["triples"] += len(triples)
            counts["passage_supported_triples"] += len(supported)
            counts["answer_hit_questions"] += int(answer_hit)
            counts["connected_questions"] += int(connected)
            arm_details.append(
                {
                    "qid": qid,
                    "kg_source": "arm" if "kg_subgraph" in row else "stored_silver",
                    "n_triples": len(triples),
                    "n_passage_supported_triples": len(supported),
                    "answer_hit": answer_hit,
                    "has_connected_pair": connected,
                }
            )
        summaries[label] = {
            "counts": dict(counts),
            "empty_rate": counts["empty"] / max(1, counts["questions"]),
            "mean_triples": counts["triples"] / max(1, counts["questions"]),
            "passage_supported_triple_rate": counts["passage_supported_triples"]
            / max(1, counts["triples"]),
            "answer_hit_rate": counts["answer_hit_questions"] / max(1, counts["questions"]),
            "connected_question_rate": counts["connected_questions"]
            / max(1, counts["questions"]),
        }
        details[label] = arm_details

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "status": "COMPLETE_STRUCTURAL_ONLY",
        "protocol": {
            "training": "none",
            "model_inference": "none",
            "gold_used_for_build": False,
            "gold_used_for_post_build_answer_hit_audit": True,
            "answer_hit_excludes_yes_no": True,
            "passage_support": "at least one triple endpoint appears in prompt passages",
            "prompt_passages": args.prompt_passages,
            "selection_jsonl": args.selection_jsonl,
        },
        "inputs": {
            "silver": {"path": str(silver_path), "sha256": _sha256(silver_path)},
            "arms": {
                label: {"path": str(path), "sha256": _sha256(path)}
                for label, (path, _) in arms.items()
            },
        },
        "summaries": summaries,
        "details": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
