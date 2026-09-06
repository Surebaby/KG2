#!/usr/bin/env python
"""Build an append-only Hotpot replay + 2Wiki Proof-KG SFT/PPO curriculum.

The 2Wiki arm is deliberately ``gold-derived, train-only``: evidence triples
and answers are used to create concise supervised traces.  It is a curriculum
asset, never an evaluation asset and never evidence that the automatic planner
retrieved the same proof.  The Hotpot arm is sampled unchanged from the existing
accepted train fold to reduce catastrophic forgetting.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kgproweight.data.silver_dataset import SilverDatasetReader, SilverTrajectory
from kgproweight.kg.question_kg import make_question_kg_record
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


DATASET = "2wikimultihopqa"
SCHEMA = "proofkg-gold-derived-train-curriculum-1"
TYPES = ("bridge_comparison", "comparison", "compositional", "inference")


def _score(seed: int, dataset: str, qid: str) -> int:
    value = f"{seed}\0{dataset}\0{qid}".encode("utf-8")
    return int(hashlib.sha256(value).hexdigest(), 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _push_lowest(
    heap: List[Tuple[int, str, Dict[str, Any]]],
    *,
    score: int,
    qid: str,
    row: Dict[str, Any],
    limit: int,
) -> None:
    item = (-score, qid, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _load_assignment_train_qids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("dataset") == DATASET and row.get("split") == "train":
                result.add(str(row["qid"]))
    return result


def _load_excluded_qids(root: Path) -> Tuple[set[str], List[str]]:
    result: set[str] = set()
    sources: List[str] = []
    for path in sorted(root.glob("**/cohort.jsonl")):
        used = False
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("dataset") not in (None, DATASET):
                    continue
                qid = str(row.get("qid") or row.get("source_id") or "").strip()
                if qid:
                    result.add(qid)
                    used = True
        if used:
            sources.append(str(path))
    return result, sources


def _evidence_triples(row: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    evidence = ((row.get("metadata") or {}).get("evidences") or {})
    heads = list(evidence.get("fact") or [])
    relations = list(evidence.get("relation") or [])
    tails = list(evidence.get("entity") or [])
    if not heads or not (len(heads) == len(relations) == len(tails)):
        raise ValueError(f"misaligned/empty evidence for qid={row.get('id')}")
    return [
        (str(h).strip(), str(r).strip(), str(t).strip())
        for h, r, t in zip(heads, relations, tails)
    ]


def _passages(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    context = ((row.get("metadata") or {}).get("context") or {})
    titles = list(context.get("title") or [])
    contents = list(context.get("content") or [])
    if len(titles) != len(contents):
        raise ValueError(f"misaligned context for qid={row.get('id')}")
    return [
        {
            "id": f"{row['id']}::{index}",
            "title": str(title),
            "contents": f"{title}\n{' '.join(str(x) for x in sentences)}",
            "source": "2wikimultihopqa_train_context",
        }
        for index, (title, sentences) in enumerate(zip(titles, contents))
    ]


def _step_text(index: int, triple: Sequence[str]) -> str:
    head, relation, tail = (str(x) for x in triple)
    rendered = f"({head}, {relation}, {tail})"
    return (
        f"Reasoning: The verified proof states that {head} has {relation} {tail}; "
        f"this supplies evidence hop {index}.\n"
        f"Knowledge Used: [{rendered}]\n"
        f"Conclusion: {head}'s {relation} is {tail}."
    )


def build_2wiki_trajectory(row: Mapping[str, Any]) -> SilverTrajectory:
    from kgproweight.data.silver_dataset import SilverStepRecord

    triples = _evidence_triples(row)
    answers = list(row.get("golden_answers") or [])
    if not answers or not str(answers[0]).strip():
        raise ValueError(f"missing gold answer for qid={row.get('id')}")
    answer = str(answers[0]).strip()
    steps = [
        SilverStepRecord(
            index=index,
            text=_step_text(index, triple),
            label=1.0,
            cited_triples=[tuple(triple)],
        )
        for index, triple in enumerate(triples, start=1)
    ]
    synth_index = len(steps) + 1
    steps.append(
        SilverStepRecord(
            index=synth_index,
            text=(
                "Reasoning: Combining the verified evidence hops resolves the "
                f"multi-hop question as {answer}.\n"
                "Knowledge Used: []\n"
                f"Conclusion: The answer is {answer}."
            ),
            label=0.0,
            cited_triples=[],
        )
    )
    trajectory = SilverTrajectory(
        qid=str(row["id"]),
        question=str(row["question"]),
        answer=answer,
        dataset=DATASET,
        steps=steps,
        kg_subgraph=triples,
        retrieved_passages=_passages(row),
        teacher_model="deterministic_gold_evidence_curriculum_v1",
        accepted=True,
        metadata={
            "gold_answer": answer,
            "question_type": str((row.get("metadata") or {}).get("type") or ""),
            "kg_bucket": "proof_complete_gold_derived",
            "curriculum_schema": SCHEMA,
            "gold_derived": True,
            "train_only": True,
            "evaluation_eligible": False,
            "source_split": "2wikimultihopqa/train",
        },
    )
    from kgproweight.training.phase3_sft import _render_assistant_trace

    trajectory.teacher_output = _render_assistant_trace(trajectory)
    return trajectory


def _select_2wiki(
    raw_path: Path,
    *,
    train_qids: set[str],
    excluded_qids: set[str],
    per_type: int,
    seed: int,
) -> List[SilverTrajectory]:
    heaps: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = defaultdict(list)
    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            qid = str(row.get("id") or "")
            qtype = str((row.get("metadata") or {}).get("type") or "")
            if qid not in train_qids or qid in excluded_qids or qtype not in TYPES:
                continue
            n_evidence = len((((row.get("metadata") or {}).get("evidences") or {}).get("fact") or []))
            if not 2 <= n_evidence <= 4:
                continue
            _push_lowest(
                heaps[qtype],
                score=_score(seed, DATASET, qid),
                qid=qid,
                row=row,
                limit=per_type,
            )
    missing = {name: per_type - len(heaps[name]) for name in TYPES if len(heaps[name]) < per_type}
    if missing:
        raise ValueError(f"insufficient eligible 2Wiki rows by type: {missing}")
    selected: List[SilverTrajectory] = []
    for qtype in TYPES:
        rows = [item[2] for item in sorted(heaps[qtype], key=lambda item: (-item[0], item[1]))]
        selected.extend(build_2wiki_trajectory(row) for row in rows)
    return selected


def _select_hotpot(path: Path, *, count: int, seed: int) -> List[SilverTrajectory]:
    reader = SilverDatasetReader(path, split="train")
    eligible = [
        traj for traj in reader.accepted()
        if traj.dataset == "hotpotqa" and str(traj.metadata.get("gold_answer") or "").strip()
    ]
    eligible.sort(key=lambda traj: (_score(seed, traj.dataset, traj.qid), traj.qid))
    if len(eligible) < count:
        raise ValueError(f"requested {count} Hotpot rows but only {len(eligible)} are eligible")
    return eligible[:count]


def _record_for(traj: SilverTrajectory) -> Dict[str, Any]:
    return make_question_kg_record(
        dataset=traj.dataset,
        qid=traj.qid,
        question=traj.question,
        triples=traj.kg_subgraph,
        provenance={
            "curriculum_schema": SCHEMA,
            "gold_derived": bool(traj.metadata.get("gold_derived", False)),
            "train_only": True,
            "source": (
                "2wiki_metadata_evidences"
                if traj.dataset == DATASET else "stored_quota70_hotpot_kg"
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_2wiki", default="data/2wikimultihopqa/train.jsonl")
    parser.add_argument(
        "--planner_assignments",
        default="data/silver_data/query_planner_supervision_split_v1_seed20260829/assignments.jsonl",
    )
    parser.add_argument(
        "--hotpot_silver",
        default="checkpoints/prm_alpha_gate_legacy_repaired_v2_quota70_hard_seed42_no_text_head/silver_with_logprobs.jsonl",
    )
    parser.add_argument("--audit_root", default="outputs/audits")
    parser.add_argument("--per_2wiki_type", type=int, default=1000)
    parser.add_argument("--hotpot_count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment_id",
        default="PROOFKG-CURRICULUM-MIX-V1-N8000-SEED42",
        help="Unique append-only experiment identifier recorded in the manifest.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/silver_data/proofkg_curriculum_mix_v1_n8000_seed42_20260829",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_2wiki)
    assignment_path = Path(args.planner_assignments)
    hotpot_path = Path(args.hotpot_silver)
    out_dir, experiment_id = prepare_new_run_dir(
        args.output_dir,
        experiment_id=args.experiment_id,
        extra={
            "phase": "data_build",
            "status_note": "gold-derived train-only curriculum; never evaluation",
        },
    )
    train_qids = _load_assignment_train_qids(assignment_path)
    excluded_qids, exclusion_sources = _load_excluded_qids(Path(args.audit_root))
    proof_rows = _select_2wiki(
        raw_path,
        train_qids=train_qids,
        excluded_qids=excluded_qids,
        per_type=args.per_2wiki_type,
        seed=args.seed,
    )
    hotpot_rows = _select_hotpot(hotpot_path, count=args.hotpot_count, seed=args.seed)
    rows = hotpot_rows + proof_rows
    rows.sort(key=lambda traj: (_score(args.seed + 1, traj.dataset, traj.qid), traj.dataset, traj.qid))

    silver_path = out_dir / "silver_curriculum.jsonl"
    SilverDatasetReader.write_jsonl(silver_path, rows)
    records_path = out_dir / "question_kg_records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for traj in rows:
            fh.write(json.dumps(_record_for(traj), ensure_ascii=False) + "\n")

    dataset_counts = Counter(traj.dataset for traj in rows)
    type_counts = Counter(
        str(traj.metadata.get("question_type") or "")
        for traj in proof_rows
    )
    report = {
        "experiment_id": experiment_id,
        "status": "COMPLETE_TRAIN_ONLY_NOT_EVALUATION",
        "schema_version": SCHEMA,
        "seed": args.seed,
        "selection": {
            "2wiki": "lowest SHA256(seed,dataset,qid), balanced by four question types",
            "hotpot": "lowest SHA256(seed,dataset,qid) from accepted quota70 train fold",
            "2wiki_evidence_hops": "2..4",
            "excluded_seen_cohort_qids": len(excluded_qids),
            "exclusion_sources": exclusion_sources,
        },
        "counts": {
            "total": len(rows),
            "by_dataset": dict(dataset_counts),
            "2wiki_by_type": dict(type_counts),
            "question_kg_records": len(rows),
            "nonempty_kg": sum(bool(traj.kg_subgraph) for traj in rows),
            "gold_derived_rows": len(proof_rows),
            "gold_derived_fraction": len(proof_rows) / len(rows),
        },
        "scientific_boundary": {
            "gold_derived_2wiki": True,
            "train_only": True,
            "evaluation_eligible": False,
            "automatic_planner_claim_allowed": False,
            "purpose": "teach KG utilization while replaying Hotpot SFT behavior",
        },
        "inputs": {
            "raw_2wiki": artifact_identity(raw_path),
            "planner_assignments": artifact_identity(assignment_path),
            "hotpot_silver": artifact_identity(hotpot_path),
        },
        "outputs": {
            "silver": {**artifact_identity(silver_path), "sha256": _sha256(silver_path)},
            "question_kg_records": {
                **artifact_identity(records_path),
                "sha256": _sha256(records_path),
            },
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dump_manifest(
        out_dir,
        extra={
            "experiment_id": experiment_id,
            "phase": "data_build",
            "report": artifact_identity(report_path),
            "silver": artifact_identity(silver_path),
            "question_kg_records": artifact_identity(records_path),
        },
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
