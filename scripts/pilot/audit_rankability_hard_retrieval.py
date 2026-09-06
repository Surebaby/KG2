#!/usr/bin/env python
"""Attribute retrieval-stage failures for rankability qids unsolved by SFT sampling.

This is a read-only, zero-retrieval audit over already-versioned artifacts.  A
"hard" qid is wrong under greedy decoding and under all K sampled candidates.
Supporting-title annotations are used only for diagnosis, never to build or
rerank retrieval inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from kgproweight.kg.entity_linker import passage_title
from kgproweight.utils.logging import artifact_identity, dump_manifest, prepare_new_run_dir


def _norm(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _title_hits(passages: Sequence[Mapping[str, Any]], supporting_titles: Sequence[str]) -> List[str]:
    seen = {_norm(passage_title(dict(p))) for p in passages}
    return [title for title in supporting_titles if _norm(title) in seen]


def _literal_hit(passages: Sequence[Mapping[str, Any]], answer: str) -> bool:
    needle = _norm(answer)
    if not needle or needle in {"yes", "no"}:
        return False
    haystack = " ".join(_norm(p.get("contents") or p.get("text") or "") for p in passages)
    return bool(re.search(rf"(?:^| ){re.escape(needle)}(?: |$)", haystack))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _by_qid(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid") or "")
        if not qid or qid in out:
            raise ValueError(f"Invalid or duplicate qid={qid!r}")
        out[qid] = dict(row)
    return out


def derive_hard_qids(rollouts: Sequence[Mapping[str, Any]]) -> List[str]:
    greedy: Dict[str, Mapping[str, Any]] = {}
    sampled: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rollouts:
        qid = str(row["qid"])
        if row["candidate_type"] == "greedy":
            greedy[qid] = row
        elif row["candidate_type"] == "sampled":
            sampled[qid].append(row)
    hard = [
        qid for qid in sorted(greedy)
        if float(greedy[qid]["em"]) == 0.0
        and sampled[qid]
        and all(float(row["em"]) == 0.0 for row in sampled[qid])
    ]
    return hard


def _stage_detail(report: Mapping[str, Any], stage: str) -> Dict[str, Dict[str, Any]]:
    rows = report["datasets"]["hotpotqa"]["details"][stage]
    return {str(row["qid"]): dict(row) for row in rows}


def _support_state(n_hits: int, n_support: int) -> str:
    if n_hits <= 0:
        return "none"
    if n_hits >= n_support:
        return "all"
    return "partial"


def build_attribution(
    *,
    rollouts: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Mapping[str, Any]],
    silver: Mapping[str, Mapping[str, Any]],
    hybrid: Mapping[str, Mapping[str, Any]],
    retrieval_report: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    hard_qids = derive_hard_qids(rollouts)
    if len(hard_qids) != 25:
        raise ValueError(f"Frozen protocol expected 25 greedy+K-all-wrong qids, got {len(hard_qids)}")
    stages = {
        name: _stage_detail(retrieval_report, name)
        for name in (
            "control_candidate", "bridge_v3_candidate",
            "control_rerank_15", "bridge_v3_rerank_15",
        )
    }
    results: List[Dict[str, Any]] = []
    for qid in hard_qids:
        if qid not in cohort or qid not in silver or qid not in hybrid:
            raise KeyError(f"qid={qid} missing from cohort/silver/hybrid inputs")
        base = stages["control_candidate"].get(qid)
        if base is None:
            raise KeyError(f"qid={qid} missing from frozen retrieval report")
        support = list(base["supporting_titles"])
        n_support = len(support)
        if n_support == 0:
            raise ValueError(f"qid={qid} has no supporting-title annotations")

        stage_counts = {
            name: int(stages[name][qid]["n_support_titles_hit"])
            for name in stages
        }
        stored_passages = list(silver[qid].get("retrieved_passages") or [])[:15]
        hybrid_passages = list(hybrid[qid]["retrieved_passages"])[:15]
        stored_hits = _title_hits(stored_passages, support)
        hybrid_hits = _title_hits(hybrid_passages, support)
        final_count = len(hybrid_hits)

        if final_count >= n_support:
            cause = "EVIDENCE_PRESENT_POLICY_UTILIZATION"
        elif stage_counts["bridge_v3_candidate"] >= n_support:
            cause = "RERANK_OR_FINAL_PACKING_LOSS"
        elif stage_counts["bridge_v3_candidate"] > stage_counts["control_candidate"]:
            cause = "BRIDGE_PARTIAL_GAIN_REMAINING_CANDIDATE_MISS"
        elif stage_counts["bridge_v3_candidate"] > 0:
            cause = "SECOND_SUPPORT_CANDIDATE_MISS"
        else:
            cause = "BOTH_SUPPORTS_CANDIDATE_MISS"

        results.append({
            "qid": qid,
            "question": cohort[qid]["question"],
            "gold_answer": cohort[qid]["gold_answer"],
            "answer_type": "yes_no" if _norm(cohort[qid]["gold_answer"]) in {"yes", "no"} else "extractive",
            "rankability_stratum": cohort[qid]["stratum"],
            "supporting_titles": support,
            "n_support_titles": n_support,
            "stored_prompt_support_hits": stored_hits,
            "hybrid_prompt_support_hits": hybrid_hits,
            "stored_prompt_support_state": _support_state(len(stored_hits), n_support),
            "hybrid_prompt_support_state": _support_state(final_count, n_support),
            "stored_prompt_gold_literal": _literal_hit(stored_passages, cohort[qid]["gold_answer"]),
            "hybrid_prompt_gold_literal": _literal_hit(hybrid_passages, cohort[qid]["gold_answer"]),
            "stage_support_hit_counts": stage_counts,
            "attribution": cause,
        })
    return results


def summarise(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def counts(key: str) -> Dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    stage_summary: Dict[str, Any] = {}
    for stage in (
        "control_candidate", "bridge_v3_candidate",
        "control_rerank_15", "bridge_v3_rerank_15",
    ):
        vals = [int(row["stage_support_hit_counts"][stage]) for row in rows]
        support_n = [int(row["n_support_titles"]) for row in rows]
        stage_summary[stage] = {
            "any_support": sum(v > 0 for v in vals),
            "all_support": sum(v >= n for v, n in zip(vals, support_n)),
            "support_title_micro_recall": sum(vals) / max(1, sum(support_n)),
        }
    return {
        "n_hard_qids": len(rows),
        "definition": "greedy EM=0 and every one of K=4 sampled candidates EM=0",
        "answer_types": counts("answer_type"),
        "rankability_strata": counts("rankability_stratum"),
        "attribution_counts": counts("attribution"),
        "stored_prompt": {
            "any_support": sum(row["stored_prompt_support_state"] != "none" for row in rows),
            "all_support": sum(row["stored_prompt_support_state"] == "all" for row in rows),
            "gold_literal_excluding_yes_no": sum(bool(row["stored_prompt_gold_literal"]) for row in rows),
        },
        "hybrid_prompt": {
            "any_support": sum(row["hybrid_prompt_support_state"] != "none" for row in rows),
            "all_support": sum(row["hybrid_prompt_support_state"] == "all" for row in rows),
            "gold_literal_excluding_yes_no": sum(bool(row["hybrid_prompt_gold_literal"]) for row in rows),
        },
        "retrieval_stages": stage_summary,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankability-dir", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--hybrid-overrides", required=True)
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank_dir = Path(args.rankability_dir)
    paths = {
        "rollouts": rank_dir / "rollouts.jsonl",
        "cohort": rank_dir / "cohort.jsonl",
        "rankability_manifest": rank_dir / "manifest.json",
        "silver": Path(args.silver),
        "hybrid_overrides": Path(args.hybrid_overrides),
        "retrieval_report": Path(args.retrieval_report),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    # Resolve and validate every input before reserving a new Experiment ID.
    rollouts = _read_jsonl(paths["rollouts"])
    cohort = _by_qid(_read_jsonl(paths["cohort"]))
    silver = _by_qid(_read_jsonl(paths["silver"]))
    hybrid = _by_qid(_read_jsonl(paths["hybrid_overrides"]))
    retrieval_report = json.loads(paths["retrieval_report"].read_text(encoding="utf-8"))
    rows = build_attribution(
        rollouts=rollouts, cohort=cohort, silver=silver,
        hybrid=hybrid, retrieval_report=retrieval_report,
    )
    summary = summarise(rows)
    run_record = {
        "phase": "rankability_hard_retrieval_attribution",
        "protocol": {
            "version": "rankability_hard25_retrieval_attribution_v1",
            "training": "none",
            "new_retrieval": "none; existing stage outputs only",
            "support_annotations_used_for": "diagnosis only",
            "hard_definition": summary["definition"],
        },
        "input_artifacts": {key: artifact_identity(path) for key, path in paths.items()},
    }
    out_dir, experiment_id = prepare_new_run_dir(
        args.output_dir, experiment_id=args.experiment_id, extra=run_record,
    )
    try:
        _write_jsonl(out_dir / "hard25_attribution.jsonl", rows)
        payload = {"experiment_id": experiment_id, **summary}
        (out_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        dump_manifest(out_dir, extra={**run_record, "experiment_id": experiment_id, "summary": payload})
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as exc:
        dump_manifest(
            out_dir,
            extra={**run_record, "experiment_id": experiment_id, "error": repr(exc), "traceback": traceback.format_exc()},
            status="FAILED",
        )
        raise


if __name__ == "__main__":
    main()
