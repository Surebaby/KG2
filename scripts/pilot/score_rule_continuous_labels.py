#!/usr/bin/env python
"""Produce rule_continuous_v1 candidate labels without replacing silver labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import string
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from kgproweight.kg.kg_filter import _pid_for_triple, _relation_question_score
from kgproweight.reward.prm_annotator import _ABSTENTION_RE
from kgproweight.utils.logging import dump_manifest


Triple = Tuple[str, str, str]
_PUNCT = str.maketrans("", "", string.punctuation)
_STOP = frozenset(
    "the a an of in on at to for and or is are was were be been being that this these "
    "those which who whom whose what when where why how from with as by into over".split()
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triple(value: Sequence[Any]) -> Triple:
    return tuple(str(part).strip() for part in value)  # type: ignore[return-value]


def _key(value: Sequence[Any]) -> Triple:
    return tuple(" ".join(str(part).casefold().split()) for part in value)  # type: ignore[return-value]


def _tokens(text: str) -> List[str]:
    normalised = text.casefold().translate(_PUNCT)
    return [token for token in normalised.split() if token not in _STOP]


def _surface_support(surface: str, text: str) -> float:
    """Continuous token recall with an exact-phrase ceiling of one."""
    surface_norm = " ".join(surface.casefold().translate(_PUNCT).split())
    text_norm = " ".join(text.casefold().translate(_PUNCT).split())
    if not surface_norm:
        return 0.0
    if re.search(rf"\b{re.escape(surface_norm)}\b", text_norm):
        return 1.0
    surface_tokens = set(_tokens(surface))
    if not surface_tokens:
        return 0.0
    return len(surface_tokens & set(_tokens(text))) / len(surface_tokens)


def _reasoning_without_citation_echo(text: str) -> str:
    return re.sub(
        r"\n?\s*Knowledge Used\s*:.*?(?=\n\s*Conclusion\s*:|\Z)",
        "\n",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _passage_text(passages: Iterable[Any]) -> str:
    blocks: List[str] = []
    for passage in passages:
        if isinstance(passage, dict):
            blocks.append(str(passage.get("contents") or passage.get("text") or ""))
        else:
            blocks.append(str(passage))
    return " ".join(blocks)


def _numeric_consistency(tail: str, step_text: str, head_support: float) -> float:
    """Reject only explicit numeric conflicts in a step that mentions the head."""
    if head_support <= 0.0:
        return 1.0
    tail_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", tail))
    if not tail_numbers:
        return 1.0
    step_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", step_text))
    if step_numbers and not (tail_numbers & step_numbers):
        return 0.0
    return 1.0


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0}

    def q(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": median(ordered),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": ordered[-1],
        "mean": mean(ordered),
        "n_negative": sum(value < 0 for value in ordered),
        "n_zero": sum(value == 0 for value in ordered),
        "n_fractional_positive": sum(0 < value < 1 for value in ordered),
        "n_one": sum(value == 1 for value in ordered),
        "unique_rounded_4dp": len({round(value, 4) for value in ordered}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected", type=int, default=90)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != args.expected:
        raise SystemExit(f"input count {len(rows)} != expected {args.expected}")

    old_labels: List[float] = []
    new_labels: List[float] = []
    by_dataset: Dict[str, List[float]] = defaultdict(list)
    transition = Counter()
    component_rows: List[Dict[str, Any]] = []
    output_rows: List[Dict[str, Any]] = []
    invalid_citations = 0

    for row in rows:
        kg = [_triple(value) for value in row.get("kg_subgraph") or [] if len(value) == 3]
        kg_keys = {_key(value) for value in kg}
        question = str(row.get("question") or "")
        passages = _passage_text(row.get("retrieved_passages") or [])
        out_row = json.loads(json.dumps(row))
        for step_pos, (step, out_step) in enumerate(zip(row.get("steps") or [], out_row.get("steps") or [])):
            old = float(step.get("label") or 0.0)
            cited = [_triple(value) for value in step.get("cited_triples") or [] if len(value) == 3]
            step_text = _reasoning_without_citation_echo(str(step.get("text") or ""))
            utilities: List[float] = []
            components: List[Dict[str, Any]] = []
            for triple in cited:
                head, relation, tail = triple
                accuracy = float(_key(triple) in kg_keys)
                h_step = _surface_support(head, step_text)
                r_step = _surface_support(relation, step_text)
                t_step = _surface_support(tail, step_text)
                faithfulness = 0.4 * h_step + 0.2 * r_step + 0.4 * t_step
                pid = _pid_for_triple(triple)
                relation_intent = max(
                    _relation_question_score(pid, question.casefold()),
                    _surface_support(relation, question),
                )
                entity_anchor = max(
                    _surface_support(head, question), _surface_support(tail, question)
                )
                question_relevance = 0.6 * relation_intent + 0.4 * entity_anchor
                passage_grounding = 0.5 * _surface_support(head, passages) + 0.5 * _surface_support(tail, passages)
                numeric_consistency = _numeric_consistency(tail, step_text, h_step)
                utility = (
                    accuracy
                    * math.sqrt(max(0.0, faithfulness * question_relevance))
                    * (0.5 + 0.5 * passage_grounding)
                    * numeric_consistency
                )
                utility = min(1.0, max(0.0, utility))
                if not accuracy:
                    invalid_citations += 1
                utilities.append(utility)
                components.append({
                    "triple": list(triple),
                    "pid": pid,
                    "accuracy": accuracy,
                    "faithfulness": faithfulness,
                    "question_relevance": question_relevance,
                    "passage_grounding": passage_grounding,
                    "numeric_consistency": numeric_consistency,
                    "utility": utility,
                })

            if old < 0:
                new = -1.0
                branch = "verified_contradiction_preserved"
            elif not cited:
                new = 0.0
                branch = "no_citation"
            elif _ABSTENTION_RE.search(step_text):
                new = 0.0
                branch = "honest_abstention"
            elif not all(component["accuracy"] == 1.0 for component in components):
                new = 0.0
                branch = "invalid_citation_zero_and_reject"
            else:
                new = sum(utilities) / len(utilities)
                branch = "verified_citation_rule_score"

            out_step["rule_continuous_v1"] = float(new)
            out_step["rule_continuous_v1_branch"] = branch
            out_step["rule_continuous_v1_components"] = components
            old_labels.append(old)
            new_labels.append(new)
            by_dataset[str(row.get("dataset") or "UNKNOWN")].append(new)
            transition[(str(old), "neg" if new < 0 else "zero" if new == 0 else "frac" if new < 1 else "one")] += 1
            component_rows.append({
                "qid": row.get("qid"),
                "dataset": row.get("dataset"),
                "step_position": step_pos,
                "step_index": step.get("index"),
                "old_label": old,
                "rule_continuous_v1": new,
                "branch": branch,
                "components": components,
            })
        output_rows.append(out_row)

    candidate_path = output_dir / "silver_with_rule_continuous_v1.candidate.jsonl"
    with candidate_path.open("w", encoding="utf-8") as fh:
        for row in output_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    component_path = output_dir / "step_components.jsonl"
    with component_path.open("w", encoding="utf-8") as fh:
        for row in component_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PILOT_NOT_PRODUCTION",
        "source": {"path": str(input_path), "md5": _md5(input_path), "records": len(rows)},
        "protocol": {
            "version": "rule_continuous_v1.1",
            "gold_answer_used": False,
            "original_label_overwritten": False,
            "citation_accuracy": "exact normalised membership in stored Teacher-visible KG",
            "formula": "A * sqrt(F * Q) * (0.5 + 0.5 * P) * numeric_consistency",
            "faithfulness": "0.4*head_step + 0.2*relation_step + 0.4*tail_step",
            "question_relevance": "0.6*max(PID intent, relation surface in question) + 0.4*question entity anchor",
            "passage_grounding": "0.5*head_passage + 0.5*tail_passage",
            "aggregation": "mean per-triple utility; no citation=0; old negative contradiction=-1",
        },
        "accounting": {
            "trajectories": len(rows),
            "steps": len(new_labels),
            "cited_steps": sum(bool(row["components"]) for row in component_rows),
            "invalid_citations": invalid_citations,
        },
        "old_label_distribution": _stats(old_labels),
        "rule_label_distribution": _stats(new_labels),
        "rule_label_by_dataset": {key: _stats(value) for key, value in sorted(by_dataset.items())},
        "transition_old_to_rule_bucket": {
            f"old={old}|new={bucket}": count for (old, bucket), count in sorted(transition.items())
        },
        "outputs": {"candidate": str(candidate_path), "components": str(component_path)},
        "decision": "RESEARCHER_REVIEW_REQUIRED",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir / "run", extra={
        "experiment": "rule_continuous_v1_1_offline_reannotation",
        "report": str(report_path),
        "status": report["status"],
        "steps": len(new_labels),
        "gold_answer_used": False,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
