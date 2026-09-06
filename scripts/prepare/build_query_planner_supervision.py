#!/usr/bin/env python
"""Build answer-free query-planner supervision from train annotations.

2Wiki evidence tails and MuSiQue decomposition answers/support paragraphs are
used only to derive graph dependencies, then omitted from the output.  The
result contains no golden answer, evidence tail, support title or paragraph.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID
from kgproweight.kg.question_kg import question_key, question_sha256
from kgproweight.utils.logging import dump_manifest


SCHEMA_VERSION = "query-planner-supervision-1"
_FORBIDDEN_KEYS = {
    "answer", "answers", "golden_answers", "entity", "paragraph_text",
    "support_paragraph", "supporting_facts", "support_title", "support_titles",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(value: object) -> str:
    return re.sub(r"\W+", " ", str(value).casefold()).strip()


def _entity_keys(value: object) -> List[str]:
    text = str(value)
    keys = [_norm(text)]
    without_parenthetical = _norm(re.sub(r"\s*\([^()]+\)\s*$", "", text))
    if without_parenthetical and without_parenthetical not in keys:
        keys.append(without_parenthetical)
    return keys


def _is_conservative_alias(left: object, right: object) -> bool:
    """Match a short evidence alias to a longer surface without fuzzy spelling.

    2Wiki sometimes emits an intermediate tail such as ``Walter Edwards`` but
    uses ``Walter Edwards (director)`` or ``V. Nagaiah`` as the next fact.  We
    permit only a contiguous/ordered token expansion or a very close spelling
    variant.  The caller additionally requires exactly one prior output slot.
    """
    left_tokens = _norm(re.sub(r"\s*\([^()]+\)\s*$", "", str(left))).split()
    right_tokens = _norm(re.sub(r"\s*\([^()]+\)\s*$", "", str(right))).split()
    if not left_tokens or not right_tokens or left_tokens == right_tokens:
        return False
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    if len(shorter) == 1 and (len(shorter[0]) < 4 or not shorter[0].isalpha()):
        return False
    width = len(shorter)
    if any(longer[offset : offset + width] == shorter for offset in range(len(longer) - width + 1)):
        return True
    # Exact ordered expansion: ``Arun Gandhi`` -> ``Arun Manilal Gandhi``.
    cursor = 0
    for token in longer:
        if cursor < len(shorter) and token == shorter[cursor]:
            cursor += 1
    if cursor == len(shorter):
        return True
    if len(left_tokens) != len(right_tokens):
        return False
    # Annotation spelling/abbreviation variants.  Never fuzzy-match numbers or
    # Roman numerals (e.g. Charles I must not resolve to Charles III).
    for left_token, right_token in zip(left_tokens, right_tokens):
        if left_token == right_token:
            continue
        if not (left_token.isalpha() and right_token.isalpha()):
            return False
        if re.fullmatch(r"[ivxlcdm]+", left_token) or re.fullmatch(r"[ivxlcdm]+", right_token):
            return False
        if len(left_token) == 1 or len(right_token) == 1:
            if left_token[0] != right_token[0]:
                return False
            continue
        shorter_token, longer_token = sorted((left_token, right_token), key=len)
        if len(shorter_token) < 4:
            return False
        if longer_token.startswith(shorter_token):
            continue
        if SequenceMatcher(None, left_token, right_token).ratio() < 0.88:
            return False
    return True


def _resolve_prior_output(
    fact: object,
    produced: Mapping[str, str],
    prior_outputs: Sequence[Tuple[str, str]],
) -> str | None:
    return _resolve_prior_output_with_method(fact, produced, prior_outputs)[0]


def _resolve_prior_output_with_method(
    fact: object,
    produced: Mapping[str, str],
    prior_outputs: Sequence[Tuple[str, str]],
) -> Tuple[str | None, str]:
    exact = {produced[key] for key in _entity_keys(fact) if key in produced}
    if len(exact) == 1:
        return next(iter(exact)), "exact"
    if len(exact) > 1:
        return None, "ambiguous_exact"
    candidates = {
        slot for tail_surface, slot in prior_outputs
        if _is_conservative_alias(fact, tail_surface)
    }
    if len(candidates) == 1:
        return next(iter(candidates)), "unique_alias"
    return None, "ambiguous_alias" if candidates else "root"


def _assert_answer_free(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_KEYS.intersection(str(key) for key in value)
        if forbidden:
            raise ValueError(f"forbidden supervision keys: {sorted(forbidden)}")
        for child in value.values():
            _assert_answer_free(child)
    elif isinstance(value, list):
        for child in value:
            _assert_answer_free(child)


def build_2wiki_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    evidences = ((row.get("metadata") or {}).get("evidences") or {})
    facts = list(evidences.get("fact") or [])
    relations = list(evidences.get("relation") or [])
    tails = list(evidences.get("entity") or [])
    if not (len(facts) == len(relations) == len(tails)):
        raise ValueError(f"misaligned 2Wiki evidences for {row.get('id')}")
    produced: Dict[str, str] = {}
    prior_outputs: List[Tuple[str, str]] = []
    anchors: List[str] = []
    steps: List[Dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    has_degenerate_source_anchor = False
    for index, (fact, relation, tail) in enumerate(zip(facts, relations, tails), start=1):
        resolved_subject, resolution = _resolve_prior_output_with_method(fact, produced, prior_outputs)
        resolution_counts[resolution] += 1
        raw_fact = str(fact)
        if resolved_subject is None and raw_fact != raw_fact.strip():
            normalized_width = len(_norm(raw_fact).replace(" ", ""))
            has_degenerate_source_anchor |= normalized_width < 2
        subject = resolved_subject or raw_fact.strip()
        is_slot_reference = bool(re.fullmatch(r"\$hop_\d+", subject))
        if not is_slot_reference and subject not in anchors:
            anchors.append(subject)
        slot = f"hop_{index}"
        pid = _RELATION_LABEL_TO_PID.get(str(relation).casefold())
        steps.append(
            {
                "step": index,
                "subject": subject,
                "relation_label": str(relation),
                "pid": pid,
                "output_slot": slot,
                "dependencies": [subject[1:]] if is_slot_reference else [],
            }
        )
        # The tail is used only to connect a later subject to this output slot;
        # it is deliberately absent from the serialized target.
        for key in _entity_keys(tail):
            produced[key] = f"${slot}"
        prior_outputs.append((str(tail), f"${slot}"))
    record = {
        "schema_version": SCHEMA_VERSION,
        "question_key": question_key("2wikimultihopqa", str(row["id"])),
        "dataset": "2wikimultihopqa",
        "qid": str(row["id"]),
        "question": str(row["question"]),
        "question_sha256": question_sha256(str(row["question"])),
        "target_type": "relation_graph",
        "target": {"anchors": anchors, "steps": steps},
        "provenance": {
            "split": "train",
            "source_annotation": "metadata.evidences",
            "dependency_resolution_counts": dict(resolution_counts),
            "has_degenerate_source_anchor": has_degenerate_source_anchor,
        },
    }
    _assert_answer_free(record)
    return record


def build_musique_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    decomposition = (((row.get("metadata") or {}).get("metadata") or {}).get("question_decomposition") or [])
    steps: List[Dict[str, Any]] = []
    for index, source in enumerate(decomposition, start=1):
        subquery = str(source.get("question") or "").strip()
        # ``#9 Dream`` is a song title, not a forward reference.  MuSiQue
        # decomposition placeholders can only refer to an already emitted step.
        dependencies = [
            f"step_{number}" for number in re.findall(r"#(\d+)", subquery)
            if int(number) < index
        ]
        steps.append(
            {
                "step": index,
                "subquery_template": subquery,
                "dependencies": dependencies,
                "output_slot": f"step_{index}",
            }
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "question_key": question_key("musique", str(row["id"])),
        "dataset": "musique",
        "qid": str(row["id"]),
        "question": str(row["question"]),
        "question_sha256": question_sha256(str(row["question"])),
        "target_type": "subquery_graph",
        "target": {"steps": steps},
        "provenance": {
            "split": "train",
            "source_annotation": "metadata.metadata.question_decomposition.question_only",
        },
    }
    _assert_answer_free(record)
    return record


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)
    elif isinstance(value, str):
        yield value


def _contains_token_phrase(value: object, phrase: object) -> bool:
    value_tokens, phrase_tokens = _norm(value).split(), _norm(phrase).split()
    if not value_tokens or not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(
        value_tokens[offset : offset + width] == phrase_tokens
        for offset in range(len(value_tokens) - width + 1)
    )


def _record_leaks_source_values(record: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    target_values = list(_string_values(record.get("target") or {}))
    question = _norm(record.get("question") or "")
    values = list(source.get("golden_answers") or [])
    if record.get("dataset") == "2wikimultihopqa":
        values.extend((((source.get("metadata") or {}).get("evidences") or {}).get("entity") or []))
    else:
        decomposition = (((source.get("metadata") or {}).get("metadata") or {}).get("question_decomposition") or [])
        values.extend(item.get("answer") for item in decomposition)
    for value in values:
        secret = _norm(value)
        if (
            len(secret) >= 3
            and secret not in question
            and any(_contains_token_phrase(target_value, value) for target_value in target_values)
        ):
            return True
    return False


def _record_exclusion_reason(record: Mapping[str, Any], source: Mapping[str, Any]) -> str | None:
    if _record_leaks_source_values(record, source):
        return "answer_or_evidence_tail_present_in_target_outside_question"
    if record.get("target_type") == "relation_graph":
        anchors = (record.get("target") or {}).get("anchors") or []
        provenance = record.get("provenance") or {}
        if provenance.get("has_degenerate_source_anchor") or any(not _norm(anchor) for anchor in anchors):
            return "invalid_degenerate_anchor"
    return None


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--selection", help="Optional frozen cohort JSONL; only selected train qids are emitted")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    output_dir.mkdir(parents=True)

    selected: Dict[str, set[str]] | None = None
    selection_path = Path(args.selection).resolve() if args.selection else None
    if selection_path:
        selected = {}
        for row in _read_jsonl(selection_path):
            source_id = row.get("source_id", row.get("qid"))
            if source_id is None:
                raise ValueError(f"selection row lacks source_id/qid: {row}")
            selected.setdefault(str(row["dataset"]), set()).add(str(source_id))

    builders = {
        "2wikimultihopqa": build_2wiki_record,
        "musique": build_musique_record,
    }
    inputs: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    for dataset, builder in builders.items():
        path = Path(args.data_root).resolve() / dataset / "train.jsonl"
        inputs[dataset] = {"path": str(path), "sha256": _sha256(path)}
        wanted = selected.get(dataset, set()) if selected is not None else None
        for row in _read_jsonl(path):
            if wanted is not None and str(row["id"]) not in wanted:
                continue
            record = builder(row)
            exclusion_reason = _record_exclusion_reason(record, row)
            if exclusion_reason:
                excluded.append(
                    {
                        "dataset": dataset,
                        "qid": str(row["id"]),
                        "reason": exclusion_reason,
                    }
                )
                continue
            records.append(record)

    output_path = output_dir / "planner_supervision.jsonl"
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    excluded_path = output_dir / "excluded_records.jsonl"
    with excluded_path.open("w", encoding="utf-8") as fh:
        for row in excluded:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    mapped = Counter()
    dependency_resolution = Counter()
    for record in records:
        if record["target_type"] == "relation_graph":
            dependency_resolution.update(
                (record.get("provenance") or {}).get("dependency_resolution_counts") or {}
            )
            for step in record["target"]["steps"]:
                mapped["relation_steps"] += 1
                mapped["pid_mapped"] += int(bool(step["pid"]))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "schema_version": SCHEMA_VERSION,
        "scope": "train-only planner supervision; answers/tails/support text omitted",
        "inputs": inputs,
        "selection": (
            {"path": str(selection_path), "sha256": _sha256(selection_path)}
            if selection_path else None
        ),
        "counts": {
            "records": len(records),
            "by_dataset": dict(Counter(record["dataset"] for record in records)),
            "excluded_records": len(excluded),
            "excluded_by_dataset": dict(Counter(record["dataset"] for record in excluded)),
            **mapped,
            "dependency_resolution": dict(dependency_resolution),
        },
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "exclusions": {"path": str(excluded_path), "sha256": _sha256(excluded_path)},
        "excluded_fields": sorted(_FORBIDDEN_KEYS),
        "hotpotqa_status": "UNKNOWN_NO_DECOMPOSITION_TARGET_BUILT",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(output_dir / "run", extra=report)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
