#!/usr/bin/env python
"""Build a leakage-controlled 2Wiki entity/property store from official IDs.

Only rows assigned to the query-planner training partition are eligible.  All
families present in the supplied confirmation cohorts are removed before any
alias or edge is collected.  The global official alias file is filtered to
QIDs already observed in those eligible evidence triples; it cannot introduce
a confirmation-only entity into the store.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Tuple

from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID
from kgproweight.kg.question_kg import question_key
from kgproweight.kg.versioned_evidence_store import (
    STORE_SCHEMA_VERSION,
    VersionedEvidenceStore,
    normalize_alias,
)
from kgproweight.utils.logging import artifact_identity, dump_manifest
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256
from scripts.prepare.audit_auto1500_v4_clean_reproducibility import (
    PROTECTED_LEDGER_SCHEMA_VERSION,
    validate_protected_ledger_release,
)


BUILDER_VERSION = (
    "official-2wiki-ids-training-partition-store-3-complete-ledger-exclusion"
)
_QID = re.compile(r"^Q[1-9][0-9]*$")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_qid(value: object) -> bool:
    return bool(_QID.fullmatch(str(value or "").strip().upper()))


def build_store(
    *,
    official_rows: Iterable[Mapping[str, Any]],
    official_alias_rows: Iterable[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    excluded_families: set[str],
    excluded_question_keys: set[str] | None = None,
    excluded_current_families: set[str] | None = None,
) -> tuple[Dict[str, Counter[Tuple[str, str]]], Dict[str, Counter[Tuple[str, ...]]], Counter[str]]:
    aliases: Dict[str, Counter[Tuple[str, str]]] = defaultdict(Counter)
    edges: Dict[str, Counter[Tuple[str, ...]]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    eligible_qids: set[str] = set()
    excluded_question_keys = set(excluded_question_keys or ())
    excluded_current_families = set(excluded_current_families or ())

    for row in official_rows:
        counts["official_rows"] += 1
        key = f"2wikimultihopqa::{row.get('_id')}"
        assignment = assignments.get(key)
        if not assignment or assignment.get("split") != "train":
            counts["excluded_nontrain_rows"] += 1
            continue
        # Exact identity exclusion is mandatory.  Older stores relied only on
        # the family hash stored in ``assignments``; after the lexical-family
        # implementation changed, that could silently re-admit the selected
        # question itself.
        if key in excluded_question_keys:
            counts["excluded_selected_question_rows"] += 1
            continue
        if str(assignment.get("family_sha256")) in excluded_families:
            counts["excluded_confirmation_family_rows"] += 1
            continue
        question = str(row.get("question") or "").strip()
        if (
            question
            and family_sha256(question) in excluded_current_families
        ):
            counts["excluded_current_lexical_family_rows"] += 1
            continue
        facts = list(row.get("evidences") or [])
        ids = list(row.get("evidences_id") or [])
        if len(facts) != len(ids):
            counts["unaligned_evidence_rows"] += 1
            continue
        counts["eligible_aligned_rows"] += 1
        for fact, id_fact in zip(facts, ids):
            if len(fact) != 3 or len(id_fact) != 3:
                counts["malformed_evidence_hops"] += 1
                continue
            head_label, relation, tail_label = map(str, fact)
            head_qid, id_relation, tail_id = map(str, id_fact)
            head_qid = head_qid.strip().upper()
            relation_key = relation.strip().casefold()
            pid = _RELATION_LABEL_TO_PID.get(relation_key)
            if not _is_qid(head_qid) or not pid:
                counts["unsupported_evidence_hops"] += 1
                continue
            if id_relation.strip().casefold() != relation_key:
                counts["relation_label_disagreements"] += 1
            tail_qid = tail_id.strip().upper() if _is_qid(tail_id) else ""
            eligible_qids.add(head_qid)
            aliases[normalize_alias(head_label)][(head_qid, head_label)] += 1
            if tail_qid:
                eligible_qids.add(tail_qid)
                aliases[normalize_alias(tail_label)][(tail_qid, tail_label)] += 1
            edge_key = VersionedEvidenceStore.edge_key(head_qid, pid)
            edges[edge_key][(
                head_qid,
                head_label,
                pid,
                relation,
                tail_qid,
                tail_label,
            )] += 1
            counts["stored_evidence_hops"] += 1

    counts["eligible_entity_qids"] = len(eligible_qids)
    for row in official_alias_rows:
        qid = str(row.get("Q_id") or "").strip().upper()
        if qid not in eligible_qids:
            continue
        for alias in list(row.get("aliases") or []) + list(row.get("demonyms") or []):
            normalized = normalize_alias(alias)
            if normalized:
                aliases[normalized][(qid, str(alias))] += 0
                counts["eligible_official_alias_surfaces"] += 1
    return aliases, edges, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official_train", required=True)
    parser.add_argument("--official_aliases", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--exclude_cohort", action="append", required=True)
    parser.add_argument(
        "--protected_ledger_dir",
        required=True,
        help="Complete versioned v4 protected-identity ledger release.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_id", required=True)
    args = parser.parse_args()

    train_path = Path(args.official_train).resolve()
    alias_path = Path(args.official_aliases).resolve()
    assignment_path = Path(args.assignments).resolve()
    ledger_path, ledger_report_path, ledger_manifest_path, _ledger_report = (
        validate_protected_ledger_release(Path(args.protected_ledger_dir))
    )
    cohort_paths = [ledger_path.resolve()]
    for raw_path in args.exclude_cohort:
        path = Path(raw_path).resolve()
        if path not in cohort_paths:
            cohort_paths.append(path)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_dir}")
    for path in (train_path, alias_path, assignment_path, *cohort_paths):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    assignments = {
        str(row["question_key"]): row for row in _read_jsonl(assignment_path)
        if row.get("dataset") == "2wikimultihopqa"
    }
    # Some newer identity-only protocols intentionally store only the current
    # question text/hash and omit the legacy assignment-family field.  Such
    # rows are still protected below by exact question key and a freshly
    # recomputed lexical family; absence of the obsolete family field must not
    # make the whole leakage-controlled store build crash.
    exclusion_rows = [
        row
        for path in cohort_paths
        for row in _read_jsonl(path)
        if str(row.get("dataset") or "2wikimultihopqa").strip().lower()
        == "2wikimultihopqa"
    ]
    excluded_families = {
        str(row.get("family_sha256") or "").strip()
        for row in exclusion_rows
        if str(row.get("family_sha256") or "").strip()
    }
    selected_keys = {
        str(
            row.get("question_key")
            or question_key(
                "2wikimultihopqa", str(row.get("qid") or row.get("id") or "").strip()
            )
        )
        for row in exclusion_rows
        if str(row.get("qid") or row.get("id") or "").strip()
    }
    excluded_current_families = {
        family_sha256(str(row.get("question") or "").strip())
        for row in exclusion_rows
        if str(row.get("question") or "").strip()
    }
    official_rows = json.loads(train_path.read_text(encoding="utf-8"))
    aliases, edges, counts = build_store(
        official_rows=official_rows,
        official_alias_rows=_read_jsonl(alias_path),
        assignments=assignments,
        excluded_families=excluded_families,
        excluded_question_keys=selected_keys,
        excluded_current_families=excluded_current_families,
    )
    output_dir.mkdir(parents=True)
    alias_output = output_dir / "aliases.jsonl"
    with alias_output.open("x", encoding="utf-8") as handle:
        for normalized in sorted(aliases):
            candidate_counts: Dict[str, Counter[str]] = defaultdict(Counter)
            for (qid, label), count in aliases[normalized].items():
                candidate_counts[qid][label] += count
            candidates = []
            for qid in sorted(candidate_counts):
                labels = candidate_counts[qid]
                label, evidence_count = sorted(
                    labels.items(), key=lambda value: (-value[1], value[0])
                )[0]
                candidates.append({
                    "qid": qid,
                    "label": label,
                    "evidence_count": sum(labels.values()),
                })
            handle.write(json.dumps({
                "schema_version": STORE_SCHEMA_VERSION,
                "normalized_alias": normalized,
                "candidates": candidates,
            }, ensure_ascii=False) + "\n")

    edge_output = output_dir / "edges.jsonl"
    with edge_output.open("x", encoding="utf-8") as handle:
        for key in sorted(edges):
            rendered = []
            for values, support_count in sorted(
                edges[key].items(), key=lambda value: (-value[1], value[0])
            ):
                head_qid, head_label, pid, relation, tail_qid, tail_value = values
                rendered.append({
                    "head_qid": head_qid,
                    "head_label": head_label,
                    "pid": pid,
                    "relation": relation,
                    "tail_qid": tail_qid or None,
                    "tail_value": tail_value,
                    "support_record_count": support_count,
                })
            handle.write(json.dumps({
                "schema_version": STORE_SCHEMA_VERSION,
                "key": key,
                "edges": rendered,
            }, ensure_ascii=False) + "\n")

    counts["alias_keys"] = len(aliases)
    counts["edge_keys"] = len(edges)
    counts["excluded_families"] = len(excluded_families)
    counts["excluded_current_lexical_families"] = len(excluded_current_families)
    counts["selected_confirmation_questions"] = len(selected_keys)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "schema_version": STORE_SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "status": "COMPLETE_NOT_EVALUATED",
        "protected_ledger": {
            "schema_version": PROTECTED_LEDGER_SCHEMA_VERSION,
            "complete": True,
            "current_family_recomputed": True,
            "report_sha256": _sha256(ledger_report_path),
            "protected_identities_sha256": _sha256(ledger_path),
            "manifest_sha256": _sha256(ledger_manifest_path),
        },
        "scientific_boundary": {
            "dataset_specific": "2WikiMultiHopQA",
            "official_training_annotations_used": True,
            "official_global_aliases_filtered_to_training_observed_qids": True,
            "selected_question_keys_excluded_exactly": True,
            "selected_current_lexical_families_recomputed_and_excluded": True,
            "selected_confirmation_questions_used": False,
            "selected_confirmation_families_used": False,
            "external_complete_wikidata_dump": False,
            "hotpot_or_musique_claim_allowed": False,
        },
        "counts": dict(counts),
        "inputs": {
            "official_train": artifact_identity(train_path),
            "official_aliases": artifact_identity(alias_path),
            "assignments": artifact_identity(assignment_path),
            "excluded_cohorts": [artifact_identity(path) for path in cohort_paths],
        },
        "outputs": {
            "aliases": artifact_identity(alias_output),
            "edges": artifact_identity(edge_output),
        },
    }
    manifest_path = output_dir / "store_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(output_dir, status=manifest["status"], extra=manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
