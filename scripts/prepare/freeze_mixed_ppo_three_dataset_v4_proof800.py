#!/usr/bin/env python3
"""Freeze the answer-free mixed3-v4 population and exact K=4 schedule.

The v4 population has exactly 1,000 questions from each dataset.  HotpotQA
and MuSiQue are consumed from the full-ledger-safe reconciliation release,
which retains safe frozen-v2 rows and deterministically replaces blocked
identities under the pre-declared strata.  The 2Wiki arm contains 800
complete, identity-safe ProofKG rows (200 per question type) plus the
full-ledger-safe ordinary200 successor with bound outcome/passage provenance.

This command is deliberately answer-free.  It never reads a Gold answer,
supporting sentence, generated trajectory, or evaluation prediction.  It
does not run retrieval and does not train a model.  The new HotpotQA/MuSiQue
identities are emitted as retrieval requests for a later versioned stage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kgproweight.kg.question_kg import question_sha256
from kgproweight.reward.trajectory_source_gate import evaluate_graph_gate
from kgproweight.utils.logging import dump_manifest
from scripts.prepare.freeze_mixed_ppo_three_dataset_v1 import rank, sha256_file
from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256


SEED = 42
K = 4
DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
QTYPES = ("inference", "comparison", "compositional", "bridge_comparison")
EXPERIMENT_ID = "MIXED-PPO-THREE-DATASET-V4-PROOF800-N3000-K4-12000-SEED42-PROTOCOL"
STATUS = "FROZEN_ANSWER_FREE_NOT_MATERIALIZED_NOT_TRAINED"
SCHEMA = "mixed-ppo-question-only-v4-proof800"
HISTORICAL_CUTOFF = "2020-12-09T23:59:59Z"
PROTECTED_LEDGER_SCHEMA = "mixed-ppo-v4-protected-identity-ledger-v2"
PROTECTED_LEDGER_STATUS = "COMPLETE_FROZEN_IDENTITY_ONLY_NOT_TRAINING_DATA"

# The cross cells are the unique integer factorisation of the two requested
# marginal distributions: type=750/250 and difficulty=200/600/200.
HOTPOT_TARGET_CELLS: dict[str, int] = {
    "bridge/easy": 150,
    "bridge/medium": 450,
    "bridge/hard": 150,
    "comparison/easy": 50,
    "comparison/medium": 150,
    "comparison/hard": 50,
}
MUSIQUE_TARGET_HOPS: dict[str, int] = {
    "2hop": 650,
    "3hop": 250,
    "4hop": 100,
}
PROOF_TARGET_TYPES: dict[str, int] = {qtype: 200 for qtype in QTYPES}

DEFAULT_PARENT_PROTOCOL = Path(
    "outputs/audits/mixed_ppo_three_dataset_v2_proof400_n1799_k4_seed42_protocol/protocol.json"
)
DEFAULT_REPLAY = Path(
    "data/silver_data/sft_replay_legacy_strong_train_rendered3to5_n2000_seed42_v2/silver_train.jsonl"
)
DEFAULT_OUT = Path(
    "outputs/audits/mixed_ppo_three_dataset_v4_proof800_n3000_k4_seed42_protocol"
)
DEFAULT_ORDINARY200_PROTOCOL = Path(
    "outputs/audits/2wiki_ordinary200_full_ledger_v2_seed42_preregistration/"
    "protocol.json"
)
ORDINARY200_PROTOCOL_SCHEMA = "2wiki-ordinary200-full-ledger-protocol-v2"
ORDINARY200_PROTOCOL_STATUS = "FROZEN_ANSWER_FREE_OUTCOME_SOURCE_BOUND_NOT_TRAINED"
DEFAULT_HM_RECONCILIATION_PROTOCOL = Path(
    "outputs/audits/mixed_ppo_v4_hm_full_ledger_reconciliation_v2_seed42_"
    "preregistration/protocol.json"
)
HM_RECONCILIATION_SCHEMA = "mixed-ppo-v4-hm-full-ledger-reconciliation-v2"
HM_RECONCILIATION_STATUS = (
    "FROZEN_HM_FULL_LEDGER_DELTA_RETRIEVAL_NOT_RUN_NOT_TRAINED"
)
# The versioned ledger recomputes every historical cohort under the current
# lexical-family function.  Do not replace this with a short hand-maintained
# path list: the previous eight-path default omitted reward/verifier cohorts.
DEFAULT_PROTECTED_LEDGER_DIR = Path(
    "outputs/audits/mixed_ppo_v4_protected_identity_ledger_v2"
)
# Compatibility for answer-free upstream request freezers which import the
# identity file path.  Formal v4 protocol creation still validates and binds
# the enclosing report+manifest release via ``DEFAULT_PROTECTED_LEDGER_DIR``.
DEFAULT_PROTECTED = (
    DEFAULT_PROTECTED_LEDGER_DIR / "protected_identities.question_only.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def ref(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_protected_ledger_release(
    directory: Path,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Bind the protocol to ledger data, report, and manifest as one release."""

    directory = directory.resolve()
    paths = {
        "ledger": directory / "protected_identities.question_only.jsonl",
        "report": directory / "report.json",
        "manifest": directory / "manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if not (
        report.get("schema_version") == PROTECTED_LEDGER_SCHEMA
        and report.get("status") == PROTECTED_LEDGER_STATUS
        and manifest.get("status") == PROTECTED_LEDGER_STATUS
        and report.get("complete") is True
        and report.get("identity_scope") == "dataset-scoped"
        and report.get("current_family_recomputed") is True
    ):
        raise ValueError("protected ledger release status/schema/completeness failed")
    binding = {name: ref(path) for name, path in paths.items()}
    output = report.get("output") or {}
    run = manifest.get("run") or {}
    if (
        output.get("sha256") != binding["ledger"]["sha256"]
        or int(output.get("rows", -1))
        != len(read_jsonl(paths["ledger"]))
        or run.get("protected_identities_sha256")
        != binding["ledger"]["sha256"]
        or run.get("report_sha256") != binding["report"]["sha256"]
    ):
        raise ValueError("protected ledger release hash/count binding failed")
    return paths["ledger"], binding


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class IdentityIndex:
    """Dataset-scoped identity registry used for fail-closed isolation."""

    qids: set[tuple[str, str]] = field(default_factory=set)
    question_hashes: set[tuple[str, str]] = field(default_factory=set)
    families: set[tuple[str, str]] = field(default_factory=set)

    def add(self, row: Mapping[str, Any]) -> None:
        dataset = str(row["dataset"]).strip().lower()
        self.qids.add((dataset, str(row["qid"]).strip()))
        self.question_hashes.add((dataset, str(row["question_sha256"])))
        self.families.add((dataset, str(row["family_sha256"])))

    def update(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.add(row)

    def overlaps(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        dataset = str(row["dataset"]).strip().lower()
        reasons = []
        if (dataset, str(row["qid"]).strip()) in self.qids:
            reasons.append("qid")
        if (dataset, str(row["question_sha256"])) in self.question_hashes:
            reasons.append("question_sha256")
        if (dataset, str(row["family_sha256"])) in self.families:
            reasons.append("family_sha256")
        return tuple(reasons)


def _identity(
    row: Mapping[str, Any],
    *,
    dataset: str | None = None,
    route: str = "unspecified",
    eligible: bool = False,
    question_type: str = "unknown",
    stratum: str = "unknown",
    source_role: str = "candidate",
    proof_source: str = "none",
    proof_record_sha256: str | None = None,
    proof_passages_sha256: str | None = None,
) -> dict[str, Any]:
    actual_dataset = str(dataset or row.get("dataset") or "").strip().lower()
    qid = str(row.get("qid") or row.get("id") or "").strip()
    question = str(row.get("question") or "").strip()
    if actual_dataset not in DATASETS or not qid or not question:
        raise ValueError(
            f"incomplete identity: dataset={actual_dataset!r}, qid={qid!r}, question={question!r}"
        )
    qhash = question_sha256(question)
    supplied_qhash = str(row.get("question_sha256") or "")
    if supplied_qhash and supplied_qhash != qhash:
        raise ValueError(f"question hash mismatch: {actual_dataset}::{qid}")
    family = family_sha256(question)
    supplied_family = str(row.get("family_sha256") or "")
    if supplied_family and supplied_family != family:
        raise ValueError(f"family hash mismatch: {actual_dataset}::{qid}")
    value = {
        "schema_version": SCHEMA,
        "question_key": f"{actual_dataset}::{qid}",
        "dataset": actual_dataset,
        "qid": qid,
        "question": question,
        "question_sha256": qhash,
        "family_version": FAMILY_VERSION,
        "family_sha256": family,
        "question_type": str(question_type),
        "stratum": str(stratum),
        "route": str(route),
        "source_role": str(source_role),
        "proof_source": str(proof_source),
        "process_reward_eligible": bool(eligible),
        "gold_access": False,
        "evaluation_eligible": False,
    }
    if proof_record_sha256 is not None:
        value["proof_record_sha256"] = proof_record_sha256
    if proof_passages_sha256 is not None:
        value["proof_passages_sha256"] = proof_passages_sha256
    return value


def _normalise_external_identities(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _identity(
            row,
            route="blocked_external_identity",
            source_role="blocked_external_identity",
        )
        for row in rows
    ]


def identity_overlap_counts(
    left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]]
) -> dict[str, int]:
    right_index = IdentityIndex()
    right_index.update(right)
    counts = Counter()
    for row in left:
        for reason in right_index.overlaps(row):
            counts[reason] += 1
    return {
        "qid": counts["qid"],
        "question_sha256": counts["question_sha256"],
        "family_sha256": counts["family_sha256"],
    }


def _hotpot_stratum(raw: Mapping[str, Any]) -> str:
    metadata = raw.get("metadata") or {}
    qtype = str(metadata.get("type") or "").strip().lower()
    level = str(metadata.get("level") or "").strip().lower()
    stratum = f"{qtype}/{level}"
    if stratum not in HOTPOT_TARGET_CELLS:
        raise ValueError(f"unsupported HotpotQA stratum {stratum!r}")
    return stratum


def _musique_stratum(raw: Mapping[str, Any]) -> str:
    metadata = raw.get("metadata") or {}
    decomposition = list((metadata.get("metadata") or {}).get("question_decomposition") or [])
    stratum = f"{len(decomposition)}hop"
    if stratum not in MUSIQUE_TARGET_HOPS:
        raise ValueError(f"unsupported MuSiQue stratum {stratum!r}")
    return stratum


def _raw_candidates(dataset: str, raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for raw in raw_rows:
        stratum = _hotpot_stratum(raw) if dataset == "hotpotqa" else _musique_stratum(raw)
        question_type = stratum.split("/", 1)[0] if dataset == "hotpotqa" else stratum
        output.append(
            _identity(
                raw,
                dataset=dataset,
                route=f"{dataset}_outcome",
                question_type=question_type,
                stratum=stratum,
                source_role="raw_train_candidate",
            )
        )
    return output


def _choose_by_stratum(
    candidates: Sequence[Mapping[str, Any]],
    *,
    needs: Mapping[str, int],
    blocked: IdentityIndex,
    label: str,
    reserve_per_stratum: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if not blocked.overlaps(row):
            by_stratum[str(row["stratum"])].append(dict(row))
    selected: list[dict[str, Any]] = []
    reserve: list[dict[str, Any]] = []
    used = IdentityIndex()
    for stratum in needs:
        ordered = sorted(
            by_stratum.get(stratum, []),
            key=lambda row: (
                rank(f"{label}-{stratum}", str(row["dataset"]), str(row["qid"])),
                str(row["qid"]),
            ),
        )
        need = int(needs[stratum])
        chosen: list[dict[str, Any]] = []
        remainder: list[dict[str, Any]] = []
        for row in ordered:
            if used.overlaps(row):
                continue
            if len(chosen) < need:
                item = dict(row)
                item["source_role"] = "new_retrieval"
                chosen.append(item)
                used.add(item)
            else:
                remainder.append(dict(row))
        if len(chosen) != need:
            raise ValueError(f"{label}/{stratum}: only {len(chosen)}/{need} isolated candidates")
        selected.extend(chosen)
        taken = 0
        for row in remainder:
            if used.overlaps(row):
                continue
            item = dict(row)
            item["source_role"] = "identity_only_reserve"
            reserve.append(item)
            used.add(item)
            taken += 1
            if taken == reserve_per_stratum:
                break
        if taken != reserve_per_stratum:
            raise ValueError(
                f"{label}/{stratum}: only {taken}/{reserve_per_stratum} reserve candidates"
            )
    return selected, reserve


def build_hotpot_population(
    parent_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    externally_blocked: IdentityIndex,
    reserve_per_stratum: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_candidates = _raw_candidates("hotpotqa", raw_rows)
    raw_by_qid = {row["qid"]: row for row in raw_candidates}
    if len(raw_by_qid) != len(raw_candidates):
        raise ValueError("HotpotQA raw train contains duplicate qids")
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for parent in parent_rows:
        raw = raw_by_qid.get(str(parent["qid"]))
        if raw is None:
            raise ValueError(f"parent HotpotQA qid missing raw row: {parent['qid']}")
        if str(parent.get("question_sha256")) != str(raw["question_sha256"]):
            raise ValueError(f"parent/raw HotpotQA hash mismatch: {parent['qid']}")
        item = dict(raw)
        item["source_role"] = "retained_parent"
        reasons = externally_blocked.overlaps(item)
        if reasons:
            item["exclusion_reasons"] = list(reasons)
            removed.append(item)
        else:
            retained.append(item)

    retained_counts = Counter(row["stratum"] for row in retained)
    needs = {
        stratum: target - retained_counts[stratum]
        for stratum, target in HOTPOT_TARGET_CELLS.items()
    }
    if any(value < 0 for value in needs.values()):
        raise ValueError(f"retained HotpotQA exceeds target cells: {needs}")
    blocked = IdentityIndex(
        qids=set(externally_blocked.qids),
        question_hashes=set(externally_blocked.question_hashes),
        families=set(externally_blocked.families),
    )
    blocked.update(parent_rows)
    selected, reserve = _choose_by_stratum(
        raw_candidates,
        needs=needs,
        blocked=blocked,
        label="v4-hotpot",
        reserve_per_stratum=reserve_per_stratum,
    )
    final = [*retained, *selected]
    if len(final) != 1000 or Counter(row["stratum"] for row in final) != Counter(HOTPOT_TARGET_CELLS):
        raise ValueError("HotpotQA target population/strata not met")
    return final, selected, reserve, {
        "parent": len(parent_rows),
        "retained_parent": len(retained),
        "removed_parent": len(removed),
        "removed_overlap_reasons": dict(
            sorted(Counter(reason for row in removed for reason in row["exclusion_reasons"]).items())
        ),
        "new_retrieval": len(selected),
        "target_cells": dict(HOTPOT_TARGET_CELLS),
        "new_by_cell": dict(sorted(Counter(row["stratum"] for row in selected).items())),
        "reserve": len(reserve),
    }


def build_musique_population(
    parent_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    externally_blocked: IdentityIndex,
    reserve_per_stratum: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_candidates = _raw_candidates("musique", raw_rows)
    raw_by_qid = {row["qid"]: row for row in raw_candidates}
    if len(raw_by_qid) != len(raw_candidates):
        raise ValueError("MuSiQue raw train contains duplicate qids")
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for parent in parent_rows:
        raw = raw_by_qid.get(str(parent["qid"]))
        if raw is None:
            raise ValueError(f"parent MuSiQue qid missing raw row: {parent['qid']}")
        if str(parent.get("question_sha256")) != str(raw["question_sha256"]):
            raise ValueError(f"parent/raw MuSiQue hash mismatch: {parent['qid']}")
        item = dict(raw)
        item["source_role"] = "retained_parent"
        reasons = externally_blocked.overlaps(item)
        if reasons:
            item["exclusion_reasons"] = list(reasons)
            removed.append(item)
        else:
            retained.append(item)
    retained_counts = Counter(row["stratum"] for row in retained)
    needs = {
        stratum: target - retained_counts[stratum]
        for stratum, target in MUSIQUE_TARGET_HOPS.items()
    }
    if any(value < 0 for value in needs.values()):
        raise ValueError(f"retained MuSiQue exceeds target hop strata: {needs}")
    blocked = IdentityIndex(
        qids=set(externally_blocked.qids),
        question_hashes=set(externally_blocked.question_hashes),
        families=set(externally_blocked.families),
    )
    blocked.update(parent_rows)
    selected, reserve = _choose_by_stratum(
        raw_candidates,
        needs=needs,
        blocked=blocked,
        label="v4-musique",
        reserve_per_stratum=reserve_per_stratum,
    )
    final = [*retained, *selected]
    if len(final) != 1000 or Counter(row["stratum"] for row in final) != Counter(MUSIQUE_TARGET_HOPS):
        raise ValueError("MuSiQue target population/strata not met")
    return final, selected, reserve, {
        "parent": len(parent_rows),
        "retained_parent": len(retained),
        "removed_parent": len(removed),
        "removed_overlap_reasons": dict(
            sorted(
                Counter(
                    reason
                    for row in removed
                    for reason in row["exclusion_reasons"]
                ).items()
            )
        ),
        "new_retrieval": len(selected),
        "target_hops": dict(MUSIQUE_TARGET_HOPS),
        "new_by_hop": dict(sorted(Counter(row["stratum"] for row in selected).items())),
        "reserve": len(reserve),
    }


def _proof_candidate(
    wrapper: Mapping[str, Any], *, historical_cutoff: str
) -> tuple[dict[str, Any] | None, str]:
    record_value = wrapper.get("question_kg_record")
    record = record_value if isinstance(record_value, Mapping) else wrapper
    qtype = str(
        wrapper.get("question_type")
        or (wrapper.get("metadata") or {}).get("question_type")
        or record.get("question_type")
        or ""
    ).strip()
    if qtype not in QTYPES:
        raise ValueError(f"Proof candidate has invalid/missing question_type: {qtype!r}")
    passages_hash = str(wrapper.get("proof_passages_sha256") or "").strip()
    if len(passages_hash) != 64 or any(
        char not in "0123456789abcdef" for char in passages_hash.lower()
    ):
        raise ValueError(
            "Proof candidate lacks a valid frozen proof_passages_sha256"
        )
    identity_source = {
        **dict(record),
        "family_sha256": wrapper.get("family_sha256") or record.get("family_sha256"),
    }
    item = _identity(
        identity_source,
        dataset="2wikimultihopqa",
        route=f"2wiki_proof_{qtype}",
        eligible=True,
        question_type=qtype,
        stratum=qtype,
        source_role="unified_proof_candidate",
        proof_source=str((record.get("provenance") or {}).get("builder_version") or "unknown"),
        proof_record_sha256=canonical_sha256(record),
        proof_passages_sha256=passages_hash,
    )
    decision = evaluate_graph_gate(
        record,
        dataset=item["dataset"],
        qid=item["qid"],
        question=item["question"],
        historical_cutoff=historical_cutoff,
    )
    if not decision.graph_eligible:
        return None, decision.routing_reason
    return item, "eligible"


def normalise_proof_candidates(
    rows: Sequence[Mapping[str, Any]], *, historical_cutoff: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    reasons = Counter()
    seen_qids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    for wrapper in rows:
        item, reason = _proof_candidate(wrapper, historical_cutoff=historical_cutoff)
        reasons[reason] += 1
        if item is None:
            continue
        qid = str(item["qid"])
        qhash = str(item["question_sha256"])
        if qid in seen_qids:
            raise ValueError(f"duplicate Proof candidate qid: {qid}")
        if qhash in seen_hashes:
            raise ValueError(
                f"duplicate Proof candidate question hash: {qid} and {seen_hashes[qhash]}"
            )
        seen_qids[qid] = qhash
        seen_hashes[qhash] = qid
        output.append(item)
    return output, dict(sorted(reasons.items()))


def select_proof800(
    candidates: Sequence[Mapping[str, Any]],
    *,
    blocked: IdentityIndex,
    reserve_per_stratum: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    blocked_counts = Counter()
    for raw in candidates:
        row = dict(raw)
        overlaps = blocked.overlaps(row)
        if overlaps:
            blocked_counts.update(overlaps)
            continue
        by_type[str(row["question_type"])].append(row)
    selected: list[dict[str, Any]] = []
    reserve: list[dict[str, Any]] = []
    used_qids: set[str] = set()
    used_hashes: set[str] = set()
    for qtype in QTYPES:
        # One row per family is preferred, then deterministic same-family rows
        # are allowed.  The latter is necessary for template-heavy 2Wiki while
        # qid and exact question identity remain unique.
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_type.get(qtype, []):
            grouped[str(row["family_sha256"])].append(row)
        for family in grouped:
            grouped[family].sort(
                key=lambda row: (
                    rank(f"v4-proof-{qtype}-within-family", row["dataset"], row["qid"]),
                    row["qid"],
                )
            )
        family_order = sorted(
            grouped,
            key=lambda family: (
                rank(f"v4-proof-{qtype}-family", "2wikimultihopqa", family),
                family,
            ),
        )
        ordered = [grouped[family][0] for family in family_order]
        ordered.extend(
            sorted(
                [row for family in family_order for row in grouped[family][1:]],
                key=lambda row: (
                    rank(f"v4-proof-{qtype}-repeat", row["dataset"], row["qid"]),
                    row["qid"],
                ),
            )
        )
        unique_ordered = [
            row
            for row in ordered
            if row["qid"] not in used_qids and row["question_sha256"] not in used_hashes
        ]
        need = PROOF_TARGET_TYPES[qtype]
        chosen = [dict(row) for row in unique_ordered[:need]]
        if len(chosen) != need:
            raise ValueError(f"Proof800/{qtype}: only {len(chosen)}/{need} eligible candidates")
        for row in chosen:
            row["source_role"] = "selected_unified_proof"
            used_qids.add(str(row["qid"]))
            used_hashes.add(str(row["question_sha256"]))
        selected.extend(chosen)
        candidates_for_reserve = unique_ordered[need:]
        taken = 0
        if reserve_per_stratum:
            for raw in candidates_for_reserve:
                if raw["qid"] in used_qids or raw["question_sha256"] in used_hashes:
                    continue
                row = dict(raw)
                row["source_role"] = "identity_only_reserve"
                reserve.append(row)
                used_qids.add(str(row["qid"]))
                used_hashes.add(str(row["question_sha256"]))
                taken += 1
                if taken == reserve_per_stratum:
                    break
        if taken != reserve_per_stratum:
            raise ValueError(
                f"Proof800/{qtype}: only {taken}/{reserve_per_stratum} reserve candidates"
            )
    return selected, reserve, {
        "input_eligible": len(candidates),
        "blocked_identity_hits": dict(sorted(blocked_counts.items())),
        "selected": len(selected),
        "selected_by_question_type": dict(
            sorted(Counter(row["question_type"] for row in selected).items())
        ),
        "selected_unique_families": len({row["family_sha256"] for row in selected}),
        "reserve": len(reserve),
    }


def build_groups(population: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_dataset = {
        dataset: sorted(
            (dict(row) for row in population if row["dataset"] == dataset),
            key=lambda row: (
                rank(f"v4-schedule-{dataset}", dataset, str(row["qid"])),
                str(row["qid"]),
            ),
        )
        for dataset in DATASETS
    }
    proof = sorted(
        (row for row in by_dataset["2wikimultihopqa"] if row["process_reward_eligible"]),
        key=lambda row: (rank("v4-schedule-proof", row["dataset"], row["qid"]), row["qid"]),
    )
    ordinary = sorted(
        (row for row in by_dataset["2wikimultihopqa"] if not row["process_reward_eligible"]),
        key=lambda row: (rank("v4-schedule-ordinary", row["dataset"], row["qid"]), row["qid"]),
    )
    if not (
        len(by_dataset["hotpotqa"]) == 1000
        and len(by_dataset["musique"]) == 1000
        and len(proof) == 800
        and len(ordinary) == 200
    ):
        raise ValueError("v4 schedule requires H1000/M1000/2W proof800+ordinary200")
    two_wiki: list[dict[str, Any]] = []
    for index, ordinary_row in enumerate(ordinary):
        # Every five-question block contains four Proof rows and one ordinary
        # row, avoiding a proof-only prefix while preserving one exposure each.
        start = 4 * index
        two_wiki.extend(
            [dict(proof[start]), dict(proof[start + 1]), dict(ordinary_row),
             dict(proof[start + 2]), dict(proof[start + 3])]
        )
    groups: list[dict[str, Any]] = []
    for index in range(1000):
        for row in (by_dataset["hotpotqa"][index], two_wiki[index], by_dataset["musique"][index]):
            item = dict(row)
            item["prompt_group_index"] = len(groups) + 1
            groups.append(item)
    return groups


def expand_k4(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        for within_group in range(1, K + 1):
            schedule.append(
                {
                    "schema_version": "mixed-ppo-fixed-rollout-schedule-v4-proof800",
                    "rollout_index": len(schedule) + 1,
                    "prompt_group_index": group_index,
                    "within_group_rollout": within_group,
                    "dataset": group["dataset"],
                    "qid": group["qid"],
                    "question_sha256": group["question_sha256"],
                    "stratum": group["route"],
                    "process_reward_eligible": bool(group["process_reward_eligible"]),
                }
            )
    return schedule


def build_weights(
    population: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    exposures = Counter((row["dataset"], row["qid"]) for row in groups)
    rows = [
        {
            "schema_version": "mixed-ppo-rollout-sampling-weight-v4-proof800",
            "dataset": row["dataset"],
            "qid": row["qid"],
            "question_sha256": row["question_sha256"],
            "stratum": row["route"],
            "process_reward_eligible": bool(row["process_reward_eligible"]),
            "scheduled_prompt_group_exposures": exposures[(row["dataset"], row["qid"])],
            "sampling_probability": exposures[(row["dataset"], row["qid"])] / len(groups),
        }
        for row in population
    ]
    if abs(sum(row["sampling_probability"] for row in rows) - 1.0) > 1e-12:
        raise ValueError("v4 sampling weights do not sum to one")
    return rows


def _retrieval_request(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["dataset"] not in {"hotpotqa", "musique"}:
        raise ValueError("only HotpotQA/MuSiQue additions need canonical retrieval")
    return {
        "schema_version": "mixed-ppo-v4-retrieval-request-v1",
        "question_key": row["question_key"],
        "dataset": row["dataset"],
        "qid": row["qid"],
        "question": row["question"],
        "question_sha256": row["question_sha256"],
        "family_version": row["family_version"],
        "family_sha256": row["family_sha256"],
        "role": "rollout_retrieval",
        "stratum": row["stratum"],
        "gold_access": False,
    }


def _load_protocol_output(protocol: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    identity = protocol.get("outputs", {}).get(name)
    if not isinstance(identity, Mapping):
        raise ValueError(f"parent protocol missing outputs.{name}")
    path = Path(str(identity.get("path") or ""))
    if not path.is_file() or sha256_file(path) != str(identity.get("sha256") or ""):
        raise ValueError(f"parent output missing/hash mismatch: {name}")
    return read_jsonl(path)


def load_ordinary200_release(protocol_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the successor ordinary200 release and verify its frozen gates."""

    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != ORDINARY200_PROTOCOL_SCHEMA
        or protocol.get("status") != ORDINARY200_PROTOCOL_STATUS
    ):
        raise ValueError("ordinary200 successor schema/status mismatch")
    counts = ((protocol.get("selection") or {}).get("counts") or {})
    if not counts or not all((counts.get("gates") or {}).values()):
        raise ValueError("ordinary200 successor gates are not all true")
    rows = _load_protocol_output(protocol, "ordinary200")
    if len(rows) != 200:
        raise ValueError(f"ordinary200 successor has {len(rows)} rows")
    return rows, protocol


def load_hm_reconciliation_release(
    protocol_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Load the clean H/M population and its retrieval reuse/delta contract."""

    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != HM_RECONCILIATION_SCHEMA
        or protocol.get("status") != HM_RECONCILIATION_STATUS
    ):
        raise ValueError("H/M reconciliation schema/status mismatch")
    if not all((protocol.get("gates") or {}).values()):
        raise ValueError("H/M reconciliation gates are not all true")
    population = _load_protocol_output(protocol, "hm_population")
    requirements = _load_protocol_output(protocol, "retrieval_requirements")
    new_requests = _load_protocol_output(protocol, "new_retrieval_requests")
    reserve = _load_protocol_output(protocol, "reserve")
    if Counter(row["dataset"] for row in population) != Counter(
        {"hotpotqa": 1000, "musique": 1000}
    ):
        raise ValueError("H/M reconciliation population is not H1000/M1000")
    requirement_keys = {(row["dataset"], row["qid"]) for row in requirements}
    new_keys = {(row["dataset"], row["qid"]) for row in new_requests}
    population_new_keys = {
        (row["dataset"], row["qid"])
        for row in population
        if row.get("source_role") == "new_retrieval"
    }
    if (
        len(requirements) != 823
        or len(requirement_keys) != 823
        or requirement_keys != population_new_keys
        or len(new_requests) != 11
        or not new_keys.issubset(requirement_keys)
    ):
        raise ValueError("H/M retrieval requirements/reuse delta contract drifted")
    return population, requirements, new_requests, reserve, protocol


def _protocol_manifest_extra(protocol_path: Path) -> dict[str, Any]:
    """Return the exact manifest contract consumed by final-v4 materialization."""

    return {
        "phase": "mixed_ppo_v4_answer_free_protocol_freeze",
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    parser.add_argument("--proof_candidates", type=Path, required=True)
    # Kept as no-op compatibility flags for already-written local runbooks.
    # H/M identities now come only from the bound reconciliation protocol.
    parser.add_argument("--hotpot_raw", type=Path, default=Path("data/hotpotqa/train.jsonl"))
    parser.add_argument("--musique_raw", type=Path, default=Path("data/musique/train.jsonl"))
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--ordinary200-protocol", type=Path, default=DEFAULT_ORDINARY200_PROTOCOL
    )
    parser.add_argument(
        "--hm-reconciliation-protocol",
        type=Path,
        default=DEFAULT_HM_RECONCILIATION_PROTOCOL,
    )
    parser.add_argument(
        "--protected-ledger-dir",
        type=Path,
        default=DEFAULT_PROTECTED_LEDGER_DIR,
    )
    parser.add_argument("--historical_cutoff", default=HISTORICAL_CUTOFF)
    parser.add_argument("--reserve_per_stratum", type=int, default=25)
    parser.add_argument("--proof_reserve_per_type", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen protocol: {args.out}")
    if not args.proof_candidates.is_file():
        raise FileNotFoundError(
            "unified ProofKG candidate supply is not materialized: "
            f"{args.proof_candidates}"
        )
    if args.reserve_per_stratum < 0 or args.proof_reserve_per_type < 0:
        raise ValueError("reserve sizes must be nonnegative")
    protected_path, protected_ledger_binding = validate_protected_ledger_release(
        args.protected_ledger_dir
    )
    protected_paths = (protected_path,)
    required = [
        args.parent_protocol,
        args.ordinary200_protocol,
        args.hm_reconciliation_protocol,
        args.replay,
        *protected_paths,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    parent = json.loads(args.parent_protocol.read_text(encoding="utf-8"))
    parent_population = _load_protocol_output(parent, "population")
    ordinary, _ordinary_protocol = load_ordinary200_release(
        args.ordinary200_protocol
    )
    parent_hotpot = [row for row in parent_population if row["dataset"] == "hotpotqa"]
    parent_musique = [row for row in parent_population if row["dataset"] == "musique"]
    if len(parent_hotpot) != 600 or len(parent_musique) != 599:
        raise ValueError("parent H/M counts are not the frozen 600/599")
    (
        hm_population,
        retrieval_requests,
        new_retrieval_requests,
        hm_reserve,
        hm_protocol,
    ) = load_hm_reconciliation_release(args.hm_reconciliation_protocol)
    hotpot = [row for row in hm_population if row["dataset"] == "hotpotqa"]
    musique = [row for row in hm_population if row["dataset"] == "musique"]
    hotpot_new = [
        row for row in hotpot if row.get("source_role") == "new_retrieval"
    ]
    musique_new = [
        row for row in musique if row.get("source_role") == "new_retrieval"
    ]
    hotpot_stats = dict((hm_protocol.get("population") or {}).get("hotpot") or {})
    musique_stats = dict((hm_protocol.get("population") or {}).get("musique") or {})

    protected_rows = _normalise_external_identities(
        row for path in protected_paths for row in read_jsonl(path)
    )
    replay_rows = _normalise_external_identities(read_jsonl(args.replay))
    external = IdentityIndex()
    external.update(protected_rows)
    external.update(replay_rows)

    hm_external_overlap = identity_overlap_counts(
        hm_population, [*protected_rows, *replay_rows]
    )
    if any(hm_external_overlap.values()):
        raise ValueError(
            f"frozen H/M reconciliation overlaps protected/replay: {hm_external_overlap}"
        )

    normalised_ordinary = []
    for row in ordinary:
        item = _identity(
            row,
            route="2wiki_ordinary_outcome",
            eligible=False,
            question_type=str(row.get("question_type") or "unknown"),
            stratum="ordinary",
            source_role=str(row.get("source_role") or "ordinary_outcome"),
        )
        for field in (
            "source_origin",
            "source_line_number",
            "source_record_sha256",
            "source_passages_sha256",
        ):
            if field not in row:
                raise ValueError(f"ordinary200 row lacks source provenance: {field}")
            item[field] = row[field]
        normalised_ordinary.append(item)
    ordinary_external_overlap = identity_overlap_counts(normalised_ordinary, [*protected_rows, *replay_rows])
    if any(ordinary_external_overlap.values()):
        raise ValueError(f"parent ordinary200 overlaps protected/replay: {ordinary_external_overlap}")
    proof_blocked = IdentityIndex()
    proof_blocked.update(protected_rows)
    proof_blocked.update(replay_rows)
    proof_blocked.update(normalised_ordinary)
    proof_candidates, proof_gate_reasons = normalise_proof_candidates(
        read_jsonl(args.proof_candidates), historical_cutoff=args.historical_cutoff
    )
    ordinary_proof_candidate_overlap = identity_overlap_counts(
        normalised_ordinary, proof_candidates
    )
    if any(ordinary_proof_candidate_overlap.values()):
        raise ValueError(
            "ordinary200 overlaps unified Proof candidates: "
            f"{ordinary_proof_candidate_overlap}"
        )
    proof, proof_reserve, proof_stats = select_proof800(
        proof_candidates,
        blocked=proof_blocked,
        reserve_per_stratum=args.proof_reserve_per_type,
    )

    population = [*hotpot, *normalised_ordinary, *proof, *musique]
    population.sort(
        key=lambda row: (
            DATASETS.index(str(row["dataset"])),
            rank("v4-population", str(row["dataset"]), str(row["qid"])),
            str(row["qid"]),
        )
    )
    reserve = [*hm_reserve, *proof_reserve]
    reserve.sort(
        key=lambda row: (
            DATASETS.index(str(row["dataset"])),
            str(row["stratum"]),
            rank("v4-reserve", str(row["dataset"]), str(row["qid"])),
            str(row["qid"]),
        )
    )
    retrieval_requests.sort(
        key=lambda row: (DATASETS.index(str(row["dataset"])), str(row["stratum"]), row["qid"])
    )
    new_retrieval_requests.sort(
        key=lambda row: (DATASETS.index(str(row["dataset"])), str(row["stratum"]), row["qid"])
    )

    groups = build_groups(population)
    schedule = expand_k4(groups)
    weights = build_weights(population, groups)
    population_index = IdentityIndex()
    population_index.update(population)
    reserve_overlap = identity_overlap_counts(reserve, population)
    protected_overlap = identity_overlap_counts(population, protected_rows)
    replay_overlap = identity_overlap_counts(population, replay_rows)
    replay_protected_overlap = identity_overlap_counts(replay_rows, protected_rows)
    dataset_counts = Counter(row["dataset"] for row in population)
    scheduled_counts = Counter(row["dataset"] for row in groups)
    gates = {
        "population_3000": len(population) == 3000,
        "dataset_1000_each": dataset_counts == Counter({dataset: 1000 for dataset in DATASETS}),
        "dataset_scoped_qid_unique": len(population_index.qids) == 3000,
        "dataset_scoped_question_hash_unique": len(population_index.question_hashes) == 3000,
        "protected_qid_hash_family_overlap_zero": not any(protected_overlap.values()),
        "replay_qid_hash_family_overlap_zero": not any(replay_overlap.values()),
        "replay_protected_qid_hash_family_overlap_zero": not any(
            replay_protected_overlap.values()
        ),
        "reserve_population_qid_hash_family_overlap_zero": not any(reserve_overlap.values()),
        "hotpot_retained_583_new_417": hotpot_stats["retained_parent"] == 583
        and hotpot_stats["new_retrieval"] == 417,
        "hotpot_target_cells_exact": Counter(row["stratum"] for row in hotpot)
        == Counter(HOTPOT_TARGET_CELLS),
        "musique_retained_594_new_406": musique_stats["retained_parent"] == 594
        and musique_stats["new_retrieval"] == 406,
        "musique_target_hops_exact": Counter(row["stratum"] for row in musique)
        == Counter(MUSIQUE_TARGET_HOPS),
        "proof800_qtype_200_each": len(proof) == 800
        and Counter(row["question_type"] for row in proof) == Counter(PROOF_TARGET_TYPES),
        "ordinary200_exact": len(normalised_ordinary) == 200,
        "ordinary200_unified_proof_candidate_overlap_zero": not any(
            ordinary_proof_candidate_overlap.values()
        ),
        "retrieval_requirements_h417_m406": Counter(row["dataset"] for row in retrieval_requests)
        == Counter({"hotpotqa": 417, "musique": 406}),
        "new_retrieval_requests_m11": Counter(
            row["dataset"] for row in new_retrieval_requests
        )
        == Counter({"musique": 11}),
        "groups_3000_balanced": len(groups) == 3000
        and scheduled_counts == Counter({dataset: 1000 for dataset in DATASETS}),
        "all_questions_scheduled_once": len(
            {(row["dataset"], row["qid"]) for row in groups}
        )
        == 3000,
        "schedule_k4_12000": len(schedule) == 12000
        and all(
            len({(row["dataset"], row["qid"]) for row in schedule[start : start + K]}) == 1
            for start in range(0, len(schedule), K)
        ),
        "scheduled_graph_groups_800": sum(
            bool(row["process_reward_eligible"]) for row in groups
        )
        == 800,
        "weights_sum_one": abs(sum(row["sampling_probability"] for row in weights) - 1.0)
        <= 1e-12,
        "all_answer_free": all(row["gold_access"] is False for row in population + reserve),
    }
    if not all(gates.values()):
        raise RuntimeError(f"v4 freeze gates failed: {gates}")

    args.out.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "population": args.out / "population.question_only.jsonl",
        "proof800": args.out / "proof800.question_only.jsonl",
        "ordinary200": args.out / "ordinary200.question_only.jsonl",
        "retrieval_requests": args.out / "retrieval_requests.question_only.jsonl",
        "new_retrieval_requests": args.out / "new_retrieval_requests.question_only.jsonl",
        "hotpot_retrieval_requests": args.out / "hotpotqa.retrieval_requests.question_only.jsonl",
        "musique_retrieval_requests": args.out / "musique.retrieval_requests.question_only.jsonl",
        "reserve": args.out / "reserve.question_only.jsonl",
        "sampling_weights": args.out / "sampling_weights.question_only.jsonl",
        "prompt_groups": args.out / "prompt_groups.question_only.jsonl",
        "fixed_rollout_schedule": args.out / "fixed_rollout_schedule.question_only.jsonl",
    }
    rows_by_output = {
        "population": population,
        "proof800": proof,
        "ordinary200": normalised_ordinary,
        "retrieval_requests": retrieval_requests,
        "new_retrieval_requests": new_retrieval_requests,
        "hotpot_retrieval_requests": [
            row for row in retrieval_requests if row["dataset"] == "hotpotqa"
        ],
        "musique_retrieval_requests": [
            row for row in retrieval_requests if row["dataset"] == "musique"
        ],
        "reserve": reserve,
        "sampling_weights": weights,
        "prompt_groups": groups,
        "fixed_rollout_schedule": schedule,
    }
    for name, path in output_paths.items():
        write_jsonl(path, rows_by_output[name])

    report = {
        "schema_version": "mixed-ppo-three-dataset-protocol-v4-proof800",
        "experiment_id": EXPERIMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "population": {
            "unique_total": len(population),
            "unique_by_dataset": dict(sorted(dataset_counts.items())),
            "hotpot": hotpot_stats,
            "musique": musique_stats,
            "2wiki": {
                "proof": proof_stats,
                "proof_gate_reasons": proof_gate_reasons,
                "ordinary": len(normalised_ordinary),
                "ordinary_unified_proof_candidate_overlap": ordinary_proof_candidate_overlap,
            },
            "retrieval_requests_by_dataset": dict(
                sorted(Counter(row["dataset"] for row in retrieval_requests).items())
            ),
            "new_retrieval_requests_by_dataset": dict(
                sorted(Counter(row["dataset"] for row in new_retrieval_requests).items())
            ),
        },
        "isolation": {
            "scope": "dataset-scoped",
            "keys": ["qid", "question_sha256", "family_sha256"],
            "protected": protected_overlap,
            "replay": replay_overlap,
            "replay_protected": replay_protected_overlap,
            "reserve_population": reserve_overlap,
            "train_side_family_repeats": (
                "allowed only inside the template-heavy 2Wiki Proof stratum; "
                "qid and exact question hashes remain unique"
            ),
        },
        "schedule": {
            "prompt_groups": len(groups),
            "rollouts_per_prompt": K,
            "trajectories": len(schedule),
            "groups_by_dataset": dict(sorted(scheduled_counts.items())),
            "proof_groups": sum(bool(row["process_reward_eligible"]) for row in groups),
        },
        "reserve": {
            "hotpot_musique_per_stratum": args.reserve_per_stratum,
            "proof_per_question_type": args.proof_reserve_per_type,
            "total": len(reserve),
            "by_dataset": dict(sorted(Counter(row["dataset"] for row in reserve).items())),
            "identity_only_not_retrieved_not_scheduled": True,
        },
        "gates": gates,
        "protected_ledger": {
            "version": PROTECTED_LEDGER_SCHEMA,
            "complete": True,
            "current_family_recomputed": True,
            **protected_ledger_binding,
        },
        "scientific_boundary": {
            "answer_free_freeze": True,
            "raw_hm_source_files_decoded_in_this_stage": False,
            "upstream_hm_reconciliation_may_decode_raw_source_objects": True,
            "gold_fields_accessed_for_selection": False,
            "upstream_hm_selection_metadata": {
                "hotpotqa": ["metadata.type", "metadata.level"],
                "musique": ["len(metadata.metadata.question_decomposition)"],
            },
            "supporting_sentence_content_accessed": False,
            "evaluation_outputs_read": False,
            "retrieval_run": False,
            "training_started": False,
            "proof_candidate_policy": (
                "Only records passing trajectory-source-gate-hard-mask-v1 are selectable."
            ),
            "old_assets_overwritten": False,
        },
        "inputs": {
            "parent_protocol": ref(args.parent_protocol),
            "ordinary200_successor_protocol": ref(args.ordinary200_protocol),
            "hm_reconciliation_protocol": ref(args.hm_reconciliation_protocol),
            "proof_candidates": ref(args.proof_candidates),
            "replay": ref(args.replay),
            "protected": [ref(path) for path in protected_paths],
            "historical_cutoff": args.historical_cutoff,
        },
        "outputs": {name: ref(path) for name, path in output_paths.items()},
    }
    protocol_path = args.out / "protocol.json"
    protocol_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dump_manifest(
        args.out,
        status=STATUS,
        extra=_protocol_manifest_extra(protocol_path),
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "population": report["population"],
                "schedule": report["schedule"],
                "gates": gates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
