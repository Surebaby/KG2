"""Pure, opt-in pretraining source verification; never repairs frozen evidence.

PASS means the supplied, hash-bound evidence supports every checked identity and
display binding. It does not replace the legacy hard gate, source provenance
audits, or independent semantic validation. UNVERIFIED is distinct from a false claim:
an absent alias is not proof that a surface is wrong. Both FAIL and UNVERIFIED
deny clearance. This module neither reads Gold nor changes the legacy hard gate.

Evidence has schema ``qid-source-evidence-v1`` and ``entities[qid]`` containing
``labels``, ``aliases``, ``demonyms``, ``bindings`` and optional ``typed_edges``.
Typed edges must be replayed from frozen source records, not inferred by voting
over display strings. Their bindings identify the original store/cache files.
The caller is responsible for checking these hashes against the actual files.
"""
from __future__ import annotations

from collections.abc import Mapping
import re
import unicodedata
from typing import Any


SOURCE_INTEGRITY_VERSION = "source-integrity-clearance-v1"
EVIDENCE_VERSION = "qid-source-evidence-v1"
_QID = re.compile(r"Q[1-9][0-9]*\Z")
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _bindings(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        return {}
    if any(not isinstance(k, str) or not k or not isinstance(v, str)
           or not _SHA.fullmatch(v) for k, v in value.items()):
        return {}
    return dict(value)


def validate_source_integrity_v1(record: Mapping[str, Any],
                                 evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Verify supplied names and typed execution; return a fail-closed decision.

    This function performs no I/O and does not mutate either argument. A FAIL is
    an internal identity/execution contradiction, not a blanket judgment about
    benchmark labels. Missing provenance or unsupported names are UNVERIFIED.
    Every non-PASS decision has ``clearance is False``.
    """
    checks: list[dict[str, Any]] = []
    used_entities: dict[str, dict[str, str]] = {}
    used_edges: list[dict[str, str]] = []
    displayed_tail_identities: dict[str, set[str]] = {}
    record = record if isinstance(record, Mapping) else {}
    evidence = evidence if isinstance(evidence, Mapping) else {}
    entities = evidence.get("entities")
    entities = entities if isinstance(entities, Mapping) else {}
    global_bindings = _bindings(evidence.get("bindings"))

    def add(status: str, reason: str, **fields: Any) -> None:
        checks.append({"status": status, "reason": reason, **fields})

    def surface(qid: Any, text: Any, location: str) -> None:
        if not isinstance(qid, str) or not _QID.fullmatch(qid):
            add("UNVERIFIED", "missing_or_invalid_qid", location=location)
            return
        source = entities.get(qid)
        if not isinstance(source, Mapping):
            add("UNVERIFIED", "qid_evidence_missing", location=location, qid=qid)
            return
        bound = _bindings(source.get("bindings"))
        if not bound:
            add("UNVERIFIED", "qid_evidence_bindings_missing", location=location, qid=qid)
            return
        used_entities[qid] = bound
        names: set[str] = set()
        for field in ("labels", "aliases", "demonyms"):
            values = source.get(field, [])
            if not isinstance(values, (list, tuple)) or any(not isinstance(x, str) for x in values):
                add("UNVERIFIED", "qid_name_evidence_malformed", location=location, qid=qid)
                return
            names.update(_norm(x) for x in values if x.strip())
        if not isinstance(text, str) or not text.strip():
            add("UNVERIFIED", "display_surface_missing", location=location, qid=qid)
        elif not names:
            add("UNVERIFIED", "qid_name_evidence_uncovered", location=location, qid=qid)
        elif _norm(text) == _norm(qid):
            add("PASS", "exact_qid_identity_preserved", location=location, qid=qid)
        elif _norm(text) in names:
            add("PASS", "qid_bound_surface_supported", location=location, qid=qid)
        else:
            add("UNVERIFIED", "surface_not_supported_by_bound_qid_names",
                location=location, qid=qid, surface=text)

    if evidence.get("schema_version") != EVIDENCE_VERSION:
        add("UNVERIFIED", "evidence_schema_missing_or_unsupported")
    if global_bindings:
        add("PASS", "evidence_bindings_present")
    else:
        add("UNVERIFIED", "evidence_bindings_missing")
    execution = record.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    complete = execution.get("complete_plan_execution")
    if complete is True:
        add("PASS", "execution_declared_complete")
    elif complete is False:
        add("FAIL", "execution_declared_incomplete")
    else:
        add("UNVERIFIED", "execution_completeness_unverified")
    anchors = execution.get("anchor_entities")
    if not isinstance(anchors, Mapping) or not anchors:
        add("UNVERIFIED", "root_identity_evidence_missing")
    else:
        for anchor, entity in anchors.items():
            if not isinstance(entity, Mapping):
                add("UNVERIFIED", "root_binding_malformed")
                continue
            requested = [anchor, entity.get("surface"), entity.get("resolved_surface")]
            for index, text in enumerate(dict.fromkeys(t for t in requested if isinstance(t, str) and t.strip())):
                surface(entity.get("qid"), text, "root:" + str(anchor) + f"/surface:{index}")
    hops = execution.get("hops")
    if not isinstance(hops, list) or not hops:
        add("UNVERIFIED", "execution_hops_missing")
        hops = []
    total_matches = 0
    execution_triples: set[tuple[str, str, str]] = set()
    for hop_number, hop in enumerate(hops, 1):
        loc = f"hop:{hop_number}"
        if not isinstance(hop, Mapping):
            add("UNVERIFIED", "hop_malformed", location=loc)
            continue
        matches, sources = hop.get("matches"), hop.get("match_sources")
        if not isinstance(matches, list) or not matches:
            add("UNVERIFIED", "hop_matches_missing", location=loc)
            continue
        if not isinstance(sources, list) or len(sources) != len(matches):
            add("UNVERIFIED", "match_source_alignment_missing", location=loc)
            continue
        inputs, pids = hop.get("input_entities"), hop.get("pids")
        if not isinstance(inputs, list) or not inputs or not isinstance(pids, list) or not pids:
            add("UNVERIFIED", "hop_typed_inputs_missing", location=loc)
            continue
        available: list[tuple[str, Mapping[str, Any]]] = []
        for entity in inputs:
            qid = entity.get("qid") if isinstance(entity, Mapping) else None
            if not isinstance(qid, str) or not _QID.fullmatch(qid):
                add("UNVERIFIED", "hop_input_identity_missing", location=loc)
                continue
            for field in ("surface", "resolved_surface"):
                if entity.get(field):
                    surface(qid, entity.get(field), loc + "/input/" + field)
            source = entities.get(qid)
            edges = source.get("typed_edges") if isinstance(source, Mapping) else None
            if not isinstance(edges, list):
                add("UNVERIFIED", "typed_edge_replay_missing", location=loc, qid=qid)
                continue
            available.extend((qid, edge) for edge in edges if isinstance(edge, Mapping)
                             and edge.get("pid") in pids)
        for match_index, (match, source) in enumerate(zip(matches, sources), 1):
            match_loc = f"{loc}/match:{match_index}"
            total_matches += 1
            if not isinstance(match, (list, tuple)) or len(match) != 3 or any(not isinstance(x, str) for x in match):
                add("UNVERIFIED", "display_match_malformed", location=match_loc)
                continue
            execution_triples.add(tuple(match))
            candidates = [(qid, edge) for qid, edge in available
                          if [edge.get("head_label"), edge.get("relation"), edge.get("tail_value")] == list(match)
                          and edge.get("source") == source]
            if not candidates:
                add("UNVERIFIED", "display_match_not_replayed", location=match_loc)
                continue
            if any(not isinstance(edge.get("head_qid"), str)
                   or not isinstance(edge.get("pid"), str)
                   or (edge.get("tail_qid") is not None
                       and not isinstance(edge.get("tail_qid"), str))
                   for _, edge in candidates):
                add("UNVERIFIED", "typed_edge_identity_malformed", location=match_loc)
                continue
            for _, edge in candidates:
                if edge.get("tail_qid"):
                    displayed_tail_identities.setdefault(_norm(match[2]), set()).add(edge["tail_qid"])
            identities = {(edge.get("head_qid"), edge.get("pid"), edge.get("tail_qid"),
                           None if edge.get("tail_qid") else edge.get("tail_value"))
                          for _, edge in candidates}
            if len(identities) != 1:
                add("FAIL", "distinct_typed_identities_collapsed_to_display", location=match_loc,
                    distinct_identities=len(identities))
                continue
            for input_qid, edge in candidates:
                if edge.get("head_qid") != input_qid:
                    add("FAIL", "typed_edge_head_disagrees_with_input", location=match_loc)
                bound = _bindings(edge.get("bindings"))
                if not bound:
                    add("UNVERIFIED", "typed_edge_bindings_missing", location=match_loc)
                else:
                    used_edges.append(bound)
                surface(edge.get("head_qid"), edge.get("head_label"), match_loc + "/head")
                if edge.get("tail_qid") is not None:
                    surface(edge.get("tail_qid"), edge.get("tail_value"), match_loc + "/tail")
                elif isinstance(edge.get("tail_value"), str) and edge["tail_value"].strip():
                    add("PASS", "bound_literal_projection_replayed", location=match_loc + "/tail")
                else:
                    add("UNVERIFIED", "literal_projection_missing", location=match_loc + "/tail")
    if not total_matches:
        add("UNVERIFIED", "no_graph_matches_verified")
    graph = record.get("kg_subgraph")
    if not isinstance(graph, list) or not graph:
        add("UNVERIFIED", "visible_graph_missing")
    elif any(not isinstance(t, (list, tuple)) or len(t) != 3 or any(not isinstance(v, str) for v in t) for t in graph):
        add("UNVERIFIED", "visible_graph_malformed")
    elif {tuple(v.strip() for v in t) for t in graph} != {
            tuple(v.strip() for v in t) for t in execution_triples}:
        add("FAIL", "visible_graph_disagrees_with_execution_matches")
    else:
        add("PASS", "visible_graph_matches_execution_projection")
    for display, qids in displayed_tail_identities.items():
        if len(qids) > 1:
            add("FAIL", "distinct_tail_qids_share_display_across_record",
                surface=display, distinct_qids=sorted(qids))
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else (
        "UNVERIFIED" if any(c["status"] == "UNVERIFIED" for c in checks) else "PASS")
    return {"schema_version": SOURCE_INTEGRITY_VERSION, "status": status,
            "clearance": status == "PASS", "checks": checks,
            "bindings": {"evidence": global_bindings, "entities": used_entities,
                         "typed_edges": used_edges},
            "semantic_correctness_claim": False, "gold_used": False,
            "legacy_gate_modified": False}
