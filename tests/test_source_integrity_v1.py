"""Independent synthetic checks for QID-bound source projection integrity."""
from copy import deepcopy

import pytest

from kgproweight.reward.source_integrity_v1 import validate_source_integrity_v1


def _fixture():
    bindings = {"synthetic/names.json": "a" * 64}
    root = {"qid": "Q101", "label": "Ada", "surface": "Ada", "resolved_surface": "Ada"}
    triples = [["Ada", "born in", "Northland"], ["Northland", "instance of", "Country"]]
    record = {
        "dataset": "synthetic", "qid": "synthetic-source-question",
        "kg_subgraph": deepcopy(triples),
        "provenance": {"gold_access": False, "complete_plan_execution": True},
        "execution": {
            "complete_plan_execution": True, "anchor_entities": {"Ada": deepcopy(root)},
            "hops": [
                {"hop_index": 1, "subject": "Ada", "input_entities": [deepcopy(root)],
                 "pids": ["P19"], "matches": [triples[0]], "match_sources": ["store"]},
                {"hop_index": 2, "subject": "$hop_1", "input_entities": [{
                    "qid": "Q201", "label": "Northland", "surface": "Northland",
                    "resolved_surface": "Northland"}],
                 "pids": ["P31"], "matches": [triples[1]], "match_sources": ["store"]},
            ],
        },
    }
    edge_bindings = {"synthetic/typed_edges.json": "b" * 64}
    edge1 = {"head_qid": "Q101", "pid": "P19", "relation": "born in", "head_label": "Ada",
             "tail_qid": "Q201", "tail_value": "Northland", "source": "store",
             "bindings": deepcopy(edge_bindings)}
    edge2 = {"head_qid": "Q201", "pid": "P31", "relation": "instance of", "head_label": "Northland",
             "tail_qid": "Q301", "tail_value": "Country", "source": "store",
             "bindings": deepcopy(edge_bindings)}
    entities = {
        "Q101": {"labels": ["Ada"], "aliases": ["A. Example"], "demonyms": [],
                 "bindings": deepcopy(bindings), "typed_edges": [edge1]},
        "Q201": {"labels": ["Northland"], "aliases": ["Northern Country"], "demonyms": ["Northern"],
                 "bindings": deepcopy(bindings), "typed_edges": [edge2]},
        "Q301": {"labels": ["Country"], "aliases": [], "demonyms": [],
                 "bindings": deepcopy(bindings), "typed_edges": []},
    }
    evidence = {"schema_version": "qid-source-evidence-v1", "bindings": {**bindings, **edge_bindings},
                "entities": entities}
    return record, evidence


def _assert_status(record, evidence, status):
    before = deepcopy((record, evidence))
    result = validate_source_integrity_v1(record, evidence)
    assert result["status"] == status
    assert result["clearance"] is (status == "PASS")
    assert isinstance(result["checks"], list)
    assert result.get("schema_version")
    assert isinstance(result.get("bindings"), dict)
    assert (record, evidence) == before
    return result


def test_bound_root_names_and_complete_unique_typed_path_pass_without_mutation():
    _assert_status(*_fixture(), "PASS")


def test_original_unknown_root_mention_cannot_be_hidden_by_valid_resolved_surface():
    record, evidence = _fixture()
    root = record["execution"]["anchor_entities"]["Ada"]
    root.update(surface="Original ambiguous mention", label="Untrusted cached label")
    _assert_status(record, evidence, "UNVERIFIED")


def test_untrusted_cached_root_label_is_not_used_as_identity_authority():
    record, evidence = _fixture()
    record["execution"]["anchor_entities"]["Ada"]["label"] = "Untrusted cached label"
    _assert_status(record, evidence, "PASS")


def test_requested_junior_cannot_receive_clearance_from_supported_senior_identity():
    record, evidence = _fixture()
    root = record["execution"]["anchor_entities"].pop("Ada")
    root.update(surface="Alex Jr", resolved_surface="Alex Sr", label="Alex Sr")
    record["execution"]["anchor_entities"]["Alex Jr"] = root
    evidence["entities"]["Q101"]["aliases"].append("Alex Sr")
    _assert_status(record, evidence, "UNVERIFIED")


@pytest.mark.parametrize("field", ["surface", "resolved_surface"])
def test_unknown_hop_input_surface_cannot_be_hidden_by_supported_edge_head(field):
    record, evidence = _fixture()
    record["execution"]["hops"][1]["input_entities"][0][field] = "Unknown input identity"
    _assert_status(record, evidence, "UNVERIFIED")


def test_empty_visible_graph_with_complete_execution_is_unverified():
    record, evidence = _fixture()
    record["kg_subgraph"] = []
    _assert_status(record, evidence, "UNVERIFIED")


@pytest.mark.parametrize("mutation", ["extra", "different"])
def test_visible_graph_must_equal_verified_execution_matches(mutation):
    record, evidence = _fixture()
    if mutation == "extra":
        record["kg_subgraph"].append(["Unsupported", "links to", "Fact"])
    else:
        record["kg_subgraph"][0][2] = "Different country"
    _assert_status(record, evidence, "FAIL")


def test_visible_graph_field_whitespace_does_not_create_projection_mismatch():
    record, evidence = _fixture()
    record["kg_subgraph"] = [[" " + value + "\t" for value in triple] for triple in record["kg_subgraph"]]
    _assert_status(record, evidence, "PASS")


@pytest.mark.parametrize("surface,field", [("Northern Country", "aliases"), ("Northern", "demonyms")])
def test_tail_aliases_and_demonyms_must_be_bound_to_the_actual_tail_qid(surface, field):
    record, evidence = _fixture()
    record["execution"]["hops"] = record["execution"]["hops"][:1]
    record["kg_subgraph"] = [["Ada", "born in", surface]]
    record["execution"]["hops"][0]["matches"] = deepcopy(record["kg_subgraph"])
    evidence["entities"]["Q101"]["typed_edges"][0]["tail_value"] = surface
    assert surface in evidence["entities"]["Q201"][field]
    _assert_status(record, evidence, "PASS")


@pytest.mark.parametrize("mutation", ["root_name", "tail_name", "head_name", "typed_edge", "root_entity"])
def test_missing_identity_support_is_unverified_instead_of_a_semantic_error(mutation):
    record, evidence = _fixture()
    if mutation == "root_name":
        record["execution"]["anchor_entities"]["Ada"]["resolved_surface"] = "Unknown root alias"
    elif mutation == "tail_name":
        evidence["entities"]["Q201"].update(labels=[], aliases=[], demonyms=[])
    elif mutation == "head_name":
        evidence["entities"]["Q101"].update(labels=[], aliases=[], demonyms=[])
    elif mutation == "typed_edge":
        evidence["entities"]["Q101"]["typed_edges"] = []
    elif mutation == "root_entity":
        evidence["entities"].pop("Q101")
    _assert_status(record, evidence, "UNVERIFIED")


@pytest.mark.parametrize("scope", ["release", "entity", "edge"])
@pytest.mark.parametrize("binding", [{}, {"source": "not-a-sha"}, {"source": "z" * 64}])
def test_missing_or_malformed_evidence_hash_binding_is_unverified(scope, binding):
    record, evidence = _fixture()
    target = evidence if scope == "release" else evidence["entities"]["Q101"]
    if scope == "edge":
        target = target["typed_edges"][0]
    target["bindings"] = deepcopy(binding)
    _assert_status(record, evidence, "UNVERIFIED")


def test_same_rendered_tail_cannot_collapse_two_distinct_qids():
    record, evidence = _fixture()
    evidence["entities"]["Q202"] = deepcopy(evidence["entities"]["Q201"])
    duplicate = deepcopy(evidence["entities"]["Q101"]["typed_edges"][0])
    duplicate["tail_qid"] = "Q202"
    evidence["entities"]["Q101"]["typed_edges"].append(duplicate)
    _assert_status(record, evidence, "FAIL")


def test_typed_edge_cannot_claim_a_different_head_qid_than_its_retrieval():
    record, evidence = _fixture()
    evidence["entities"]["Q101"]["typed_edges"][0]["head_qid"] = "Q999"
    _assert_status(record, evidence, "FAIL")


def test_different_hops_cannot_project_distinct_tail_qids_to_one_surface():
    record, evidence = _fixture()
    record["kg_subgraph"][1][2] = "Northland"
    record["execution"]["hops"][1]["matches"] = [deepcopy(record["kg_subgraph"][1])]
    evidence["entities"]["Q201"]["typed_edges"][0]["tail_value"] = "Northland"
    evidence["entities"]["Q301"]["labels"] = ["Northland"]
    _assert_status(record, evidence, "FAIL")


@pytest.mark.parametrize("field,value", [("head_qid", {"qid": "Q101"}),
                                        ("pid", ["P19"]), ("tail_qid", {"qid": "Q201"})])
def test_malformed_typed_identity_returns_unverified_without_type_error(field, value):
    record, evidence = _fixture()
    evidence["entities"]["Q101"]["typed_edges"][0][field] = value
    _assert_status(record, evidence, "UNVERIFIED")


def test_unknown_tail_surface_is_unverified_even_if_another_qid_supports_the_name():
    record, evidence = _fixture()
    evidence["entities"]["Q201"].update(labels=["Southland"], aliases=[], demonyms=[])
    evidence["entities"]["Q999"] = {
        "labels": ["Northland"], "aliases": [], "demonyms": [],
        "bindings": {"synthetic/names.json": "a" * 64}, "typed_edges": []}
    _assert_status(record, evidence, "UNVERIFIED")


def test_absent_graph_is_unverified_and_cannot_receive_clearance():
    record, evidence = _fixture()
    record["kg_subgraph"] = []
    record["execution"]["hops"] = []
    record["execution"]["anchor_entities"] = {}
    _assert_status(record, evidence, "UNVERIFIED")


def test_incomplete_execution_cannot_receive_clearance():
    record, evidence = _fixture()
    record["execution"]["complete_plan_execution"] = False
    record["provenance"]["complete_plan_execution"] = False
    result = validate_source_integrity_v1(record, evidence)
    assert result["status"] != "PASS"
    assert result["clearance"] is False


class _GoldReadForbidden(dict):
    def get(self, key, *args):
        if key in {"gold", "gold_answer", "answer", "answer_aliases"}:
            raise AssertionError("source validator read Gold")
        return super().get(key, *args)

    def __getitem__(self, key):
        if key in {"gold", "gold_answer", "answer", "answer_aliases"}:
            raise AssertionError("source validator read Gold")
        return super().__getitem__(key)


def test_source_clearance_does_not_read_gold_fields():
    record, evidence = _fixture()
    record = _GoldReadForbidden(record)
    record.update(gold="UNUSED", gold_answer="UNUSED", answer="UNUSED", answer_aliases=["UNUSED"])
    result = validate_source_integrity_v1(record, evidence)
    assert result["status"] == "PASS"
    assert result["clearance"] is True
