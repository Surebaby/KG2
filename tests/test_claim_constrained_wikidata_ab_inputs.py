from scripts.prepare.build_claim_constrained_wikidata_ab_inputs import merge_triples


def test_merge_triples_prefers_new_claims_and_is_bounded():
    proof = [["A", "new relation", "B"]]
    legacy = [["A", "old relation", "C"], ["A", "new relation", "B"]]
    assert merge_triples(proof, legacy, cap=2) == [
        ["A", "new relation", "B"], ["A", "old relation", "C"]
    ]


def test_merge_triples_rejects_malformed_values():
    assert merge_triples([["A", "r"], ["A", "r", ""]], []) == []
