from scripts.pilot.audit_historical_wikidata_value_conflicts import (
    extract_claim_values,
    reference_matches,
)


def test_extract_historical_entity_and_literal_values() -> None:
    entity = {
        "claims": {
            "P26": [{
                "rank": "normal",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q2"}},
                },
            }],
            "P569": [{
                "rank": "normal",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {
                        "type": "time",
                        "value": {"time": "+1911-12-22T00:00:00Z", "precision": 11},
                    },
                },
            }],
        }
    }
    spouse = extract_claim_values(entity, "P26")
    born = extract_claim_values(entity, "P569")
    assert spouse == [{"tail_qid": "Q2", "literal": None}]
    assert reference_matches(spouse, expected_id="Q2", expected_label="Person")
    assert born == [{"tail_qid": None, "literal": "1911-12-22"}]
    assert reference_matches(born, expected_id="22 December 1911", expected_label="22 December 1911")
    assert not reference_matches(born, expected_id="22 December 1915", expected_label="22 December 1915")
