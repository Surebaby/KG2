from kgproweight.data.parsers import parse_steps
from kgproweight.data.prompts import build_saeg_sft_messages, format_passage_evidence_block
from kgproweight.data.saeg_parsers import parse_saeg_steps


KG = [["Ed Wood", "occupation", "film director"]]
PASSAGE_EVIDENCE = [{
    "passage_id": "P1",
    "title": "Ed Wood",
    "sentence": "Ed Wood was an American filmmaker.",
}]


def _trace(passage="[P1]", knowledge="[(Ed Wood, occupation, film director)]"):
    return f"""[Step 1]
Reasoning: The supplied evidence identifies Ed Wood's occupation.
Knowledge Used: {knowledge}
Passage Used: {passage}
Conclusion: Ed Wood was a film director.
[Final Answer]
film director"""


def test_prompt_keeps_passages_out_of_knowledge_graph():
    messages = build_saeg_sft_messages(
        "What was Ed Wood's occupation?",
        [{"contents": "Ed Wood\nEd Wood was an American filmmaker."}],
        KG,
        PASSAGE_EVIDENCE,
    )
    user = messages[1]["content"]
    assert "(Ed Wood, occupation, film director)" in user
    assert "[P1] Ed Wood\nSentence: Ed Wood was an American filmmaker." in user
    assert "(Ed Wood, evidence sentence," not in user


def test_passage_formatter_never_serializes_pseudo_triple():
    block = format_passage_evidence_block(PASSAGE_EVIDENCE)
    assert block.startswith("[P1] Ed Wood")
    assert "evidence sentence" not in block


def test_saeg_parser_accepts_independent_known_citations():
    parsed = parse_saeg_steps(
        _trace(), known_kg=KG, known_passage_ids=["P1"]
    )
    assert len(parsed) == 1
    assert parsed[0].citation_contract_valid
    assert parsed[0].cited_triples == [("Ed Wood", "occupation", "film director")]
    assert parsed[0].cited_passage_ids == ["P1"]


def test_saeg_parser_rejects_unknown_or_malformed_passage_ids():
    unknown = parse_saeg_steps(_trace("[P2]"), known_kg=KG, known_passage_ids=["P1"])[0]
    malformed = parse_saeg_steps(_trace("P1"), known_kg=KG, known_passage_ids=["P1"])[0]
    assert not unknown.citation_contract_valid
    assert unknown.unknown_passage_ids == ["P2"]
    assert "unknown_passage_id" in unknown.citation_contract_errors
    assert not malformed.citation_contract_valid
    assert "passage_used_must_be_bracketed_list" in malformed.citation_contract_errors


def test_saeg_parser_requires_exactly_one_passage_used_field():
    missing = _trace().replace("Passage Used: [P1]\n", "")
    parsed = parse_saeg_steps(missing, known_kg=KG, known_passage_ids=["P1"])[0]
    assert parsed.passage_used_field_count == 0
    assert not parsed.citation_contract_valid


def test_legacy_parser_contract_is_unchanged():
    parsed = parse_steps(_trace(), known_kg=KG)
    assert len(parsed) == 1
    assert parsed[0].knowledge_used_valid
    assert parsed[0].cited_triples == [("Ed Wood", "occupation", "film director")]

