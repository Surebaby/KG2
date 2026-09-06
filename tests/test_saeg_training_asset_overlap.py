from scripts.prepare.audit_saeg_training_asset_overlap import (
    jaccard,
    normalise_text,
    passage_body,
    passage_signatures,
    passage_title,
)
from scripts.prepare.freeze_qpeg_v1_protocol import family_sha256


def test_normalise_text_is_unicode_case_and_space_stable():
    assert normalise_text("  Molière\n IS  Here ") == "molière is here"


def test_passage_signature_ignores_id_source_and_repeated_title_line():
    left = {"id": "a", "title": "Alpha", "contents": "Alpha\nFirst fact.", "source": "raw"}
    right = {"id": "b", "title": "alpha", "contents": "alpha\n First   fact. ", "source": "other"}
    assert passage_title(left) == passage_title(right)
    assert passage_body(left) == passage_body(right)
    assert passage_signatures([left]) == passage_signatures([right])


def test_passage_title_falls_back_to_first_contents_line():
    passage = {"contents": "Fallback Title\nBody"}
    assert passage_title(passage) == "fallback title"
    assert passage_body(passage) == "body"


def test_jaccard_handles_empty_and_partial_sets():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_cross_protocol_family_must_be_recomputed_from_question():
    assert family_sha256("Who is the mother of Ada Lovelace?") == family_sha256(
        "Who is the mother of Grace Hopper?"
    )
