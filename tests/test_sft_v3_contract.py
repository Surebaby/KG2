"""Reject malformed supervision and hidden evidence/token truncation paths."""

from copy import deepcopy

import pytest

from kgproweight.data.prompts import SFT_SYSTEM_PROMPT, build_sft_messages
from kgproweight.data.sft_v3_contract import (
    EVIDENCE_SCHEMA_VERSION,
    SFT_V3_SYSTEM_PROMPT,
    build_sft_v3_messages,
    tokenize_frozen_sft_v3_record,
    tokenize_sft_v3_example,
    validate_sft_v3_evidence_sidecar,
    validate_sft_v3_trace,
    visible_passages_v3,
)


def passages():
    return [{"id": str(i), "contents": f"Evidence passage {i} gives a fact about subject {i}."}
            for i in range(1, 11)]


def trace(n=2):
    blocks = [
        f"[Step {i}]\nReasoning: Evidence passage {i} identifies the relevant subject {i}.\n"
        f"Knowledge Used: []\nConclusion: Subject {i} is identified."
        for i in range(1, n + 1)
    ]
    return "\n\n".join(blocks) + "\n\n[Final Answer]\nExample"


def sidecar():
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "steps": [
            {"step_index": 1, "supports": [{"passage_index": 1, "quote": "Evidence passage 1"}],
             "derivation_from_steps": []},
            {"step_index": 2, "supports": [], "derivation_from_steps": [1]},
        ],
    }


class Tokenizer:
    """Whitespace tokens plus explicit role/EOT tokens, for boundary tests."""
    def __init__(self):
        self.vocab = {}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "".join(f"<{m['role']}> {m['content']} <eot> " for m in messages)
        return text + ("<assistant> " if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        for word in text.split():
            self.vocab.setdefault(word, len(self.vocab) + 1)
        return {"input_ids": [self.vocab[word] for word in text.split()]}


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_natural_two_through_five_steps_are_allowed(n):
    result = validate_sft_v3_trace(trace(n))
    assert result["valid"]
    assert result["step_count"] == n
    assert result["final_answer"] == "Example"
    assert result["semantic_grounding_verified"] is False


@pytest.mark.parametrize("bad", [
    trace(1), trace(6),
    trace().replace("[Step 2]", "[Step 3]"),
    trace().replace("[Step 2]", "[Step 1]"),
    trace().replace("[Step 2]", "[step 2]"),
    trace().replace("[Step 2]", "[Step 2"),
    "I will think.\n" + trace(),
    trace().replace("Knowledge Used: []", "Knowledge Used: []\nKnowledge Used: []", 1),
    trace().replace("Reasoning: Evidence passage 1 identifies the relevant subject 1.", "Reasoning: Short"),
    trace().replace("Conclusion: Subject 1 is identified.", "Conclusion: ---"),
    trace().replace("Knowledge Used: []", "Knowledge Used: [,]", 1),
    trace().replace("Knowledge Used: []", "Knowledge Used: [(invented, fact, here)]", 1),
    trace().replace("[Final Answer]\nExample", "[Final Answer]Example"),
    trace().replace("\n\n[Final Answer]", "[Final Answer]"),
    trace().replace("[Final Answer]\nExample", "[Final Answer]\n---"),
    trace() + "\n[Final Answer]\nAnother",
    trace() + "\nThis is an explanation.",
    trace() + "\n[Step 3]",
    trace().replace("Example", "<|eot_id|> Example"),
    trace().replace("\n", "\r\n"),
    trace().replace("Evidence passage 2 identifies the relevant subject 2.",
                    "Evidence passage 1 identifies the relevant subject 1."),
])
def test_reject_malformed_or_ambiguous_supervision(bad):
    assert not validate_sft_v3_trace(bad)["valid"]


def test_exact_kg_surfaces_with_commas_are_preserved():
    kg = [("Washington, D.C.", "located in", "United States")]
    target = trace().replace("Knowledge Used: []", "Knowledge Used: [(Washington, D.C., located in, United States)]", 1)
    result = validate_sft_v3_trace(target, known_kg=kg)
    assert result["valid"]
    assert result["steps"][0]["cited_triples"] == [list(kg[0])]


def test_prompt_is_opt_in_with_exact_legacy_user_evidence():
    original = build_sft_messages(question="A question?", retrieved_passages=passages(), kg_triples=[], top_k=10)
    new = build_sft_v3_messages(question="A question?", retrieved_passages=passages())
    assert original[0]["content"] == SFT_SYSTEM_PROMPT
    assert new[0]["content"] == SFT_V3_SYSTEM_PROMPT
    assert new[1] == original[1]


@pytest.mark.parametrize("bad_passages", [passages()[:9], passages() + [passages()[0]],
                                           passages()[:9] + [{"contents": "  "}]])
def test_ten_nonempty_passages_are_mandatory(bad_passages):
    with pytest.raises(ValueError):
        build_sft_v3_messages(question="Q", retrieved_passages=bad_passages)


def test_annotation_fields_cannot_be_passed_to_evidence_builder():
    bad = passages()
    bad[0]["metadata"] = {"gold_answer": "never enter teacher prompt"}
    with pytest.raises(ValueError, match="annotation key"):
        build_sft_v3_messages(question="Q", retrieved_passages=bad)


def test_no_kg_truncation():
    with pytest.raises(ValueError, match="at most 12"):
        build_sft_v3_messages(question="Q", retrieved_passages=passages(),
                              kg_triples=[(f"entity{i}", "relation", "value") for i in range(13)])


def test_quote_or_prior_inference_sidecar_is_mechanical_only():
    result = validate_sft_v3_evidence_sidecar(sidecar(), trace=trace(), retrieved_passages=passages())
    assert result["valid"]
    assert result["quote_count"] == 1
    assert result["semantic_grounding_verified"] is False


def test_quote_in_raw_hidden_tail_is_rejected():
    rows = passages()
    rows[0]["contents"] = "x" * 1200 + " HIDDEN TAIL"
    proposed = sidecar()
    proposed["steps"][0]["supports"][0]["quote"] = "HIDDEN TAIL"
    result = validate_sft_v3_evidence_sidecar(proposed, trace=trace(), retrieved_passages=rows)
    assert not result["valid"]
    assert "HIDDEN TAIL" not in visible_passages_v3(rows)[0]


@pytest.mark.parametrize("mutate", [
    lambda s: s["steps"][0].update(derivation_from_steps=[2]),
    lambda s: s["steps"][0]["supports"][0].update(passage_index=11),
    lambda s: s["steps"][0]["supports"][0].update(passage_index=True),
    lambda s: s["steps"][0]["supports"][0].update(quote="invented evidence"),
    lambda s: s["steps"][0]["supports"].clear(),
    lambda s: s["steps"][0].update(step_index=True),
    lambda s: s["steps"].reverse(),
    lambda s: s.update(gold_answer="forbidden extra field"),
])
def test_sidecar_rejects_bad_source_or_dependency_bindings(mutate):
    proposed = sidecar()
    mutate(proposed)
    assert not validate_sft_v3_evidence_sidecar(proposed, trace=trace(), retrieved_passages=passages())["valid"]


def test_tokenization_trains_exact_teacher_and_eot_only():
    tokenizer = Tokenizer()
    record = dict(question="Q", retrieved_passages=passages(), kg_subgraph=[],
                  teacher_output=trace(), answer="wrong answer", metadata={"gold_answer": "wrong gold"})
    record["messages"] = build_sft_v3_messages(question="Q", retrieved_passages=passages(), answer_trace=trace())
    encoded = tokenize_frozen_sft_v3_record(record, tokenizer)
    n = encoded["prompt_tokens"]
    assert encoded["labels"][:n] == [-100] * n
    assert encoded["labels"][n:] == encoded["input_ids"][n:]
    assert encoded["labels"][-1] == tokenizer.vocab["<eot>"]
    assert tokenizer.vocab["Example"] in encoded["labels"][n:]
    assert "wrong" not in tokenizer.vocab
    assert encoded["passage_count"] == 10


def test_frozen_messages_must_match_source_record_exactly():
    record = dict(question="Q", retrieved_passages=passages(), kg_subgraph=[], teacher_output=trace())
    record["messages"] = build_sft_v3_messages(question="Q", retrieved_passages=passages(), answer_trace=trace())
    record["messages"][2]["content"] = trace().replace("Example", "a corrected gold")
    with pytest.raises(ValueError, match="frozen messages differ"):
        tokenize_frozen_sft_v3_record(record, Tokenizer())


def test_long_input_is_rejected_without_dropping_passages():
    original = passages()
    before = deepcopy(original)
    with pytest.raises(ValueError, match="passages retained"):
        tokenize_sft_v3_example(Tokenizer(), question="Q", retrieved_passages=original,
                               answer_trace=trace(), max_length=20)
    assert original == before


def test_assistant_budget_counts_actual_template_suffix():
    target = trace().replace("Subject 1 is identified.", "Subject 1 " + "word " * 390)
    with pytest.raises(ValueError, match="assistant token budget"):
        tokenize_sft_v3_example(Tokenizer(), question="Q", retrieved_passages=passages(), answer_trace=target)


def test_prompt_prefix_mismatch_never_unmasks_prompt():
    class BadTokenizer(Tokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            result = super().apply_chat_template(messages, tokenize=tokenize,
                                                  add_generation_prompt=add_generation_prompt)
            return ("WRONG " + result) if not add_generation_prompt else result
    with pytest.raises(ValueError, match="exact token prefix"):
        tokenize_sft_v3_example(BadTokenizer(), question="Q", retrieved_passages=passages(), answer_trace=trace())
