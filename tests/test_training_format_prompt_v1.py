from copy import deepcopy

import pytest

from kgproweight.data import prompts
from kgproweight.data.training_format_prompt_v1 import (
    TRAINING_FORMAT_PROMPT_VERSION, build_training_format_messages_v1,
    clarify_training_format_messages_v1,
)


def material():
    return {"question": "  Which city is connected?  ",
            "retrieved_passages": [{"contents": "Passage\nwith (commas, and punctuation)."}, "中文 evidence"],
            "kg_triples": [("City, A", "located in", "Place B")], "top_k": 10, "max_kg_triples": 12}


@pytest.mark.parametrize("minimum", [2, 3])
def test_only_system_changes_and_inputs_are_untouched(minimum):
    data = material()
    before = deepcopy(data)
    original = prompts.build_sft_messages(**data)
    original_copy = deepcopy(original)
    new = clarify_training_format_messages_v1(original, required_steps=minimum)
    assert TRAINING_FORMAT_PROMPT_VERSION == "training-format-prompt-v1"
    assert original == original_copy and data == before
    assert new[1]["content"].encode() == original[1]["content"].encode()
    assert new[0]["content"] != original[0]["content"]
    assert f"between {minimum} and 5 steps" in new[0]["content"]
    assert "Stop generating after [Final Answer]." not in new[0]["content"]
    assert "answer before ending the response" in new[0]["content"]
    assert "20 characters" in new[0]["content"]
    assert build_training_format_messages_v1(**data, required_steps=minimum) == new
    new[1]["content"] = "changed copy"
    assert original == original_copy


@pytest.mark.parametrize("minimum", [None, True, False, 1, 4, 5, "2", 2.0])
def test_invalid_or_guessed_minimum_rejected(minimum):
    with pytest.raises(ValueError, match="required_steps"):
        build_training_format_messages_v1(**material(), required_steps=minimum)


def test_no_default_sft_rl_or_inference_change():
    data = material()
    before = {name: getattr(prompts, name) for name in ("SFT_SYSTEM_PROMPT", "RL_SYSTEM_PROMPT", "TEACHER_SYSTEM_PROMPT")}
    canonical = prompts.build_sft_messages(**data)
    build_training_format_messages_v1(**data, required_steps=3)
    assert {name: getattr(prompts, name) for name in before} == before
    assert prompts.build_sft_messages(**data) == canonical
    kwargs = {k: v for k, v in data.items() if k != "max_kg_triples"}
    assert prompts.build_rl_messages(**kwargs) == canonical
    assert prompts.build_inference_messages(**kwargs) == canonical


def test_other_prompt_family_and_double_application_rejected():
    messages = prompts.build_sft_messages(**material())
    modified = clarify_training_format_messages_v1(messages, required_steps=3)
    with pytest.raises(ValueError, match="unchanged canonical"):
        clarify_training_format_messages_v1(modified, required_steps=3)
    messages[0]["content"] = prompts.TEACHER_SYSTEM_PROMPT
    with pytest.raises(ValueError, match="unchanged canonical"):
        clarify_training_format_messages_v1(messages, required_steps=3)


def test_assistant_targets_are_not_an_input_to_prompt_clarification():
    messages = prompts.build_sft_messages(**material(), answer_trace="target deliberately not consumed")
    with pytest.raises(ValueError, match="system/user"):
        clarify_training_format_messages_v1(messages, required_steps=3)


def test_rendering_options_are_forwarded_to_original_factory():
    data = material()
    data.update(top_k=1, max_kg_triples=0)
    original = prompts.build_sft_messages(**data)
    new = build_training_format_messages_v1(**data, required_steps=2)
    assert new[1] == original[1]
