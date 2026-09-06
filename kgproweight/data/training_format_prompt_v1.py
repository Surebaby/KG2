"""Opt-in training prompt clarification; canonical builders stay unchanged.

This frozen variant clarifies the existing format-v2 output contract and adds
explicit anti-padding and relevant-citation guidance as one prompt variable. The
caller must obtain required_steps from the unchanged production validator
using the original source hard gate, before any source-credit mask. No Gold,
derived answer, score or candidate generation is an input to this module.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from kgproweight.data.prompts import SFT_SYSTEM_PROMPT, build_sft_messages


TRAINING_FORMAT_PROMPT_VERSION = "training-format-prompt-v1"
_OLD_STOP = "- Stop generating after [Final Answer]."
_NEW_STOP = "- Stop only after writing the concise answer on the line following [Final Answer]."
_RULES = """
Output format for this question:
- Write between {required_steps} and 5 steps, inclusive. Start with [Step 1]
  and number every step consecutively, without skipping or repeating a number.
- In EVERY step, include these three fields exactly once: Reasoning:,
  Knowledge Used:, and Conclusion:. Put each field on a separate line,
  with its content after its heading. Give a brief, nonempty factual Conclusion.
- Write the Reasoning content as a complete, concise sentence of at least
  20 characters. Each step should add a distinct useful fact or inference;
  do not repeat or split the same claim merely to reach the minimum step count.
- Keep Knowledge Used on a single line. If no supplied Knowledge Graph
  triple is relevant to that step, write Knowledge Used: []. Do not cite
  an unrelated triple merely because the graph is nonempty.
- After the last step, write exactly one [Final Answer] heading. On the next
  line, write the brief answer itself. Do not stop at the heading; finish the
  answer before ending the response.
"""


def clarify_training_format_messages_v1(
    messages: Sequence[Mapping[str, str]], *, required_steps: int,
) -> list[dict[str, str]]:
    """Copy a canonical two-message prompt and change only system content.

    Requiring the original system string prevents silently applying this
    variant twice, to a different prompt family, or to an already edited
    prompt. The unchanged user message carries the question and evidence.
    """
    if type(required_steps) is not int or required_steps not in (2, 3):
        raise ValueError("required_steps must be the existing validator's integer 2 or 3")
    if len(messages) != 2 or [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("training-format-v1 requires the canonical system/user prompt")
    if messages[0].get("content") != SFT_SYSTEM_PROMPT or SFT_SYSTEM_PROMPT.count(_OLD_STOP) != 1:
        raise ValueError("training-format-v1 requires the unchanged canonical SFT system prompt")
    if not isinstance(messages[1].get("content"), str):
        raise ValueError("canonical user content must be a string")
    result = deepcopy(list(messages))
    result[0]["content"] = SFT_SYSTEM_PROMPT.replace(_OLD_STOP, _NEW_STOP) + _RULES.format(required_steps=required_steps)
    assert result[1] == messages[1]
    return result


def build_training_format_messages_v1(
    *, question: str, retrieved_passages: Sequence[Mapping | str],
    kg_triples: Sequence[Sequence[str]], required_steps: int,
    top_k: int = 10, max_kg_triples: int = 12,
) -> list[dict[str, str]]:
    """Build through the canonical factory, then opt into this one variant."""
    messages = build_sft_messages(question=question, retrieved_passages=retrieved_passages,
                                  kg_triples=kg_triples, top_k=top_k, max_kg_triples=max_kg_triples)
    return clarify_training_format_messages_v1(messages, required_steps=required_steps)
