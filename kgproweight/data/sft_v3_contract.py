"""Opt-in data contract for the new, from-base mixed SFT experiment.

This module does not change legacy prompts, evaluation, PPO rewards or the
legacy silver loader.  It trains the supplied teacher trace verbatim: Gold is
not a constructor argument, and there is no answer replacement, passage
dropping or token truncation.  Mechanical checks below are not an entailment
verifier; independently generated semantic reviews remain a separate gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from kgproweight.data.parsers import ParsedStep
from kgproweight.data.prompts import build_sft_messages, format_retrieved_block


CONTRACT_VERSION = "mixed-sft-v3-contract-v1"
EVIDENCE_SCHEMA_VERSION = "sft-v3-evidence-v1"
PASSAGE_COUNT = 10
MAX_KG_TRIPLES = 12
MIN_STEPS = 2
MAX_STEPS = 5
MAX_ASSISTANT_TOKENS = 384
MIN_REASONING_CHARS = 20

SFT_V3_SYSTEM_PROMPT = """You are a careful multi-hop reasoning assistant.
Answer using only the supplied evidence. Write 2 to 5 useful steps, numbered
consecutively from 1. Use the fewest steps that complete the reasoning; do not
repeat facts or invent intermediate claims to make a longer trace.

Use this exact format, with every field once on its own line:
[Step 1]
Reasoning: <one concise factual sentence of at least 20 characters>
Knowledge Used: [(head, relation, tail), ...]
Conclusion: <one short factual conclusion>

[Step 2]
Reasoning: <the next supported fact or inference>
Knowledge Used: []
Conclusion: <one short factual conclusion>

[Final Answer]
<a concise answer on one line>

Copy each cited Knowledge Used triple verbatim from the supplied Knowledge
Graph. Use [] when no supplied triple supports that step. Passages are evidence,
not KG triples: never turn passage text into an invented Knowledge Used triple.
Each step must add a supported fact or a valid inference from earlier steps.
Keep the whole response concise, and stop immediately after the answer line.
"""

_STEP = re.compile(r"(?m)^\[Step ([1-9][0-9]*)\]$")
_MARKER_START = re.compile(r"\[\s*(?:step\b|final\s+answer\b)", re.I)
_BODY = re.compile(
    r"Reasoning: (?P<reasoning>[^\r\n]+)\n"
    r"Knowledge Used: (?P<knowledge>[^\r\n]+)\n"
    r"Conclusion: (?P<conclusion>[^\r\n]+)"
)
_FIELD = re.compile(r"(?:Reasoning|Knowledge Used|Conclusion)\s*:", re.I)
_CHAT_SPECIAL = re.compile(r"<\|[^\r\n<>]*\|>")
_FORBIDDEN_INPUT_KEYS = frozenset({
    "answer", "answers", "gold_answer", "gold_answers", "gold_answer_aliases",
    "golden_answers", "supporting_facts", "decomposition", "teacher_output",
    "target", "labels", "label", "is_supporting", "supporting",
})


def _reject_annotation_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_INPUT_KEYS:
                raise ValueError(f"annotation key is forbidden in model evidence: {key}")
            _reject_annotation_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_annotation_keys(item)


def _clean_kg(kg_triples: Sequence[Sequence[str]]) -> list[tuple[str, str, str]]:
    if isinstance(kg_triples, (str, bytes)) or len(kg_triples) > MAX_KG_TRIPLES:
        raise ValueError("KG must contain at most 12 triples; no silent truncation")
    result = []
    for triple in kg_triples:
        if (not isinstance(triple, (list, tuple)) or len(triple) != 3
                or any(not isinstance(x, str) or not x.strip() or "\n" in x or "\r" in x
                       for x in triple)):
            raise ValueError("KG triples must contain three nonempty single-line strings")
        result.append(tuple(x.strip() for x in triple))
    if len(set(result)) != len(result):
        raise ValueError("duplicate KG triples are not allowed")
    return result


def visible_passages_v3(
    retrieved_passages: Sequence[Mapping[str, Any] | str],
) -> list[str]:
    """Return exactly the texts visible under the unchanged passage formatter.

    Its existing 1200-character cap is explicit here.  Quotes in hidden raw
    tails cannot pass the sidecar check.  Passage identities stay outside the
    prompt and are the caller's provenance responsibility.
    """
    if isinstance(retrieved_passages, (str, bytes)) or len(retrieved_passages) != PASSAGE_COUNT:
        raise ValueError("SFT v3 requires exactly 10 passages")
    _reject_annotation_keys(retrieved_passages)
    result = []
    for passage in retrieved_passages:
        if isinstance(passage, str):
            content = passage
        elif isinstance(passage, Mapping):
            content = passage.get("contents") or passage.get("text") or ""
        else:
            raise ValueError("passages must be strings or mappings")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("each of the 10 passages must have nonempty text")
        if _CHAT_SPECIAL.search(content):
            raise ValueError("literal chat special tokens are forbidden in passages")
        text = content.strip()
        if len(text) > 1200:
            text = text[:1200] + " …"
        result.append(text)
    # Bind to the actual canonical formatter instead of trusting a second
    # implementation of its clipping/rank behavior.
    expected = "\n".join(f"[{i}] {text}" for i, text in enumerate(result, 1))
    if format_retrieved_block(retrieved_passages, max_passages=PASSAGE_COUNT) != expected:
        raise ValueError("canonical passage renderer changed; contract must be versioned")
    return result


def build_sft_v3_messages(
    *, question: str, retrieved_passages: Sequence[Mapping[str, Any] | str],
    kg_triples: Sequence[Sequence[str]] = (), answer_trace: str | None = None,
) -> list[dict[str, str]]:
    """Build the opt-in prompt without copying arbitrary row metadata."""
    if not isinstance(question, str) or not question.strip() or _CHAT_SPECIAL.search(question):
        raise ValueError("question must be nonempty text without chat special tokens")
    visible_passages_v3(retrieved_passages)
    kg = _clean_kg(kg_triples)
    if any(_CHAT_SPECIAL.search(value) for triple in kg for value in triple):
        raise ValueError("literal chat special tokens are forbidden in KG")
    if answer_trace is not None and not isinstance(answer_trace, str):
        raise TypeError("answer_trace must be a string")
    messages = build_sft_messages(
        question=question, retrieved_passages=retrieved_passages, kg_triples=kg,
        top_k=PASSAGE_COUNT, max_kg_triples=MAX_KG_TRIPLES, answer_trace=answer_trace,
    )
    messages[0]["content"] = SFT_V3_SYSTEM_PROMPT
    return messages


def validate_sft_v3_trace(
    trace: str, *, known_kg: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Strict target syntax/duplicate/citation checks, not factual validation.

    No normalization, repair or Gold answer substitution is performed.  Steps
    and Final returned here are exact substrings of the accepted target.
    """
    if not isinstance(trace, str):
        raise TypeError("trace must be a string")
    kg = _clean_kg(known_kg)
    errors: list[str] = []
    if "\r" in trace:
        errors.append("noncanonical_line_endings")
    if _CHAT_SPECIAL.search(trace):
        errors.append("literal_chat_special_token")
    if trace.count("[Final Answer]") != 1:
        errors.append("final_marker_count_not_one")
    prefix, marker, suffix = trace.partition("[Final Answer]")
    final = ""
    if marker:
        if prefix and not prefix.endswith("\n"):
            errors.append("final_heading_not_on_own_line")
        if not suffix.startswith("\n"):
            errors.append("answer_not_on_next_line")
        final = suffix.strip()
        if not final or "\n" in final or not any(c.isalnum() for c in final):
            errors.append("final_not_one_nonempty_answer_line")
        if _FIELD.search(final) or _MARKER_START.search(final):
            errors.append("schema_content_after_final")
    headers = list(_STEP.finditer(prefix))
    if len(_MARKER_START.findall(trace)) != len(headers) + int(bool(marker)):
        errors.append("noncanonical_or_extra_marker")
    indices = [int(match.group(1)) for match in headers]
    if not MIN_STEPS <= len(headers) <= MAX_STEPS:
        errors.append("step_count_outside_2_5")
    if indices != list(range(1, len(headers) + 1)):
        errors.append("step_indices_not_contiguous")
    if not headers or prefix[:headers[0].start()].strip():
        errors.append("preamble_or_missing_step")
    parsed_steps = []
    reasonings: list[str] = []
    for position, match in enumerate(headers):
        end = headers[position + 1].start() if position + 1 < len(headers) else len(prefix)
        raw_body = prefix[match.end():end].strip()
        fields = _BODY.fullmatch(raw_body)
        step_index = int(match.group(1))
        if not fields or len(_FIELD.findall(raw_body)) != 3:
            errors.append(f"step_{step_index}:field_order_or_count_or_line_contract")
            continue
        reasoning = fields.group("reasoning").strip()
        conclusion = fields.group("conclusion").strip()
        if len(reasoning) < MIN_REASONING_CHARS or not any(c.isalpha() for c in reasoning):
            errors.append(f"step_{step_index}:reasoning_empty_or_short")
        if not any(c.isalnum() for c in conclusion):
            errors.append(f"step_{step_index}:conclusion_empty")
        normalized_reasoning = " ".join(reasoning.casefold().split())
        reasonings.append(normalized_reasoning)
        parsed = ParsedStep.from_text(step_index, raw_body, known_kg=kg)
        if (not parsed.knowledge_used_valid or parsed.unknown_citation_surfaces
                or parsed.knowledge_used_malformed_content):
            errors.append(f"step_{step_index}:unknown_or_malformed_kg_citation")
        # ParsedStep accepts punctuation residue such as '[,]'; supervised
        # targets must obey the exact rendered list grammar as well.
        knowledge = fields.group("knowledge").strip()
        if knowledge != "[]":
            surfaces = [f"({h}, {r}, {t})" for h, r, t in kg]
            pieces = "|".join(re.escape(surface) for surface in surfaces)
            exact_list = rf"\[(?:{pieces})(?:, (?:{pieces}))*\]" if pieces else r"(?!)"
            if re.fullmatch(exact_list, knowledge) is None:
                errors.append(f"step_{step_index}:noncanonical_kg_list")
        parsed_steps.append({
            "index": step_index, "text": raw_body, "reasoning": reasoning,
            "conclusion": conclusion, "cited_triples": [list(t) for t in parsed.cited_triples],
        })
    if len(set(reasonings)) != len(reasonings):
        errors.append("duplicate_normalized_reasoning")
    return {
        "contract_version": CONTRACT_VERSION, "valid": not errors,
        "violations": list(dict.fromkeys(errors)), "steps": parsed_steps,
        "step_count": len(headers), "final_answer": final,
        "semantic_grounding_verified": False,
    }


def validate_sft_v3_evidence_sidecar(
    sidecar: Mapping[str, Any], *, trace: str,
    retrieved_passages: Sequence[Mapping[str, Any] | str],
    known_kg: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Check external evidence locations and DAG references mechanically.

    Every step needs a visible quote, an exact supplied KG citation, or a
    dependency on earlier supported steps.  The quote need not entail the
    claim: independent semantic review must decide that before admission.
    """
    visible = visible_passages_v3(retrieved_passages)
    checked = validate_sft_v3_trace(trace, known_kg=known_kg)
    errors = [] if checked["valid"] else ["trace_contract_failed"]
    if not isinstance(sidecar, Mapping) or set(sidecar) != {"schema_version", "steps"}:
        return {"valid": False, "violations": ["sidecar_schema"], "quote_count": 0,
                "semantic_grounding_verified": False}
    if sidecar.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        errors.append("sidecar_schema_version")
    rows = sidecar.get("steps")
    if not isinstance(rows, list):
        rows = []
        errors.append("sidecar_steps_not_list")
    expected_indices = [step["index"] for step in checked["steps"]]
    indices = [row.get("step_index") if isinstance(row, Mapping) else None for row in rows]
    if any(type(i) is not int for i in indices) or indices != expected_indices:
        errors.append("sidecar_step_indices_mismatch")
    by_index = {step["index"]: step for step in checked["steps"]}
    quote_count = 0
    for row in rows:
        if (not isinstance(row, Mapping)
                or set(row) != {"step_index", "supports", "derivation_from_steps"}):
            errors.append("sidecar_step_schema")
            continue
        index = row["step_index"]
        if type(index) is not int or index not in by_index:
            errors.append("sidecar_unknown_step")
            continue
        supports, dependencies = row["supports"], row["derivation_from_steps"]
        if not isinstance(supports, list) or not isinstance(dependencies, list):
            errors.append(f"step_{index}:supports_or_dependencies_not_list")
            continue
        if (any(type(i) is not int or i < 1 or i >= index for i in dependencies)
                or len(set(str(i) for i in dependencies)) != len(dependencies)):
            errors.append(f"step_{index}:invalid_forward_or_duplicate_dependency")
        if not supports and not dependencies and not by_index[index]["cited_triples"]:
            errors.append(f"step_{index}:no_evidence_or_prior_derivation")
        seen_quotes = set()
        for item in supports:
            if not isinstance(item, Mapping) or set(item) != {"passage_index", "quote"}:
                errors.append(f"step_{index}:support_schema")
                continue
            passage_index, quote = item["passage_index"], item["quote"]
            if type(passage_index) is not int or not 1 <= passage_index <= PASSAGE_COUNT:
                errors.append(f"step_{index}:unknown_passage_index")
                continue
            if (not isinstance(quote, str) or not quote.strip()
                    or not any(c.isalnum() for c in quote)
                    or quote not in visible[passage_index - 1]):
                errors.append(f"step_{index}:quote_not_in_visible_passage")
                continue
            key = (passage_index, quote)
            if key in seen_quotes:
                errors.append(f"step_{index}:duplicate_quote")
            seen_quotes.add(key)
            quote_count += 1
    return {
        "valid": not errors, "violations": list(dict.fromkeys(errors)),
        "quote_count": quote_count, "semantic_grounding_verified": False,
        "limitation": "Exact quotes and prior-step references do not establish entailment or answerability.",
    }


def tokenize_sft_v3_example(
    tokenizer: Any, *, question: str,
    retrieved_passages: Sequence[Mapping[str, Any] | str],
    answer_trace: str, kg_triples: Sequence[Sequence[str]] = (),
    max_length: int = 6144, max_assistant_tokens: int = MAX_ASSISTANT_TOKENS,
) -> dict[str, Any]:
    """Tokenize once with assistant-only labels, rejecting any budget overflow.

    The assistant suffix includes the chat-template end-of-turn token.  Prompt
    prefix disagreement is an error, never a fallback to training prompt text.
    """
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if type(max_assistant_tokens) is not int or not 1 <= max_assistant_tokens <= MAX_ASSISTANT_TOKENS:
        raise ValueError("max_assistant_tokens must be a positive integer at most 384")
    checked = validate_sft_v3_trace(answer_trace, known_kg=kg_triples)
    if not checked["valid"]:
        raise ValueError("invalid SFT v3 target: " + ";".join(checked["violations"]))
    messages = build_sft_v3_messages(
        question=question, retrieved_passages=retrieved_passages,
        kg_triples=kg_triples, answer_trace=answer_trace,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not an exact token prefix")
    assistant_tokens = len(full_ids) - len(prompt_ids)
    if not 0 < assistant_tokens <= max_assistant_tokens:
        raise ValueError(f"assistant token budget exceeded: {assistant_tokens}/{max_assistant_tokens}")
    if len(full_ids) > max_length:
        raise ValueError(f"full sequence token budget exceeded: {len(full_ids)}/{max_length}; passages retained")
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    return {
        "input_ids": list(full_ids), "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "prompt_tokens": len(prompt_ids), "assistant_tokens": assistant_tokens,
        "full_tokens": len(full_ids), "passage_count": PASSAGE_COUNT,
        "contract_version": CONTRACT_VERSION,
    }


def tokenize_frozen_sft_v3_record(
    record: Mapping[str, Any], tokenizer: Any, *, max_length: int = 6144,
    max_assistant_tokens: int = MAX_ASSISTANT_TOKENS,
) -> dict[str, Any]:
    """Load an explicitly frozen chat record without trusting its messages.

    ``teacher_output`` is authoritative for the exact assistant text.  The
    ``answer`` and ``metadata`` fields, if present, are never consulted.  The
    caller must verify release/acceptance/identity manifests before loading a
    corpus; this per-record check alone does not certify data admission.
    """
    required = {"question", "retrieved_passages", "kg_subgraph", "teacher_output", "messages"}
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise ValueError("frozen SFT v3 record is missing required input/trace/messages fields")
    expected = build_sft_v3_messages(
        question=record["question"], retrieved_passages=record["retrieved_passages"],
        kg_triples=record["kg_subgraph"], answer_trace=record["teacher_output"],
    )
    if record["messages"] != expected:
        raise ValueError("frozen messages differ from the declared question/evidence/teacher trace")
    return tokenize_sft_v3_example(
        tokenizer, question=record["question"],
        retrieved_passages=record["retrieved_passages"], kg_triples=record["kg_subgraph"],
        answer_trace=record["teacher_output"], max_length=max_length,
        max_assistant_tokens=max_assistant_tokens,
    )
