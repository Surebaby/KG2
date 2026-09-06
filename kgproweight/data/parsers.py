"""Robust parsers for the unified ``[Step N] ... [Final Answer]`` schema.

Used by:
  • the PRM annotator (Phase 1),
  • the RL reward function (Phase 3b),
  • the inference pipeline (eval),
  • the LLM-as-Judge for IHR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Compiled patterns (exported because other modules reuse them)
# ---------------------------------------------------------------------------

DISCOURSE_RE = re.compile(
    "|".join(
        [
            r"^(let'?s|now|first|next|then|finally|therefore|so|thus|in summary|in conclusion)",
            r"^(based on|according to|given that|we can|we know|it follows)",
            r"^(to (answer|solve|determine|find))",
            r"^(the question (asks|is|requires))",
            r"^(i need to|let me|we need to)",
            r"step \d+[:.]\s*(let|now|first|we)",
        ]
    ),
    re.IGNORECASE,
)

TRIPLE_RE = re.compile(
    r"\(([^,()]+),\s*([^,()]+),\s*([^,()]+)\)"
    r"|([A-Z][^–\-\n]+?)\s*[-–]+[>→]\s*([^\n(]+)\s*[-–]+[>→]\s*([A-Z][^\n(]+)"
)

# Citations are a schema field, not arbitrary parenthesised prose.  The old
# parser applied TRIPLE_RE to the whole step and interpreted text such as
# ``(e.g., a composer, a language)`` as a KG citation.  Keep TRIPLE_RE exported
# for compatibility, but scope actual citation extraction to this one-line
# field.
KNOWLEDGE_USED_RE = re.compile(
    r"(?im)^[ \t]*Knowledge Used\s*:\s*([^\r\n]*)$"
)
# A raw parenthesised item is sufficient for telemetry to say that the model
# attempted a citation.  We deliberately do not coerce an unknown item into a
# three-tuple: values may contain commas, so such a split is ambiguous without
# the prompt KG.  Reward continues to consume only ``cited_triples`` matched
# against ``known_kg``.
RAW_CITATION_SURFACE_RE = re.compile(r"\([^()\r\n]+\)")

ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedStep:
    index: int
    raw_text: str
    cited_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    mentioned_entities: List[str] = field(default_factory=list)
    intermediate_conclusion: Optional[str] = None
    knowledge_used_field_count: int = 0
    knowledge_used_valid: bool = False
    citation_contract_errors: List[str] = field(default_factory=list)
    # Telemetry-only raw contract observations.  These never enter r_kg, alpha,
    # trajectory validity, or the Phase-1 accept/retry decision.
    unknown_citation_surfaces: List[str] = field(default_factory=list)
    knowledge_used_malformed_content: bool = False

    @classmethod
    def from_text(
        cls,
        index: int,
        text: str,
        known_kg: Optional[Sequence[Sequence[str]]] = None,
    ) -> "ParsedStep":
        fields = KNOWLEDGE_USED_RE.findall(text)
        cited: List[Tuple[str, str, str]] = []
        unknown_surfaces: List[str] = []
        malformed_content = False
        errors: List[str] = []
        if len(fields) != 1:
            errors.append(f"knowledge_used_field_count={len(fields)}")

        for value in fields:
            value = value.strip()
            if value == "[]":
                continue

            residual = value
            if known_kg is not None:
                # Match the exact strings rendered in the prompt instead of
                # splitting on commas: KG values legitimately contain commas
                # (e.g. ``Washington, D.C.`` and organization names).
                rendered = []
                for triple in known_kg:
                    if len(triple) != 3:
                        continue
                    h, r, t = (str(x).strip() for x in triple)
                    rendered.append((f"({h}, {r}, {t})", (h, r, t)))
                for surface, triple in sorted(rendered, key=lambda item: len(item[0]), reverse=True):
                    while surface in residual:
                        cited.append(triple)
                        residual = residual.replace(surface, "", 1)
                # Preserve raw unknown citation attempts for diagnostics before
                # the legacy contract check collapses unknown and malformed into
                # one backwards-compatible error code.
                found_unknown = RAW_CITATION_SURFACE_RE.findall(residual)
                unknown_surfaces.extend(s.strip() for s in found_unknown)
                residual_without_unknown = RAW_CITATION_SURFACE_RE.sub("", residual)
                if re.sub(r"[\[\],\s]", "", residual_without_unknown):
                    malformed_content = True
            else:
                # Generic parsing remains available when the caller has no KG,
                # but still only inside the schema field.  KG-aware Phase 1,
                # PPO reward and evaluation paths pass ``known_kg``.
                spans = []
                for match in TRIPLE_RE.finditer(value):
                    if match.group(1):
                        h, r, t = (
                            match.group(1).strip(),
                            match.group(2).strip(),
                            match.group(3).strip(),
                        )
                    else:
                        h, r, t = (
                            match.group(4).strip(),
                            match.group(5).strip(),
                            match.group(6).strip(),
                        )
                    if h and r and t:
                        cited.append((h, r, t))
                        spans.append(match.span())
                chars = list(value)
                for start, end in spans:
                    chars[start:end] = " " * (end - start)
                residual = "".join(chars)
                if re.sub(r"[\[\],\s]", "", residual):
                    malformed_content = True

            # After removing known/extracted triples, a valid field contains
            # only the surrounding list syntax and separators.
            if re.sub(r"[\[\],\s]", "", residual):
                errors.append("unknown_or_malformed_knowledge_used_content")
            if not (value.startswith("[") and value.endswith("]")):
                errors.append("knowledge_used_must_be_bracketed_list")

        entities = list(dict.fromkeys(e for e in ENTITY_RE.findall(text) if len(e) > 2))

        conclusion: Optional[str] = None
        m = re.search(r"(?:Conclusion|Therefore|Thus|So)[:\s]+(.+?)(?:\.|$)", text, re.IGNORECASE)
        if m:
            conclusion = m.group(1).strip()
        elif text.strip():
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            conclusion = sentences[-1] if sentences else None

        return cls(
            index=index,
            raw_text=text,
            cited_triples=cited,
            mentioned_entities=entities,
            intermediate_conclusion=conclusion,
            knowledge_used_field_count=len(fields),
            knowledge_used_valid=not errors,
            citation_contract_errors=list(dict.fromkeys(errors)),
            unknown_citation_surfaces=unknown_surfaces,
            knowledge_used_malformed_content=malformed_content,
        )


def parsed_step_from_silver_dict(step: Dict[str, Any], fallback_index: int = 0) -> ParsedStep:
    """Build a :class:`ParsedStep` from a silver-data ``steps[]`` object."""
    idx = int(step.get("index", fallback_index))
    text = (step.get("text") or "").strip()
    parsed = ParsedStep.from_text(idx, text)

    merged: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    raw_cited = step.get("cited_triples")
    if isinstance(raw_cited, list):
        for t in raw_cited:
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                continue
            h, r, t2 = (str(x).strip() for x in t)
            if not (h and r and t2):
                continue
            blob = " ".join((h, r, t2)).lower()
            if blob in {"none none none", "[none] [none] [none]"}:
                continue
            if h.lower() in {"none", "[none]"} or "no kg support" in blob:
                continue
            trip = (h, r, t2)
            if trip not in seen:
                seen.add(trip)
                merged.append(trip)

    cited = merged if merged else parsed.cited_triples
    return ParsedStep(
        index=idx,
        raw_text=text,
        cited_triples=cited,
        mentioned_entities=parsed.mentioned_entities,
        intermediate_conclusion=parsed.intermediate_conclusion,
        knowledge_used_field_count=parsed.knowledge_used_field_count,
        knowledge_used_valid=parsed.knowledge_used_valid,
        citation_contract_errors=parsed.citation_contract_errors,
        unknown_citation_surfaces=parsed.unknown_citation_surfaces,
        knowledge_used_malformed_content=parsed.knowledge_used_malformed_content,
    )


# ---------------------------------------------------------------------------
# Block splitters
# ---------------------------------------------------------------------------

def _normalize_step_markers(raw: str) -> str:
    """Map common Step heading variants to ``[Step N]``."""
    t = raw
    t = re.sub(r"(?m)^\s*#{1,3}\s*Step\s+(\d+)\s*$", r"\n[Step \1]\n", t, flags=re.IGNORECASE)
    t = re.sub(r"(?m)^\s*Step\s+(\d+)\s*[:.)]\s*", r"\n[Step \1]\n", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\s*step\s+(\d+)\s*\]", lambda m: f"[Step {m.group(1)}]", t, flags=re.IGNORECASE)
    return t


def parse_steps(
    raw_output: str,
    known_kg: Optional[Sequence[Sequence[str]]] = None,
) -> List[ParsedStep]:
    """Parse a Teacher / Student trace into ``ParsedStep`` objects."""
    normalised = _normalize_step_markers(raw_output)
    blocks = re.split(r"\[Step\s+(\d+)\]", normalised)
    # blocks[0] = preamble, then alternating (number, content) pairs
    steps: List[ParsedStep] = []
    i = 1
    while i + 1 < len(blocks):
        try:
            step_index = int(blocks[i])
        except ValueError:
            i += 2
            continue
        body = blocks[i + 1].strip()
        body = re.split(r"\[Final Answer\]|Final Answer\s*[:：]", body, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if body:
            steps.append(ParsedStep.from_text(step_index, body, known_kg=known_kg))
        i += 2
    return steps


# Backwards-compat alias used by legacy modules.
parse_teacher_output = parse_steps


def extract_final_answer(raw_output: str) -> Optional[str]:
    """Extract the final answer regardless of which spelling appears."""
    patterns = [
        r"\[Final Answer\]\s*(.+?)(?:\[|$)",
        r"(?is)\*{0,3}\s*Final Answer\*{0,3}\s*[:：]?\s*(.+?)(?:\n\n\[|\n\[Step|\Z)",
        r"(?im)^\s*Final Answer\s*[:：]\s*(.+?)(?:\n\n|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, raw_output, re.DOTALL)
        if m:
            ans = m.group(1).strip()
            if ans:
                return ans
    return None


def extract_step_token_spans(
    raw_output: str,
    tokenizer,
    add_special_tokens: bool = False,
) -> List[Tuple[int, int]]:
    """Return list of ``(start_token, end_token)`` spans, one per ``[Step N]`` block.

    Used by the PPO trainer to align per-token reward to step boundaries.
    """
    if not raw_output:
        return []
    full_ids = tokenizer(raw_output, add_special_tokens=add_special_tokens)["input_ids"]
    full_text = tokenizer.decode(full_ids)

    # Find character offsets of each step header in the decoded full text.
    spans: List[Tuple[int, int]] = []
    matches = list(re.finditer(r"\[Step\s+\d+\]", full_text))
    if not matches:
        return [(0, len(full_ids))]
    bounds = [m.start() for m in matches] + [len(full_text)]
    # token-level: we walk through input_ids character lengths using offsets.
    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    offsets = enc.get("offset_mapping") or []
    if not offsets:
        # Fallback: approximate by even split.
        n = len(full_ids)
        chunk = max(1, n // len(matches))
        for i in range(len(matches)):
            start = i * chunk
            end = (i + 1) * chunk if i < len(matches) - 1 else n
            spans.append((start, end))
        return spans
    for i in range(len(matches)):
        char_start, char_end = bounds[i], bounds[i + 1]
        tok_start = next((k for k, (a, _) in enumerate(offsets) if a >= char_start), 0)
        tok_end = next((k for k, (_, b) in enumerate(offsets) if b >= char_end), len(offsets))
        spans.append((tok_start, max(tok_end, tok_start + 1)))
    return spans
