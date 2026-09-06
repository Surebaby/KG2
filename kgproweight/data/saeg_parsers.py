"""Parser for SAEG traces with separate KG and passage citation fields.

The legacy parser remains unchanged.  SAEG's ``Knowledge Used`` accepts only
standard KG triples, while ``Passage Used`` accepts only visible ``P<n>`` IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional, Sequence, Tuple

from kgproweight.data.parsers import ParsedStep, parse_steps


PASSAGE_USED_RE = re.compile(r"(?im)^[ \t]*Passage Used\s*:\s*([^\r\n]*)$")
PASSAGE_ID_RE = re.compile(r"P[1-9]\d*")


@dataclass
class ParsedSAEGStep:
    index: int
    raw_text: str
    cited_triples: list[Tuple[str, str, str]] = field(default_factory=list)
    cited_passage_ids: list[str] = field(default_factory=list)
    unknown_passage_ids: list[str] = field(default_factory=list)
    intermediate_conclusion: Optional[str] = None
    knowledge_used_field_count: int = 0
    passage_used_field_count: int = 0
    knowledge_used_valid: bool = False
    passage_used_valid: bool = False
    citation_contract_errors: list[str] = field(default_factory=list)

    @property
    def citation_contract_valid(self) -> bool:
        return self.knowledge_used_valid and self.passage_used_valid


def _parse_passage_field(
    text: str,
    known_passage_ids: Optional[Sequence[str]],
) -> tuple[list[str], list[str], int, list[str]]:
    fields = PASSAGE_USED_RE.findall(text)
    errors: list[str] = []
    cited: list[str] = []
    unknown: list[str] = []
    if len(fields) != 1:
        errors.append(f"passage_used_field_count={len(fields)}")
    known = set(str(value) for value in known_passage_ids) if known_passage_ids is not None else None
    for raw in fields:
        value = raw.strip()
        if not (value.startswith("[") and value.endswith("]")):
            errors.append("passage_used_must_be_bracketed_list")
            continue
        inner = value[1:-1].strip()
        if not inner:
            continue
        tokens = [token.strip() for token in inner.split(",")]
        if any(not PASSAGE_ID_RE.fullmatch(token) for token in tokens):
            errors.append("malformed_passage_id")
            continue
        if len(tokens) != len(set(tokens)):
            errors.append("duplicate_passage_id")
        for token in tokens:
            if known is not None and token not in known:
                unknown.append(token)
            elif token not in cited:
                cited.append(token)
    if unknown:
        errors.append("unknown_passage_id")
    return cited, list(dict.fromkeys(unknown)), len(fields), list(dict.fromkeys(errors))


def parse_saeg_steps(
    raw_output: str,
    *,
    known_kg: Optional[Sequence[Sequence[str]]] = None,
    known_passage_ids: Optional[Sequence[str]] = None,
) -> list[ParsedSAEGStep]:
    """Parse SAEG steps while enforcing both independent citation contracts."""
    legacy_steps: list[ParsedStep] = parse_steps(raw_output, known_kg=known_kg)
    parsed: list[ParsedSAEGStep] = []
    for step in legacy_steps:
        passage_ids, unknown, field_count, passage_errors = _parse_passage_field(
            step.raw_text, known_passage_ids
        )
        errors = list(dict.fromkeys(step.citation_contract_errors + passage_errors))
        parsed.append(ParsedSAEGStep(
            index=step.index,
            raw_text=step.raw_text,
            cited_triples=step.cited_triples,
            cited_passage_ids=passage_ids,
            unknown_passage_ids=unknown,
            intermediate_conclusion=step.intermediate_conclusion,
            knowledge_used_field_count=step.knowledge_used_field_count,
            passage_used_field_count=field_count,
            knowledge_used_valid=step.knowledge_used_valid,
            passage_used_valid=not passage_errors,
            citation_contract_errors=errors,
        ))
    return parsed
