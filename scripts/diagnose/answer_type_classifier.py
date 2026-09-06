"""
Answer type classification for multi-hop QA.

Distinguishes between entity/span, boolean, numeric, temporal, comparison,
and derived phrase answers to apply appropriate evaluation metrics.
"""

import re
from typing import Literal, Optional
from dataclasses import dataclass


AnswerType = Literal[
    "entity_span",      # Named entities or text spans
    "boolean",          # yes/no/true/false
    "numeric",          # Numbers, counts, quantities
    "temporal",         # Dates, years, time periods
    "comparison",       # Comparative answers (more/less, first/second, etc.)
    "derived_phrase",   # Multi-word phrases derived from reasoning
    "unknown"           # Cannot confidently classify
]


@dataclass
class AnswerClassification:
    """Classification result for an answer."""
    answer_type: AnswerType
    confidence: float  # 0.0 to 1.0
    reasoning: str     # Why this classification was chosen


# Boolean answer patterns
BOOLEAN_PATTERNS = [
    r"^yes$",
    r"^no$",
    r"^true$",
    r"^false$",
]

# Numeric patterns
NUMERIC_PATTERNS = [
    r"^\d+$",                          # Pure numbers: 42
    r"^\d{1,3}(,\d{3})*$",            # With commas: 1,335,907
    r"^\d+\.\d+$",                     # Decimals: 42.195
    r"^about\s+\d+",                   # About X: about 400
    r"^approximately\s+\d+",
    r"^\d+\s*(km|miles|meters|kg|tons|pounds)",  # With units
]

# Temporal patterns
TEMPORAL_PATTERNS = [
    r"^\d{4}$",                        # Year: 1989
    r"^\d{1,2}/\d{1,2}/\d{2,4}$",     # Date: 12/25/1989
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d",
    r"^\d+\s+(years?|months?|days?|centuries?|decades?)",  # Duration: 400 years
    r"^in\s+\d{4}",                    # In 1989
    r"^\d{4}-\d{2}-\d{2}",            # ISO date
]

# Comparison patterns (in question, not answer)
COMPARISON_QUESTION_PATTERNS = [
    r"\bmore\b",
    r"\bless\b",
    r"\blarger\b",
    r"\bsmaller\b",
    r"\bfirst\b",
    r"\bsecond\b",
    r"\bearlier\b",
    r"\blater\b",
    r"\bolder\b",
    r"\byounger\b",
    r"\b-est\b",  # superlatives
]

# Spatial/relational patterns
SPATIAL_PATTERNS = [
    r"\bthrough\s+the\b",              # through the Florida Straits
    r"\bin\s+\w+",                     # in California
    r"\bat\s+\w+",
    r"\bnear\s+\w+",
]

# Ordinal patterns
ORDINAL_PATTERNS = [
    r"\bfirst\b",
    r"\bsecond\b",
    r"\bthird\b",
    r"\bthird-largest\b",
    r"\b\d+(st|nd|rd|th)\b",
]


def classify_answer_type(
    answer: str,
    question: str,
    allow_unknown: bool = True
) -> AnswerClassification:
    """Classify the type of an answer given the question context.

    Args:
        answer: The answer string to classify
        question: The question text (provides context)
        allow_unknown: If False, force classification even when uncertain

    Returns:
        AnswerClassification with type, confidence, and reasoning
    """
    answer_lower = answer.lower().strip()
    question_lower = question.lower()

    # Boolean (high confidence)
    for pattern in BOOLEAN_PATTERNS:
        if re.match(pattern, answer_lower):
            return AnswerClassification(
                answer_type="boolean",
                confidence=1.0,
                reasoning=f"Exact boolean match: '{answer}'"
            )

    # Numeric (high confidence for pure numbers)
    for pattern in NUMERIC_PATTERNS:
        if re.match(pattern, answer_lower):
            # Check if it's actually a year (could be temporal)
            if re.match(r"^\d{4}$", answer_lower):
                # Ambiguous: could be year or count
                if any(word in question_lower for word in ["when", "year", "date"]):
                    return AnswerClassification(
                        answer_type="temporal",
                        confidence=0.8,
                        reasoning="Four-digit number in temporal context"
                    )
                else:
                    return AnswerClassification(
                        answer_type="numeric",
                        confidence=0.7,
                        reasoning="Four-digit number, context unclear"
                    )

            return AnswerClassification(
                answer_type="numeric",
                confidence=0.9,
                reasoning=f"Matches numeric pattern: '{answer}'"
            )

    # Temporal (dates, durations)
    for pattern in TEMPORAL_PATTERNS:
        if re.search(pattern, answer_lower):
            return AnswerClassification(
                answer_type="temporal",
                confidence=0.9,
                reasoning=f"Matches temporal pattern: '{answer}'"
            )

    # Ordinal/Comparison
    is_ordinal_answer = any(re.search(p, answer_lower) for p in ORDINAL_PATTERNS)
    is_comparison_q = any(re.search(p, question_lower) for p in COMPARISON_QUESTION_PATTERNS)

    if is_ordinal_answer or is_comparison_q:
        return AnswerClassification(
            answer_type="comparison",
            confidence=0.8,
            reasoning="Ordinal or comparative context detected"
        )

    # Derived phrase (spatial, relational, multi-word)
    if any(re.search(p, answer_lower) for p in SPATIAL_PATTERNS):
        return AnswerClassification(
            answer_type="derived_phrase",
            confidence=0.7,
            reasoning="Spatial/relational phrase"
        )

    # Multi-word answers without clear type
    word_count = len(answer.split())
    if word_count >= 4:
        return AnswerClassification(
            answer_type="derived_phrase",
            confidence=0.6,
            reasoning=f"Multi-word phrase ({word_count} words)"
        )

    # Default: entity/span (most common in multi-hop QA)
    if word_count <= 3 and not allow_unknown:
        return AnswerClassification(
            answer_type="entity_span",
            confidence=0.5,
            reasoning="Short answer, default to entity"
        )

    # Unknown (conservative)
    if allow_unknown:
        return AnswerClassification(
            answer_type="unknown",
            confidence=0.0,
            reasoning="Could not confidently classify"
        )

    # Forced classification
    return AnswerClassification(
        answer_type="entity_span",
        confidence=0.3,
        reasoning="Forced entity classification (low confidence)"
    )


def should_check_answer_surface(answer_type: AnswerType) -> bool:
    """Determine if answer surface matching is appropriate for this type.

    Only entity/span answers should have their surface form checked in KG triples.
    For yes/no, numeric results, etc., we need to check the operands instead.
    """
    return answer_type == "entity_span"


# Example usage and test cases
if __name__ == "__main__":
    test_cases = [
        ("yes", "Were Scott Derrickson and Ed Wood of the same nationality?"),
        ("no", "Did both directors work in Hollywood?"),
        ("42.195", "What is the marathon distance in kilometers?"),
        ("1,335,907", "What is the population?"),
        ("about 400 years", "How long ago was it founded?"),
        ("third-largest", "What ranking does it have?"),
        ("through the Florida Straits", "Which route does it take?"),
        ("1989", "When was it released?"),
        ("President Richard Nixon", "Who was it named after?"),
        ("Delhi", "What city is the head office in?"),
    ]

    print("Answer Type Classification Tests")
    print("=" * 80)

    for answer, question in test_cases:
        result = classify_answer_type(answer, question)
        print(f"\nAnswer: '{answer}'")
        print(f"Question: {question[:60]}...")
        print(f"Type: {result.answer_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Check surface: {should_check_answer_surface(result.answer_type)}")
