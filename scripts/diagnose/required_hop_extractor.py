"""
Required-hop extractor for HotpotQA supporting facts.

Extracts the minimal set of entities, relations, and values needed
to answer a question, based on supporting facts annotations.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RequiredHop:
    """A single hop in the reasoning chain."""
    # Source entity mention (from question or previous hop)
    source_mention: str
    # Relation type needed (extracted from supporting text)
    relation: str
    # Target entity/value (answer or intermediate)
    target_mention: str
    # Supporting sentence where this hop is grounded
    support_title: str
    support_sent_id: int
    support_text: str
    # Confidence in extraction
    confidence: float


@dataclass
class RequiredHopChain:
    """Complete chain of hops needed to answer a question."""
    question_id: str
    question: str
    answer: str
    answer_type: str

    # Extracted hops
    hops: List[RequiredHop]

    # Anchor entities (mentioned in question)
    anchor_mentions: List[str]

    # Extraction metadata
    extraction_method: str  # "rule_based_v1", etc.
    extraction_confidence: float
    can_extract: bool  # False if had to abstain
    abstain_reason: Optional[str] = None


def extract_entities_from_text(text: str) -> List[str]:
    """Simple entity extraction from text.

    This is a placeholder - in production, use a proper NER model.
    For now, we'll use capitalized phrases as a heuristic.
    """
    import re

    # Find capitalized phrases (basic heuristic)
    # Match: One or more capitalized words
    pattern = r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b'
    entities = re.findall(pattern, text)

    # Remove common false positives
    stopwords = {'The', 'A', 'An', 'In', 'On', 'At', 'Of', 'For', 'To', 'From'}
    entities = [e for e in entities if e not in stopwords]

    return entities


def extract_required_hops_hotpotqa(
    question_data: Dict[str, Any],
    answer_type: str,
    method: str = "rule_based_v1"
) -> RequiredHopChain:
    """Extract required hops from HotpotQA supporting facts.

    Args:
        question_data: Question dict with metadata.supporting_facts
        answer_type: Answer type from classifier
        method: Extraction method identifier

    Returns:
        RequiredHopChain with extracted hops
    """
    qid = question_data["id"]
    question = question_data["question"]
    answer = question_data["golden_answers"][0] if question_data["golden_answers"] else ""
    metadata = question_data.get("metadata", {})

    # Check if we have supporting facts
    supporting_facts = metadata.get("supporting_facts", {})
    if not supporting_facts or not supporting_facts.get("title"):
        return RequiredHopChain(
            question_id=qid,
            question=question,
            answer=answer,
            answer_type=answer_type,
            hops=[],
            anchor_mentions=[],
            extraction_method=method,
            extraction_confidence=0.0,
            can_extract=False,
            abstain_reason="no_supporting_facts"
        )

    # Get context passages
    context = metadata.get("context", {})
    titles = context.get("title", [])
    sentences = context.get("sentences", [])

    # Build title -> sentences map
    passage_map = {}
    for i, title in enumerate(titles):
        if i < len(sentences):
            passage_map[title] = sentences[i]

    # Extract supporting sentences
    support_titles = supporting_facts["title"]
    support_sent_ids = supporting_facts["sent_id"]

    support_sentences = []
    for title, sent_id in zip(support_titles, support_sent_ids):
        if title in passage_map and sent_id < len(passage_map[title]):
            support_sentences.append({
                "title": title,
                "sent_id": sent_id,
                "text": passage_map[title][sent_id]
            })

    if not support_sentences:
        return RequiredHopChain(
            question_id=qid,
            question=question,
            answer=answer,
            answer_type=answer_type,
            hops=[],
            anchor_mentions=[],
            extraction_method=method,
            extraction_confidence=0.0,
            can_extract=False,
            abstain_reason="support_sentences_not_found"
        )

    # Extract anchor entities from question
    anchor_mentions = extract_entities_from_text(question)

    # Extract hops from supporting sentences
    hops = []

    for i, support in enumerate(support_sentences):
        # Extract entities from this support sentence
        entities = extract_entities_from_text(support["text"])

        # For bridge questions: typically 2 hops
        # For comparison: typically 2 parallel chains
        # We'll create simplified hops based on sentence order

        if i == 0:
            # First hop: usually from question entity to intermediate
            source = anchor_mentions[0] if anchor_mentions else "UNKNOWN"
            # Relation is implicit in the text (placeholder)
            relation = "relates_to"
            # Target is the main entity in this sentence
            target = entities[0] if entities else "UNKNOWN"

            hop = RequiredHop(
                source_mention=source,
                relation=relation,
                target_mention=target,
                support_title=support["title"],
                support_sent_id=support["sent_id"],
                support_text=support["text"],
                confidence=0.6  # Lower confidence for rule-based
            )
            hops.append(hop)

        else:
            # Subsequent hops: chain from previous
            prev_target = hops[-1].target_mention if hops else "UNKNOWN"
            source = prev_target
            relation = "relates_to"
            # Target might be the answer
            target = entities[0] if entities else answer

            hop = RequiredHop(
                source_mention=source,
                relation=relation,
                target_mention=target,
                support_title=support["title"],
                support_sent_id=support["sent_id"],
                support_text=support["text"],
                confidence=0.6
            )
            hops.append(hop)

    # Calculate overall confidence
    # Lower confidence because this is rule-based extraction
    overall_confidence = 0.5 if hops else 0.0

    return RequiredHopChain(
        question_id=qid,
        question=question,
        answer=answer,
        answer_type=answer_type,
        hops=hops,
        anchor_mentions=anchor_mentions,
        extraction_method=method,
        extraction_confidence=overall_confidence,
        can_extract=len(hops) > 0,
        abstain_reason=None if hops else "no_hops_extracted"
    )


def save_required_hops(
    hop_chains: List[RequiredHopChain],
    output_path: Path
) -> None:
    """Save extracted hop chains to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chain in hop_chains:
            f.write(json.dumps(asdict(chain)) + "\n")


def load_required_hops(input_path: Path) -> List[RequiredHopChain]:
    """Load extracted hop chains from JSON."""
    hop_chains = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert dict back to dataclass
                hops = [RequiredHop(**h) for h in data["hops"]]
                data["hops"] = hops
                hop_chains.append(RequiredHopChain(**data))

    return hop_chains


# Example usage
if __name__ == "__main__":
    # Test with a sample HotpotQA question
    sample_question = {
        "id": "5a7a06935542990198eaf050",
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "golden_answers": ["Arthur's Magazine"],
        "metadata": {
            "type": "comparison",
            "level": "medium",
            "supporting_facts": {
                "title": ["Arthur's Magazine", "First for Women"],
                "sent_id": [0, 0]
            },
            "context": {
                "title": ["Arthur's Magazine", "First for Women"],
                "sentences": [
                    ["Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century."],
                    ["First for Women is a woman's magazine published by Bauer Media Group in the USA.", "The magazine was started in 1989."]
                ]
            }
        }
    }

    from answer_type_classifier import classify_answer_type

    answer_class = classify_answer_type(
        sample_question["golden_answers"][0],
        sample_question["question"]
    )

    hop_chain = extract_required_hops_hotpotqa(
        sample_question,
        answer_class.answer_type
    )

    print("Required Hop Extraction Test")
    print("=" * 80)
    print(f"Question: {hop_chain.question}")
    print(f"Answer: {hop_chain.answer}")
    print(f"Answer Type: {hop_chain.answer_type}")
    print(f"Can Extract: {hop_chain.can_extract}")
    print(f"Confidence: {hop_chain.extraction_confidence:.2f}")
    print(f"Anchor Mentions: {hop_chain.anchor_mentions}")
    print(f"\nExtracted Hops ({len(hop_chain.hops)}):")

    for i, hop in enumerate(hop_chain.hops, 1):
        print(f"\nHop {i}:")
        print(f"  {hop.source_mention} --[{hop.relation}]--> {hop.target_mention}")
        print(f"  Support: {hop.support_title} (sent {hop.support_sent_id})")
        print(f"  Text: {hop.support_text[:80]}...")
        print(f"  Confidence: {hop.confidence:.2f}")
