"""KG triple filtering and scoring for question_kg_index construction.

Three-layer strategy:
  1. Hard delete: Wikimedia metadata, disambiguation, empty labels, self-loops
  2. Quota limits: instance_of ≤ 2/entity, subclass_of ≤ 2/entity, same PID ≤ 20%
  3. Question-aware scoring: entity_anchor + relation similarity + passage support

Output format (v2 rich):
  {
    "question_id": "hotpotqa_<hash>",
    "question": "...",
    "linked_entities": [{"mention": "...", "qid": "Q...", "score": 0.9, ...}],
    "triples": [{"h": "...", "pid": "P27", "r": "country of citizenship",
                 "t": "...", "score": 0.88, "hop": 1}],
    "builder_version": "r9v6-kg-1",
    "relation_policy_version": "rel-1"
  }
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# ── Layer 1: Hard deletion ──

# Relations to ALWAYS drop (Wikimedia meta, disambiguation, maintenance)
_HARD_DELETE_RELATIONS: Set[str] = frozenset({
    "instance of",                          # handled by quota, not hard delete
})

# Head/tail entity labels to ALWAYS drop
_HARD_DELETE_TAIL_LABELS: Set[str] = {
    "Wikimedia disambiguation page",
    "Wikimedia category",
    "Wikimedia list",
    "Wikimedia information list",
    "Wikimedia internal item",
    "Wikimedia article covering multiple topics",
    "Wikimedia article covering two opposite properties or topics",
    "MediaWiki main namespace page",
    "disambiguation page",
    "Wikipedia disambiguation page",
}

# Relations that are Wikimedia-internal (always noise for QA)
_HARD_DELETE_PIDS: Set[str] = {
    "P31",  # instance of — handled by quota, not hard delete
}

# Relations to hard-delete by LABEL (the English display name)
_HARD_DELETE_RELATION_LABELS: Set[str] = {
    "topic's main category",
    "category's main topic",
    "on focus list of Wikimedia project",
    "described by source",
    "properties for this type",
    "said to be the same as",
    "different from",
    "Wikimedia import URL",
    "Freebase ID",
    "Quora topic ID",
    "Google Knowledge Graph ID",
    "external data available at",
    "page banner",
    "located on street",
    "street address",
    "postal code",
    "official website",
    "Commons category",
    "image",
    "logo image",
    "locator map image",
    "detail map",
    "coat of arms image",
    "flag image",
    "seal image",
    "has list",
    "related image",
    "audio",
    "video",
    "pronunciation audio",
    "described at URL",
    "reference URL",
    "ISBN-13",
    "ISBN-10",
    "DOI",
    "PubMed ID",
    "arXiv ID",
    "Library of Congress authority ID",
    "VIAF ID",
    "GND ID",
    "BNF ID",
    "ORCID ID",
    "ISNI",
    "IMDb ID",
    "MusicBrainz artist ID",
    "Discogs artist ID",
    "AllMusic artist ID",
    "X username",
    "Facebook username",
    "Instagram username",
    "YouTube channel ID",
    "LinkedIn company ID",
    "GitHub username",
    "Stack Exchange tag",
    "Twitter username",
    "subreddit",
}

# ── Layer 2: Quota limits ──

_QUOTA_PID_LIMITS: Dict[str, int] = {
    "P31": 1,   # instance of: max 1 per entity
    "P279": 1,  # subclass of: max 1 per entity
}

_MAX_SAME_RELATION_RATIO = 0.15  # same PID ≤ 15% of final top-K

# Global taxonomic cap: instance_of + subclass_of ≤ 20% of final prompt
_MAX_TAXONOMIC_RATIO = 0.20

# ── Layer 3: Question-aware scoring ──

# Keyword → Wikidata PID mapping for relation_question_similarity
_QUESTION_KEYWORD_TO_PID: Dict[str, List[str]] = {
    # Nationality / location
    "nationality": ["P27"],
    "country": ["P17", "P27"],
    "citizen": ["P27"],
    "born": ["P19"],
    "birth": ["P19"],
    "birthplace": ["P19"],
    "place of birth": ["P19"],
    "died": ["P20"],
    "death": ["P20"],
    "located": ["P131", "P276"],
    "location": ["P131", "P276"],
    "headquarters": ["P159"],
    "capital": ["P36"],
    # Occupation / role
    "occupation": ["P106"],
    "job": ["P106"],
    "profession": ["P106"],
    "position": ["P39"],
    "role": ["P39", "P286"],
    "member": ["P463", "P102"],
    # Family
    "spouse": ["P26"],
    "married": ["P26"],
    "wife": ["P26"],
    "husband": ["P26"],
    "parent": ["P25", "P22"],
    "father": ["P22"],
    "mother": ["P25"],
    "child": ["P40"],
    "son": ["P40"],
    "daughter": ["P40"],
    "brother": ["P3373"],
    "sister": ["P9"],
    # Works / creation
    "director": ["P57"],
    "directed": ["P57"],
    "film": ["P57", "P161", "P162"],
    "movie": ["P57", "P161"],
    "wrote": ["P50"],
    "author": ["P50"],
    "written": ["P50"],
    "producer": ["P162"],
    "produced": ["P162"],
    "starring": ["P161"],
    "cast": ["P161"],
    "actor": ["P161"],
    "actress": ["P161"],
    "album": ["P175"],
    "song": ["P175"],
    "music": ["P175"],
    "band": ["P463"],
    "group": ["P463"],
    # Organization
    "founded": ["P112"],
    "founder": ["P112"],
    "company": ["P112", "P355"],
    "subsidiary": ["P355"],
    "owner": ["P127"],
    "owned": ["P127"],
    "manufacturer": ["P176"],
    "built": ["P176"],
    # Education
    "university": ["P69"],
    "college": ["P69"],
    "school": ["P69"],
    "educated": ["P69"],
    "alumnus": ["P69"],
    "degree": ["P512"],
    # Sports
    "team": ["P54"],
    "club": ["P54"],
    "player": ["P54"],
    "coach": ["P286"],
    "league": ["P118"],
    "stadium": ["P115"],
    "arena": ["P115"],
    "championship": ["P1346"],
    "tournament": ["P1346"],
}


_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b")

# Common English words that look like entities but aren't
_STOP_ENTITIES: Set[str] = {
    "which", "what", "where", "when", "who", "whom", "whose", "why", "how",
    "are", "is", "was", "were", "been", "the", "this", "that", "these", "those",
    "first", "second", "last", "same", "both", "each", "every", "all", "some",
    "does", "did", "name", "year", "years", "many", "more", "most",
}

# Relation label → PID mapping for common relations
_RELATION_LABEL_TO_PID: Dict[str, str] = {
    "instance of": "P31",
    "subclass of": "P279",
    "part of": "P361",
    "has part(s)": "P527",
    "country": "P17",
    "country of citizenship": "P27",
    "place of birth": "P19",
    "date of birth": "P569",
    "date of death": "P570",
    "occupation": "P106",
    "position held": "P39",
    "member of": "P463",
    "member of political party": "P102",
    "spouse": "P26",
    "father": "P22",
    "mother": "P25",
    "child": "P40",
    "educated at": "P69",
    "employer": "P108",
    "director": "P57",
    "cast member": "P161",
    "producer": "P162",
    "screenwriter": "P58",
    "author": "P50",
    "creator": "P170",
    "manufacturer": "P176",
    "owned by": "P127",
    "parent organization": "P749",
    "subsidiary": "P355",
    "headquarters location": "P159",
    "located in the administrative territorial entity": "P131",
    "location": "P276",
    "capital": "P36",
    "shares border with": "P47",
    "sex or gender": "P21",
    "genre": "P136",
    "language of work or name": "P407",
    "named after": "P138",
    "inception": "P571",
    "dissolved, abolished or demolished date": "P576",
    "contains the administrative territorial entity": "P150",
    "head of government": "P6",
    "head of state": "P35",
    "population": "P1082",
    "area": "P2046",
    "elevation above sea level": "P2044",
    "official language": "P37",
    "currency": "P38",
    "time zone": "P421",
    "coordinate location": "P625",
    "described by source": "P1343",
    "properties for this type": "P1963",
    "said to be the same as": "P460",
    "different from": "P1889",
    "topic's main category": "P910",
}


def make_question_id(question: str, dataset: str = "") -> str:
    """Generate a stable question_id from the question text."""
    h = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:12]
    prefix = f"{dataset}_" if dataset else ""
    return f"{prefix}{h}"


def extract_question_entities(question: str) -> List[Dict[str, Any]]:
    """Extract multi-word capitalized entity mentions from the question.

    Filters out common English question words and single capitalized words
    that are likely not entities.
    """
    seen = set()
    entities = []
    for m in _ENTITY_RE.finditer(question):
        mention = m.group(1).strip()
        key = mention.lower()
        if key in seen or key in _STOP_ENTITIES:
            continue
        # Require at least 2 words or be clearly a proper noun (>5 chars)
        if " " not in mention and len(mention) <= 5:
            continue
        seen.add(key)
        entities.append({"mention": mention, "qid": None, "score": 0.0, "type": None})
    return entities


def hard_delete_triple(
    triple: Tuple[str, str, str],
    pid: str = "",
) -> bool:
    """Return True if this triple should be hard-deleted."""
    h, r, t = triple

    # Disambiguation / list / category pages as tail
    if t in _HARD_DELETE_TAIL_LABELS:
        return True
    if h in _HARD_DELETE_TAIL_LABELS:
        return True

    # Metadata relations by label
    if r in _HARD_DELETE_RELATION_LABELS:
        return True

    # Self-loop (head == tail)
    if h.lower() == t.lower():
        return True

    # Empty labels
    if not h.strip() or not r.strip() or not t.strip():
        return True

    # No English label heuristic: contains high Unicode chars (non-Latin scripts)
    # Skip for now — Wikidata should have English labels via our SPARQL FILTER

    return False


def quota_filter(
    triples: List[Tuple[str, str, str]],
    pid_map: Dict[Tuple[str, str, str], str],
) -> List[Tuple[str, str, str]]:
    """Apply quota limits: instance_of ≤ 2/entity, subclass_of ≤ 2, same PID ≤ 20%."""
    instance_of = {}
    subclass_of = {}
    pid_counts: Dict[str, int] = {}
    kept: List[Tuple[str, str, str]] = []

    for t in triples:
        pid = pid_map.get(t, "")

        # Instance-of quota
        if pid == "P31":
            entity = t[0]
            n = instance_of.get(entity, 0)
            if n >= _QUOTA_PID_LIMITS.get("P31", 2):
                continue
            instance_of[entity] = n + 1

        # Subclass-of quota
        if pid == "P279":
            entity = t[0]
            n = subclass_of.get(entity, 0)
            if n >= _QUOTA_PID_LIMITS.get("P279", 2):
                continue
            subclass_of[entity] = n + 1

        # Per-PID ratio (checked after collecting)
        pid_counts[pid] = pid_counts.get(pid, 0) + 1
        kept.append(t)

    # Clamp per-PID ratio
    total = len(kept)
    max_per_pid = int(total * _MAX_SAME_RELATION_RATIO)

    # Build ratio-clamped result
    result = []
    clamped_pid_counts: Dict[str, int] = {}
    for t in kept:
        pid = pid_map.get(t, "")
        n = clamped_pid_counts.get(pid, 0)
        if total > 0 and n >= max(max_per_pid, 1):
            continue
        clamped_pid_counts[pid] = n + 1
        result.append(t)

    return result


def _pid_for_triple(triple: Tuple[str, str, str]) -> str:
    """Extract PID from a triple's relation string or label mapping."""
    r = triple[1]
    # Direct PID prefix match: "P27" or "P27 country of citizenship"
    m = re.match(r"^(P\d+)", r)
    if m:
        return m.group(1)
    # Fallback: label → PID mapping
    return _RELATION_LABEL_TO_PID.get(r.lower(), "")


def _entity_in_question(entity: str, question_lower: str) -> float:
    """Score 1.0 if entity phrase appears in question, 0.5 if partial match, 0.0 otherwise."""
    e = entity.strip().lower()
    if not e:
        return 0.0
    if e in question_lower:
        return 1.0
    # Partial: first/last name matches
    parts = e.split()
    if len(parts) >= 2:
        if parts[-1] in question_lower:
            return 0.5
    return 0.0


def _relation_question_score(pid: str, question_lower: str) -> float:
    """Score based on question keywords mapping to this PID."""
    if not pid:
        return 0.0
    for keyword, pids in _QUESTION_KEYWORD_TO_PID.items():
        if pid in pids and keyword in question_lower:
            return 1.0
    return 0.0


def score_triple(
    triple: Tuple[str, str, str],
    question: str,
    pid: str = "",
) -> float:
    """Score a triple's relevance to the question (0.0–1.0)."""
    q_lower = question.lower()
    h, r, t = triple
    pid = pid or _pid_for_triple(triple)

    # entity_anchor: 0.30 weight
    entity_score = max(_entity_in_question(h, q_lower), _entity_in_question(t, q_lower))
    # relation_question_similarity: 0.25 weight
    rel_score = _relation_question_score(pid, q_lower)
    # triple_question_similarity: 0.25 weight — simplified: relation label overlap
    triple_score = 0.0
    r_lower = r.lower()
    for word in r_lower.split():
        if len(word) > 2 and word in q_lower:
            triple_score = 0.25
            break

    # Taxonomic penalty: instance_of / subclass_of / has_part are rarely useful
    # Penalty must offset entity_anchor (0.30) so these only survive with
    # additional question-relevance signal (relation or triple similarity).
    taxonomic_penalty = 0.0
    if pid in ("P31", "P279"):  # instance of, subclass of
        taxonomic_penalty = 0.35
    elif pid == "P527":  # has part(s)
        taxonomic_penalty = 0.20
    elif pid == "P361":  # part of
        taxonomic_penalty = 0.15

    return (
        0.30 * entity_score
        + 0.25 * rel_score
        + 0.25 * triple_score
        - taxonomic_penalty
    )


def filter_and_rank_triples(
    triples: List[Tuple[str, str, str]],
    question: str,
    pid_map: Optional[Dict[Tuple[str, str, str], str]] = None,
    max_keep: int = 30,
    rich: bool = False,
) -> List:
    """Full pipeline: hard delete → quota → score → rank → top-K.

    When ``rich=True``, returns list of dicts with pid, score, hop metadata.
    Otherwise returns plain (h, r, t) tuples.
    """
    if pid_map is None:
        pid_map = {t: _pid_for_triple(t) for t in triples}

    # Layer 1: hard delete
    surviving = [
        t for t in triples
        if not hard_delete_triple(t, pid=pid_map.get(t, ""))
    ]

    # Layer 2: quota
    after_quota = quota_filter(surviving, pid_map)

    # Layer 3: score and rank
    scored = [
        (score_triple(t, question, pid=pid_map.get(t, "")), t)
        for t in after_quota
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Post-process: enforce global taxonomic cap
    selected = scored[:max_keep * 2]
    tax_triples = []
    non_tax = []
    for s, t in selected:
        pid = pid_map.get(t, "")
        if pid in ("P31", "P279"):
            tax_triples.append((s, t))
        else:
            non_tax.append((s, t))

    actual_total = min(max_keep, len(tax_triples) + len(non_tax))
    max_tax = max(0, int(actual_total * _MAX_TAXONOMIC_RATIO))
    result_scores = []
    result_scores.extend(tax_triples[:max_tax])
    result_scores.extend(non_tax)
    result_scores.sort(key=lambda x: x[0], reverse=True)
    result_scores = result_scores[:max_keep]

    if rich:
        return [
            {
                "h": t[0],
                "pid": pid_map.get(t, ""),
                "r": t[1],
                "t": t[2],
                "score": round(s, 4),
                "hop": 1,  # v1 cache doesn't store hop; assume 1-hop
            }
            for s, t in result_scores
        ]
    return [t for _, t in result_scores]
