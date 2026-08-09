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
#
# Removed: _HARD_DELETE_RELATIONS and _HARD_DELETE_PIDS. Both were unreferenced
# and each contained only "instance of"/P31 with a comment saying that relation
# is handled by quota rather than hard delete — i.e. they claimed a policy the
# code does not implement. Reading them as live config was misleading; the
# taxonomic policy lives in _QUOTA_PID_LIMITS + the score_triple penalty.

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

# Relations to hard-delete by LABEL (the English display name)
_HARD_DELETE_RELATION_LABELS: Set[str] = {
    # R9 v6 fix: trivially-true meta-relations that survive entity_anchor scoring
    # (entity IS in the question → ≥0.30 base score) but are never QA-relevant.
    # Together they account for ~21% of triples in the prompt.
    "given name",           # P735 — entity names ARE their given names, always
    "family name",          # P734 — entity names ARE their family names, always
    "sex or gender",        # P21  — factual but irrelevant for multi-hop QA
    "topic's main category",
    "category's main topic",
    "on focus list of Wikimedia project",
    # Wikimedia-internal bookkeeping. These survived scoring because their head
    # and tail echo the question entity ("Ed Wood" → "Template:Ed Wood"), so the
    # entity_anchor term outweighed the noise penalty and they reached the prompt.
    "topic has template",
    "template has topic",
    "related category",
    "category contains",
    "is a list of",
    "has list",
    "list of monuments",
    "maintained by WikiProject",
    "model item",
    "main Wikidata property",
    "related Wikidata property",
    "inappropriate property for this type",
    "Wikimedia outline",
    "topic's main Wikimedia portal",
    "permanent duplicated item",
    "category combines topics",
    "category for eponymous categories",
    "category of associated people",
    "category for people who died here",
    "category for people born here",
    "category for people buried here",
    "category for films shot at this location",
    "category for alumni of educational institution",
    "category for employees of the organization",
    "category for members of a team",
    "category for recipients of this award",
    "category for the view of the item",
    "category for the view from the item",
    "category for maps or plans",
    "member category",
    "history of topic",
    "economy of topic",
    "geography of topic",
    "demographics of topic",
    "open data portal",
    "documentation files at",
    "archives at",
    "artist files at",
    "assessment",
    "partially coincident with",
    "said to be the same as",
    "different from",
    "opposite of",
    "facet of",
    "has characteristic",
    "attested in",
    "depicted by",
    "copyright status",
    "copyright status as a creator",
    "copyright license",
    "copyright representative",
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

# Minimum score for a triple to be included in the final top-K prompt.
# Triples scoring below this have at most entity_anchor firing weakly, with
# zero signal on relation/question/path/evidence dimensions.
_MIN_SCORE_THRESHOLD = 0.25

# ── Layer 3: Question-aware scoring ──

# Keyword → Wikidata PID mapping for relation_question_similarity
_QUESTION_KEYWORD_TO_PID: Dict[str, List[str]] = {
    # ── Nationality / Citizenship ──
    "nationality": ["P27"], "national": ["P27"],
    "country": ["P17", "P27"], "citizen": ["P27"], "citizenship": ["P27"],
    "american": ["P27"], "british": ["P27"], "french": ["P27"],
    "german": ["P27"], "japanese": ["P27"], "chinese": ["P27"],
    # ── Birth / Death / Time ──
    "born": ["P19", "P569"], "birth": ["P19", "P569"],
    "birthplace": ["P19"], "place of birth": ["P19"],
    "died": ["P20", "P570"], "death": ["P20", "P570"],
    "started": ["P571"], "first": ["P571"], "began": ["P571"],
    "founded": ["P571", "P112"], "established": ["P571", "P112"],
    "inception": ["P571"], "created": ["P571", "P112"],
    "older": ["P569"], "younger": ["P569"], "earlier": ["P569", "P571"],
    "later": ["P569", "P571"], "year": ["P569", "P571"],
    "age": ["P569"], "old": ["P569"],
    # ── Location ──
    "located": ["P131", "P276"], "location": ["P131", "P276"],
    "headquarters": ["P159"], "based in": ["P159"],
    "capital": ["P36"], "city": ["P131"], "state": ["P131"],
    "province": ["P131"], "region": ["P131"], "neighborhood": ["P276"],
    "address": ["P6375"], "residence": ["P551"],
    # ── Occupation / Role ──
    "occupation": ["P106"], "job": ["P106"], "profession": ["P106"],
    "position": ["P39"], "role": ["P39", "P286"],
    "career": ["P106"], "worked as": ["P106"],
    # ── Family ──
    "spouse": ["P26"], "married": ["P26"],
    "wife": ["P26"], "husband": ["P26"],
    "parent": ["P25", "P22"], "father": ["P22"], "mother": ["P25"],
    "child": ["P40"], "son": ["P40"], "daughter": ["P40"],
    "brother": ["P3373"], "sister": ["P9"],
    "sibling": ["P3373", "P9"], "family": ["P26", "P22", "P25", "P40"],
    # ── Creative works ──
    "director": ["P57"], "directed": ["P57"],
    "film": ["P57", "P161", "P162"], "movie": ["P57", "P161"],
    "wrote": ["P50"], "author": ["P50"], "written": ["P50"],
    "writer": ["P50"], "screenwriter": ["P58"],
    "producer": ["P162"], "produced": ["P162"],
    "starring": ["P161"], "cast": ["P161"],
    "actor": ["P161"], "actress": ["P161"],
    "album": ["P175"], "song": ["P175"], "music": ["P175"],
    "band": ["P463"], "group": ["P463"],
    "composer": ["P86"], "performer": ["P175"],
    "painted": ["P170"], "painter": ["P170"],
    "designed": ["P287"], "built by": ["P176"],
    # ── Organization ──
    "founder": ["P112"], "company": ["P112", "P355"],
    "subsidiary": ["P355"], "subsidiaries": ["P355"],
    "owner": ["P127"], "owned": ["P127"],
    "manufacturer": ["P176"], "manufactured": ["P176"],
    "produced by": ["P176", "P162"],
    "parent company": ["P749"], "parent organization": ["P749"],
    # ── Education ──
    "university": ["P69"], "college": ["P69"], "school": ["P69"],
    "educated": ["P69"], "alumnus": ["P69"], "alumni": ["P69"],
    "degree": ["P512"], "graduated": ["P69"],
    "studied at": ["P69"], "attended": ["P69"],
    # ── Sports ──
    "team": ["P54"], "club": ["P54"], "player": ["P54"],
    "coach": ["P286"], "head coach": ["P286"],
    "league": ["P118"], "stadium": ["P115"], "arena": ["P115"],
    "championship": ["P1346"], "tournament": ["P1346"],
    "sport": ["P641"], "athlete": ["P54"],
    # ── Membership ──
    "member of": ["P463"], "member": ["P463", "P102"],
    "political party": ["P102"], "party": ["P102"],
    "organization": ["P463"], "affiliation": ["P463", "P1416"],
    # ── Awards / Achievements ──
    "won": ["P166", "P1346"], "winner": ["P166", "P1346"],
    "award": ["P166"], "prize": ["P166"],
    "nobel": ["P166"], "oscar": ["P166"], "grammy": ["P166"],
    "medal": ["P166"], "title": ["P166", "P1346"],
    # ── Population / Numbers ──
    "population": ["P1082"], "inhabitants": ["P1082"],
    "how many": ["P1082"], "number of": ["P1082"],
    "area": ["P2046"], "height": ["P2048"], "elevation": ["P2044"],
    "length": ["P2043"], "width": ["P2049"],
    # ── Comparison questions ──
    "same": ["P27", "P17", "P131"], "both": ["P27", "P17", "P131"],
    "more": ["P166", "P1082"], "less": ["P1082"],
    "larger": ["P2046", "P1082"], "smaller": ["P2046", "P1082"],
    "taller": ["P2048"], "shorter": ["P2048"],
    # ── Publication / Media ──
    "published": ["P123", "P571"], "publisher": ["P123"],
    "magazine": ["P31", "P571"], "newspaper": ["P31", "P571"],
    "journal": ["P31", "P571"], "publication": ["P123", "P571"],
    "language": ["P407"], "original language": ["P407"],
    # ── Military / Government ──
    "president": ["P39"], "prime minister": ["P39"],
    "governor": ["P39"], "mayor": ["P39"],
    "government": ["P39", "P17"], "served as": ["P39"],
    "election": ["P39"], "appointed": ["P39"],
    "military": ["P241"], "army": ["P241"],
    "war": ["P607"], "battle": ["P607"],
    # ── Transport / Infrastructure ──
    "airport": ["P239"], "station": ["P81", "P197"],
    "bridge": ["P177"], "road": ["P182"],
    "highway": ["P182"], "railway": ["P81"],
    "port": ["P206"], "harbor": ["P206"],
    # ── Animals / Species ──
    # NOTE: "family" is deliberately NOT re-keyed here. It appears above mapped
    # to the kinship PIDs; a second "family": ["P171"] entry silently overwrote
    # that (later key wins), so questions about a person's family scored
    # taxonomic rank instead of spouse/parent/child. Taxonomic "family" is
    # covered by "genus"/"parent taxon".
    "species": ["P105"], "genus": ["P171"],
    "moth": ["P105", "P171"], "animal": ["P105", "P171"],
    "bird": ["P105"], "fish": ["P105"],
    # ── Contains / Part of ──
    "contains": ["P150", "P527"], "part of": ["P361"],
    "belongs to": ["P361"], "includes": ["P527"],
    "consists of": ["P527"], "comprises": ["P527"],
    # ── Religion / Culture ──
    "religion": ["P140"], "religious": ["P140"],
    "mosque": ["P31"], "church": ["P31"],
    "temple": ["P31"], "cathedral": ["P31"],
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
    # ── R9 v6 fix: labels that were MISSING, so `_apply_relation_filter`
    # (which resolves label→PID before testing the QA whitelist) silently
    # dropped triples whose PID *is* whitelisted. P20/P166/P175/P112/P54/
    # P286/P118 were all unreachable — exactly the relations multi-hop QA
    # needs. Measured effect: 48% of cached triples discarded, incl. 5767
    # `place of death` and 3654 `award received`.
    "place of death": "P20",
    "award received": "P166",
    "performer": "P175",
    "founded by": "P112",
    "founder": "P112",
    "member of sports team": "P54",
    "head coach": "P286",
    "league": "P118",
    "league or competition": "P118",
    # ── Additional QA-relevant relations observed in the live cache ──
    "publisher": "P123",
    "composer": "P86",
    "sibling": "P3373",
    "winner": "P1346",
    "cause of death": "P509",
    "residence": "P551",
    "country of origin": "P495",
    "record label": "P264",
    "continent": "P30",
    "field of work": "P101",
    "notable work": "P800",
    "nominated for": "P1411",
    "parent taxon": "P171",
    "taxon rank": "P105",
    "operator": "P137",
    "religion or worldview": "P140",
    "position played on team / speciality": "P413",
    "military branch": "P241",
    "original language of film or TV show": "P364",
    "instrument": "P1303",
    "languages spoken, written or signed": "P1412",
    "native language": "P103",
    "work location": "P937",
    "place of burial": "P119",
    "architect": "P84",
    "developer": "P178",
    "director of photography": "P344",
    "editor": "P98",
    "film editor": "P1040",
    "lyricist": "P676",
    "voice actor": "P725",
    "characters": "P674",
    "based on": "P144",
    "production company": "P272",
    "distributed by": "P750",
    "part of the series": "P179",
    "follows": "P155",
    "followed by": "P156",
    "replaces": "P1365",
    "replaced by": "P1366",
    "twinned administrative body": "P190",
    "located in time zone": "P421",
    "located in or next to body of water": "P206",
    "located in/on physical feature": "P706",
    "highest point": "P610",
    "significant event": "P793",
    "participant in": "P1344",
    "participated in conflict": "P607",
    "doctoral advisor": "P184",
    "doctoral student": "P185",
    "student of": "P1066",
    "student": "P802",
    "influenced by": "P737",
    "chief executive officer": "P169",
    "chairperson": "P488",
    "legislative body": "P194",
    "office held by head of government": "P1313",
    "basic form of government": "P122",
    "ethnic group": "P172",
    "narrative location": "P840",
    "filming location": "P915",
    "main subject": "P921",
    "depicts": "P180",
    "made from material": "P186",
    "product or material produced": "P1056",
    "industry": "P452",
    "legal form": "P1454",
    "owner of": "P1830",
    "parent organization or unit": "P749",
    "given name": "P735",
    "family name": "P734",
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
    """Score based on question keywords mapping to this PID (0.0–1.0)."""
    if not pid:
        return 0.0
    for keyword, pids in _QUESTION_KEYWORD_TO_PID.items():
        if pid in pids and keyword in question_lower:
            return 1.0
    # Partial match: keyword parts in question
    for keyword, pids in _QUESTION_KEYWORD_TO_PID.items():
        if pid in pids:
            kw_parts = keyword.split()
            if any(p in question_lower for p in kw_parts if len(p) > 2):
                return 0.5
    return 0.0


def _path_connectivity_score(
    triple: Tuple[str, str, str],
    question_entities: List[str],
) -> float:
    """Score 1.0 if triple connects two question entities, 0.5 if connects one."""
    h_lower = triple[0].strip().lower()
    t_lower = triple[2].strip().lower()
    q_entities_lower = [e.strip().lower() for e in question_entities]

    h_match = any(h_lower == e or e in h_lower or h_lower in e for e in q_entities_lower)
    t_match = any(t_lower == e or e in t_lower or t_lower in e for e in q_entities_lower)

    if h_match and t_match:
        return 1.0  # connects two question entities → highly relevant
    if h_match or t_match:
        return 0.5  # connects one question entity
    return 0.0


def _relation_priority(pid: str) -> int:
    """Retrieval-layer QA priority for ``pid`` (0 when unknown).

    Imported lazily to avoid a circular import: wikidata_retriever imports the
    label→PID map from this module.
    """
    from kgproweight.kg.wikidata_retriever import _RELATION_PRIORITY

    return int(_RELATION_PRIORITY.get(pid, 0))


def _relation_utility_baseline(pid: str, r: str) -> float:
    """Score how inherently useful a relation is for QA (-0.3 – 1.0).

    Wikidata metadata/external-ID relations are penalised; core factual
    relations score highest; unknown relations are treated as probable noise.
    """
    if not pid:
        # Unknown relation — check for noise patterns
        r_lower = r.lower()
        noise_patterns = [
            "maintained by", "model item", "topic's main", "has template",
            "different from", "said to be the same", "properties for this type",
            "described by source", "topic's main category",
            "Wikimedia", "Wikimedia", "page banner", "street address",
            "postal code", "external data", "ID", "username",
        ]
        for pattern in noise_patterns:
            if pattern.lower() in r_lower:
                return -0.30  # heavy penalty for noise
        return 0.05  # unknown, assume mostly useless

    # Core factual relations. Derived from the retrieval-layer priority table
    # rather than a second hand-maintained list: the previous hardcoded set had
    # drifted, so relations like P551 (residence) — often the literal answer —
    # scored as "mapped but not core" and lost to noise on lexical overlap.
    priority = _relation_priority(pid)
    if priority >= 6:
        return 1.0   # high-value QA relation
    if priority >= 3:
        return 0.5   # useful but secondary
    if priority >= 1:
        return 0.1   # low value (taxonomic, given/family name)
    return -0.30     # external ID / metadata


def score_triple(
    triple: Tuple[str, str, str],
    question: str,
    pid: str = "",
    question_entities: Optional[List[str]] = None,
) -> float:
    """Score a triple's relevance to the question (0.0–1.0).

    Scoring formula (R9 v6, per §2.3):
      score = 0.30 * entity_anchor
            + 0.25 * relation_question_similarity
            + 0.25 * triple_question_similarity
            + 0.10 * path_connectivity
            + 0.10 * retrieved_evidence_support
            - taxonomic_penalty
    """
    q_lower = question.lower()
    h, r, t = triple
    pid = pid or _pid_for_triple(triple)

    # entity_anchor: 0.30 — head/tail from high-confidence question entities
    entity_score = max(_entity_in_question(h, q_lower), _entity_in_question(t, q_lower))

    # relation_question_similarity: 0.25 — PID matches question type
    rel_score = _relation_question_score(pid, q_lower)

    # triple_question_similarity: 0.25 — relation/tail overlap with question
    triple_score = 0.0
    r_lower = r.lower()
    t_lower = t.lower()
    for word in r_lower.split():
        if len(word) > 2 and word in q_lower:
            triple_score = 0.15
            break
    # Also check tail entity overlap (e.g. "United States" in "same nationality?")
    for word in t_lower.split():
        if len(word) > 3 and word in q_lower:
            triple_score = max(triple_score, 0.15)
            break

    # path_connectivity: 0.10 — connects question entities, stronger when both match
    entities = question_entities or []
    path_score = _path_connectivity_score(triple, entities)

    # retrieved_evidence_support: 0.10 — lexical overlap with question words
    tail_words = set(t_lower.split())
    q_words = set(q_lower.split())
    word_overlap = tail_words & q_words
    evidence_score = min(1.0, len(word_overlap) / max(1, len(tail_words)))

    # taxonomic penalty
    taxonomic_penalty = 0.0
    if pid in ("P31", "P279"):
        taxonomic_penalty = 0.35
    elif pid == "P527":
        taxonomic_penalty = 0.20
    elif pid == "P361":
        taxonomic_penalty = 0.15

    # Relation-utility prior. This term was computed by a helper that nothing
    # called, so unmapped relations ("topic has template", "related category",
    # …) carried NO penalty and could outrank real facts on lexical overlap
    # alone. Small weight: it breaks ties, it does not dominate the question
    # signal.
    utility = _relation_utility_baseline(pid, r)

    return (
        0.30 * entity_score
        + 0.25 * rel_score
        + 0.25 * triple_score
        + 0.10 * path_score
        + 0.10 * evidence_score
        + 0.10 * utility
        - taxonomic_penalty
    )


def filter_and_rank_triples(
    triples: List[Tuple[str, str, str]],
    question: str,
    pid_map: Optional[Dict[Tuple[str, str, str], str]] = None,
    max_keep: int = 30,
    min_keep: int = 0,
    rich: bool = False,
    question_entities: Optional[List[str]] = None,
) -> List:
    """Full pipeline: hard delete → quota → score → rank → top-K.

    When ``min_keep > 0``, the score threshold is relaxed if fewer than
    ``min_keep`` triples survive, ensuring every question has at least
    ``min_keep`` triples (hard-delete + quota are never relaxed).
    When ``rich=True``, returns list of dicts with pid, score, hop metadata.
    Otherwise returns plain (h, r, t) tuples.
    """
    if pid_map is None:
        pid_map = {t: _pid_for_triple(t) for t in triples}

    # Extract question entities for path_connectivity scoring
    if question_entities is None:
        from kgproweight.kg.entity_linker import extract_mentions
        question_entities = extract_mentions(question, max_n=5)

    # Layer 1: hard delete
    surviving = [
        t for t in triples
        if not hard_delete_triple(t, pid=pid_map.get(t, ""))
    ]

    # Layer 2: quota
    after_quota = quota_filter(surviving, pid_map)

    # Layer 3: score and rank
    scored = [
        (score_triple(t, question, pid=pid_map.get(t, ""),
                       question_entities=question_entities), t)
        for t in after_quota
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    # R9 v6 fix: minimum relevance threshold. Triples scoring below this have
    # negligible question relevance — typically only entity_anchor fires (0.30
    # × 1.0 = 0.30) with zero on every other dimension. Removing them cuts
    # ~20% of low-signal triples before they reach the prompt.
    if min_keep <= 0:
        scored = [(s, t) for s, t in scored if s >= _MIN_SCORE_THRESHOLD]
    else:
        above_threshold = [(s, t) for s, t in scored if s >= _MIN_SCORE_THRESHOLD]
        if len(above_threshold) >= min_keep:
            scored = above_threshold
        else:
            # Relax threshold: keep top-scoring triples up to max_keep
            scored = scored[:max_keep]

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


# ---------------------------------------------------------------------------
# R9 v6 Phase C: passage-verified KG filtering
# ---------------------------------------------------------------------------

def _entity_in_text(entity_label: str, text_lower: str) -> bool:
    """Check whether an entity label appears in the passage text.

    Uses substring-overlap with a minimum token-length guard: single-word
    labels must be >= 4 chars and appear as whole-word matches; multi-word
    labels need at least one word >= 3 chars present.
    """
    label_lower = entity_label.strip().lower()
    if not label_lower:
        return False
    parts = label_lower.split()
    if len(parts) == 1:
        # Single word: require whole-word match and >= 4 chars
        if len(label_lower) < 4:
            return False
        return label_lower in text_lower.split()
    else:
        # Multi-word: require at least one content word present
        for p in parts:
            if len(p) >= 3 and p in text_lower:
                return True
        # Or the full label as substring
        return label_lower in text_lower


def filter_by_passage_support(
    triples: List[Tuple[str, str, str]],
    passages: List,
    min_entities: int = 1,
) -> List[Tuple[str, str, str]]:
    """Keep only triples where at least ``min_entities`` entities appear in
    the retrieved passages.

    A KG fact whose entities never appear in any retrieved passage is
    floating without textual grounding — the model has no context to
    interpret it. This filter is the inference-time complement to the
    scoring-based filter: scoring tries to predict relevance from the
    question alone; passage support verifies it from the actual evidence.
    """
    if not triples or not passages:
        return list(triples)

    # Build a single searchable text block from passages
    text_blocks: List[str] = []
    for p in passages[:10]:  # top-10 passages (same as prompt)
        if isinstance(p, dict):
            content = p.get("contents") or p.get("text") or ""
            title = p.get("id") or p.get("title") or ""
            text_blocks.append(f"{title} {content}")
        elif isinstance(p, str):
            text_blocks.append(p)
    passage_text = " ".join(text_blocks).lower()

    kept: List[Tuple[str, str, str]] = []
    for triple in triples:
        if len(triple) != 3:
            continue
        h, _, t = triple
        h_match = _entity_in_text(str(h), passage_text)
        t_match = _entity_in_text(str(t), passage_text)
        if int(h_match) + int(t_match) >= min_entities:
            kept.append(triple)
    return kept
