"""Wikidata SPARQL subgraph retriever (1- and 2-hop).

R9 v6: SPARQL queries use relation-importance ordering instead of alphabetical.
QA-relevant relations (nationality, occupation, location, etc.) are returned
before administrative/metadata relations (instance of, genre, external IDs).
"""

from __future__ import annotations

import time
from typing import List, Optional, Set, Tuple

import requests

from kgproweight.kg.cache import SubgraphCache

# R9 v6: Relation priority for SPARQL ORDER BY.
# Higher score = more QA-relevant = returned first (before LIMIT truncation).
# Relations not in this dict get default score 0 (lowest priority).
_RELATION_PRIORITY: dict[str, int] = {
    # Identity/type (useful but low priority — one per entity is enough)
    "P31": 1,   # instance of
    "P279": 1,  # subclass of
    # Compositional relations: kept in the candidate pool (2Wiki location
    # questions need P150) but scored down by kg_filter's taxonomic penalty
    # rather than dropped outright at the retrieval layer.
    "P150": 2,  # contains the administrative territorial entity
    "P361": 1,  # part of
    "P527": 1,  # has part(s)
    # Core biographical
    "P27": 10,  # country of citizenship
    "P19": 9,   # place of birth
    "P20": 9,   # place of death
    "P569": 8,  # date of birth
    "P570": 8,  # date of death
    "P106": 8,  # occupation
    "P39": 8,   # position held
    "P21": 5,   # sex or gender
    # Family
    "P26": 7,   # spouse
    "P22": 7,   # father
    "P25": 7,   # mother
    "P40": 6,   # child
    # Location/geography
    "P17": 9,   # country
    "P131": 8,  # located in administrative entity
    "P276": 7,  # location
    "P159": 8,  # headquarters location
    "P36": 7,   # capital
    "P47": 5,   # shares border with
    # Organization
    "P112": 7,  # founded by
    "P571": 8,  # inception (founding date)
    "P576": 8,  # dissolved date
    "P127": 7,  # owned by
    "P176": 7,  # manufacturer
    "P355": 6,  # subsidiary
    "P749": 6,  # parent organization
    # Creative/film
    "P57": 8,   # director
    "P58": 8,   # screenwriter
    "P161": 8,  # cast member
    "P162": 7,  # producer
    "P50": 8,   # author
    "P170": 7,  # creator
    "P175": 7,  # performer
    # Education
    "P69": 7,   # educated at
    "P108": 7,  # employer
    # Sports
    "P54": 6,   # member of sports team
    "P118": 5,  # league
    "P286": 6,  # head coach
    # Membership
    "P463": 6,  # member of
    "P102": 5,  # member of political party
    # Awards
    "P166": 6,  # award received
    # Language/genre
    "P136": 4,  # genre
    "P407": 3,  # language of work
    # ── R9 v6 fix: QA-relevant relations that were absent from the whitelist,
    # so `_QA_RELATION_FILTER` discarded them even after label→PID resolution.
    "P509": 6,  # cause of death
    "P1346": 6, # winner
    "P123": 6,  # publisher
    "P86": 7,   # composer
    "P3373": 6, # sibling
    "P551": 6,  # residence
    "P495": 6,  # country of origin
    "P264": 5,  # record label
    "P30": 6,   # continent
    "P101": 5,  # field of work
    "P800": 6,  # notable work
    "P413": 5,  # position played on team
    "P241": 5,  # military branch
    "P364": 5,  # original language of film/TV
    "P1303": 4, # instrument
    "P103": 5,  # native language
    "P937": 5,  # work location
    "P119": 5,  # place of burial
    "P84": 6,   # architect
    "P178": 6,  # developer
    "P344": 5,  # director of photography
    "P98": 5,   # editor
    "P676": 5,  # lyricist
    "P725": 5,  # voice actor
    "P674": 5,  # characters
    "P144": 5,  # based on
    "P272": 6,  # production company
    "P750": 4,  # distributed by
    "P179": 4,  # part of the series
    "P155": 4,  # follows
    "P156": 4,  # followed by
    "P190": 3,  # twinned administrative body
    "P206": 5,  # located in/next to body of water
    "P706": 4,  # located in/on physical feature
    "P610": 4,  # highest point
    "P793": 4,  # significant event
    "P1344": 5, # participant in
    "P607": 6,  # conflict
    "P184": 5,  # doctoral advisor
    "P185": 4,  # doctoral student
    "P1066": 5, # student of
    "P737": 4,  # influenced by
    "P169": 6,  # chief executive officer
    "P488": 5,  # chairperson
    "P194": 5,  # legislative body
    "P1313": 5, # office held by head of government
    "P122": 4,  # basic form of government
    "P172": 4,  # ethnic group
    "P840": 5,  # narrative location
    "P915": 4,  # filming location
    "P921": 5,  # main subject
    "P186": 4,  # made from material
    "P1056": 4, # product or material produced
    "P452": 4,  # industry
    "P1830": 4, # owner of
    "P137": 5,  # operator
    "P140": 5,  # religion or worldview
    "P171": 4,  # parent taxon
    "P105": 4,  # taxon rank
    "P735": 2,  # given name
    "P734": 2,  # family name
    "P6": 7,    # head of government
    "P35": 7,   # head of state
    "P138": 5,  # named after
    "P37": 5,   # official language
    "P38": 4,   # currency
    "P1082": 5, # population
    "P1411": 4, # nominated for
    "P1412": 4, # languages spoken/written/signed
    # External IDs (lowest — noise for QA)
    "P213": -1, "P214": -1, "P244": -1, "P245": -1, "P268": -1, "P269": -1,
    "P345": -1, "P434": -1, "P435": -1, "P496": -1, "P646": -1, "P672": -1,
    "P785": -1, "P856": -1, "P910": -1, "P932": -1, "P973": -1, "P1006": -1,
    "P1015": -1, "P1146": -1, "P1343": -1,
    "P1551": -1, "P1552": -1, "P1566": -1, "P1630": -1,
    "P1667": -1, "P1670": -1, "P1741": -1, "P1810": -1, "P2002": -1,
    "P2003": -1, "P2013": -1, "P2163": -1, "P2276": -1, "P2397": -1,
    "P2427": -1, "P2581": -1, "P2685": -1, "P2699": -1, "P2847": -1,
    "P2888": -1, "P2959": -1, "P2963": -1, "P3138": -1, "P3153": -1,
    "P3162": -1, "P3184": -1, "P3219": -1, "P3265": -1, "P3267": -1,
    "P3365": -1, "P3417": -1, "P3452": -1, "P3496": -1, "P3544": -1,
    "P3569": -1, "P3734": -1, "P3782": -1, "P3937": -1, "P3984": -1,
    "P4012": -1, "P4025": -1, "P4033": -1, "P4073": -1, "P4084": -1,
    "P4125": -1, "P4145": -1, "P4147": -1, "P4159": -1, "P4196": -1,
}

# R9 v6: QA-relevant relations for silver data generation.
# When passed as relation_filter, SPARQL returns ONLY these properties.
_QA_RELATION_FILTER: set[str] = {
    k for k, v in _RELATION_PRIORITY.items() if v >= 1  # exclude -1 (external IDs)
}


from kgproweight.utils.logging import get_logger

import os as _os

logger = get_logger(__name__)

# R9 v6: support reverse proxy for firewalled environments (same env vars as entity_linker).
_WIKIDATA_PROXY_BASE = _os.getenv("KGPW_WIKIDATA_PROXY_BASE", "").rstrip("/")
_WIKIDATA_PROXY_TOKEN = _os.getenv("KGPW_WIKIDATA_PROXY_TOKEN", "")

if _WIKIDATA_PROXY_BASE:
    _WIKIDATA_API_BASE = f"{_WIKIDATA_PROXY_BASE}/https://query.wikidata.org"
else:
    _WIKIDATA_API_BASE = "https://query.wikidata.org"

SPARQL_ENDPOINT = f"{_WIKIDATA_API_BASE}/sparql"
SPARQL_HEADERS = {
    "User-Agent": "KGProWeight/1.0 (research; contact: anonymous@example.com)",
    "Accept": "application/sparql-results+json",
}
if _WIKIDATA_PROXY_TOKEN:
    SPARQL_HEADERS["X-Proxy-Token"] = _WIKIDATA_PROXY_TOKEN

REQUEST_DELAY = 0.5


def _apply_relation_filter(
    triples: List[Tuple[str, str, str]], relation_filter: set,
) -> List[Tuple[str, str, str]]:
    """Filter triples in Python by QA-relevant relation PIDs (Layer 1 post-filter)."""
    from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID
    import re
    filtered = []
    for t in triples:
        pid = _RELATION_LABEL_TO_PID.get(t[1].lower(), "")
        if not pid:
            m = re.match(r"^(P\d+)", t[1])
            pid = m.group(1) if m else ""
        if pid in relation_filter:
            filtered.append(t)
    return filtered


def _sparql_query(query: str, retries: int = 3, timeout: int = 30) -> Optional[dict]:
    # Use POST for long queries (GET URL length limit ~8000 chars)
    use_post = len(query) > 2000
    for attempt in range(retries):
        try:
            if use_post:
                resp = requests.post(
                    SPARQL_ENDPOINT,
                    data={"query": query, "format": "json"},
                    headers={**SPARQL_HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
                    timeout=timeout,
                )
            else:
                resp = requests.get(
                    SPARQL_ENDPOINT,
                    params={"query": query, "format": "json"},
                    headers=SPARQL_HEADERS,
                    timeout=timeout,
                )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("SPARQL attempt %d/%d failed: %s", attempt + 1, retries, exc)
            time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


class WikidataSubgraphRetriever:
    """Retrieves a 2-hop subgraph from Wikidata with optional disk caching."""

    def __init__(
        self,
        max_hops: int = 2,
        max_neighbors: int = 30,
        relation_filter: Optional[Set[str]] = None,
        cache_dir: Optional[str] = None,
        request_delay: float = REQUEST_DELAY,
        offline: bool = False,
    ) -> None:
        if max_hops not in (1, 2):
            raise ValueError("max_hops must be 1 or 2")
        self.max_hops = max_hops
        self.max_neighbors = max_neighbors
        self.relation_filter = relation_filter
        self.request_delay = request_delay
        # offline=True: never hit the SPARQL endpoint. Cache hits still return
        # real subgraphs; a miss returns [] INSTANTLY (no 30s×3 SPARQL timeout,
        # no inter-request sleep). Use when query.wikidata.org is unreachable.
        self.offline = offline
        cache_path = None
        if cache_dir:
            from pathlib import Path

            cache_path = Path(cache_dir) / "kg_subgraph_cache.jsonl"
        self.cache = SubgraphCache(cache_path)

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    def _build_1hop_query(self, qid: str) -> str:
        limit = self.max_neighbors
        return f"""
SELECT ?headLabel ?propLabel ?tailLabel WHERE {{
  wd:{qid} ?prop ?tail .
  ?propEntity wikibase:directClaim ?prop .
  ?propEntity rdfs:label ?propLabel . FILTER(LANG(?propLabel)="en")
  wd:{qid} rdfs:label ?headLabel . FILTER(LANG(?headLabel)="en")
  ?tail rdfs:label ?tailLabel . FILTER(LANG(?tailLabel)="en")
}} LIMIT {limit}
"""

    def _build_2hop_query(self, qid: str) -> str:
        limit = self.max_neighbors
        return f"""
SELECT ?headLabel ?p1Label ?midLabel ?p2Label ?tailLabel WHERE {{
  wd:{qid} ?p1 ?mid .
  ?p1Ent wikibase:directClaim ?p1 .
  ?p1Ent rdfs:label ?p1Label . FILTER(LANG(?p1Label)="en")
  wd:{qid} rdfs:label ?headLabel . FILTER(LANG(?headLabel)="en")
  ?mid rdfs:label ?midLabel . FILTER(LANG(?midLabel)="en")
  ?mid ?p2 ?tail .
  ?p2Ent wikibase:directClaim ?p2 .
  ?p2Ent rdfs:label ?p2Label . FILTER(LANG(?p2Label)="en")
  ?tail rdfs:label ?tailLabel . FILTER(LANG(?tailLabel)="en")
}} LIMIT {limit}
"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _raw_cache_keys(self, qid: str) -> List[str]:
        """Cache keys to try for ``qid``, newest format first.

        ``relation_filter`` is a *Python* post-filter (SPARQL returns everything),
        so it must NOT be part of the key: keying on it stored already-filtered
        triples under a filter-specific key, which (a) duplicated storage per
        filter and (b) silently missed the 63k entries written under the older
        ``{qid}_{hops}`` scheme, making the whole on-disk cache dead weight in
        offline mode. We now cache RAW subgraphs and filter at read time.
        """
        return [
            f"{qid}_{self.max_hops}_{self.max_neighbors}",   # current
            f"{qid}_{self.max_hops}_{self.max_neighbors}_all",  # R9 v6 interim
            f"{qid}_{self.max_hops}",                        # legacy (bulk of cache)
        ]

    def _cached_raw(self, qid: str) -> Optional[List[Tuple[str, str, str]]]:
        for key in self._raw_cache_keys(qid):
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        return None

    def fetch(self, entity_ids: List[str]) -> List[Tuple[str, str, str]]:
        """Aggregate the deduplicated 2-hop subgraph for a list of QIDs."""
        all_triples: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()

        for qid in entity_ids:
            raw = self._cached_raw(qid)
            if raw is None:
                raw = self._fetch_single(qid)
                # Cache the RAW subgraph so a later relation-policy change reuses
                # it. Never cache empty (a failed/blocked fetch must be retried).
                if raw:
                    self.cache.set(self._raw_cache_keys(qid)[0], raw)
                if not self.offline:
                    time.sleep(self.request_delay)

            triples = _apply_relation_filter(raw, self.relation_filter) if self.relation_filter else raw
            for triple in triples:
                if triple not in seen:
                    all_triples.append(triple)
                    seen.add(triple)
        return all_triples

    def _fetch_single(self, qid: str) -> List[Tuple[str, str, str]]:
        if self.offline:
            return []
        triples: List[Tuple[str, str, str]] = []

        result = _sparql_query(self._build_1hop_query(qid))
        if result:
            for row in result.get("results", {}).get("bindings", []):
                try:
                    triples.append(
                        (
                            row["headLabel"]["value"],
                            row["propLabel"]["value"],
                            row["tailLabel"]["value"],
                        )
                    )
                except KeyError:
                    continue

        if self.max_hops == 2:
            r2 = _sparql_query(self._build_2hop_query(qid))
            if r2:
                for row in r2.get("results", {}).get("bindings", []):
                    try:
                        h = row["headLabel"]["value"]
                        r1 = row["p1Label"]["value"]
                        m = row["midLabel"]["value"]
                        r_p2 = row["p2Label"]["value"]
                        t = row["tailLabel"]["value"]
                        triples.append((h, r1, m))
                        triples.append((m, r_p2, t))
                    except KeyError:
                        continue
        return triples
