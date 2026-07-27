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
    # External IDs (lowest — noise for QA)
    "P213": -1, "P214": -1, "P244": -1, "P245": -1, "P268": -1, "P269": -1,
    "P345": -1, "P434": -1, "P435": -1, "P496": -1, "P646": -1, "P672": -1,
    "P785": -1, "P856": -1, "P910": -1, "P932": -1, "P973": -1, "P1006": -1,
    "P1015": -1, "P1082": -1, "P1146": -1, "P1343": -1, "P1411": -1,
    "P1412": -1, "P1551": -1, "P1552": -1, "P1566": -1, "P1630": -1,
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


def _build_relation_order_case(var: str = "?prop") -> str:
    """Build a SPARQL CASE expression for relation importance ordering."""
    cases = []
    for pid, score in sorted(_RELATION_PRIORITY.items(), key=lambda x: -x[1]):
        if score > 0:
            cases.append(f"    WHEN({var} = wdt:{pid}) THEN {score} ")
    return "CASE\n" + "\n".join(cases) + "\n    ELSE 0\nEND"
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_HEADERS = {
    "User-Agent": "KGProWeight/1.0 (research; contact: anonymous@example.com)",
    "Accept": "application/sparql-results+json",
}
REQUEST_DELAY = 0.5


def _sparql_query(query: str, retries: int = 3, timeout: int = 30) -> Optional[dict]:
    for attempt in range(retries):
        try:
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
        filter_clause = ""
        if self.relation_filter:
            pids = " ".join(f"wdt:{p}" for p in self.relation_filter)
            filter_clause = f"FILTER(?prop IN ({pids}))"
        rel_order = _build_relation_order_case()
        return f"""
SELECT ?headLabel ?propLabel ?tailLabel WHERE {{
  wd:{qid} ?prop ?tail .
  ?propEntity wikibase:directClaim ?prop .
  ?propEntity rdfs:label ?propLabel . FILTER(LANG(?propLabel)="en")
  wd:{qid} rdfs:label ?headLabel . FILTER(LANG(?headLabel)="en")
  ?tail rdfs:label ?tailLabel . FILTER(LANG(?tailLabel)="en")
  {filter_clause}
  BIND({rel_order} AS ?propScore)
}} ORDER BY DESC(?propScore) LIMIT {limit}
"""

    def _build_2hop_query(self, qid: str) -> str:
        limit = self.max_neighbors
        filter_clause = ""
        if self.relation_filter:
            pids = " ".join(f"wdt:{p}" for p in self.relation_filter)
            filter_clause = f"FILTER(?p1 IN ({pids}) && ?p2 IN ({pids}))"
        rel_order = _build_relation_order_case()
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
  {filter_clause}
  BIND({rel_order} AS ?p1Score)
  BIND({_build_relation_order_case("?p2")} AS ?p2Score)
  BIND(IF(?p1Score > ?p2Score, ?p1Score, ?p2Score) AS ?maxPropScore)
}} ORDER BY DESC(?maxPropScore) LIMIT {limit}
"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, entity_ids: List[str]) -> List[Tuple[str, str, str]]:
        """Aggregate the deduplicated 2-hop subgraph for a list of QIDs."""
        all_triples: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()

        for qid in entity_ids:
            filter_tag = "_".join(sorted(self.relation_filter)) if self.relation_filter else "all"
            cache_key = f"{qid}_{self.max_hops}_{self.max_neighbors}_{filter_tag}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                for triple in cached:
                    if triple not in seen:
                        all_triples.append(triple)
                        seen.add(triple)
                continue

            triples = self._fetch_single(qid)
            # In offline mode a miss yields []; do NOT persist that empty result,
            # so a later networked run still fetches it for real (no cache poison).
            if not (self.offline and not triples):
                self.cache.set(cache_key, triples)
            for triple in triples:
                if triple not in seen:
                    all_triples.append(triple)
                    seen.add(triple)

            if not self.offline:
                time.sleep(self.request_delay)
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
