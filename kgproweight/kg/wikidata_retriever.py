"""Wikidata SPARQL subgraph retriever (1- and 2-hop)."""

from __future__ import annotations

import time
from typing import List, Optional, Set, Tuple

import requests

from kgproweight.kg.cache import SubgraphCache
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
        return f"""
SELECT ?headLabel ?propLabel ?tailLabel WHERE {{
  wd:{qid} ?prop ?tail .
  ?propEntity wikibase:directClaim ?prop .
  ?propEntity rdfs:label ?propLabel . FILTER(LANG(?propLabel)="en")
  wd:{qid} rdfs:label ?headLabel . FILTER(LANG(?headLabel)="en")
  ?tail rdfs:label ?tailLabel . FILTER(LANG(?tailLabel)="en")
  {filter_clause}
}} ORDER BY ?prop LIMIT {limit}
"""

    def _build_2hop_query(self, qid: str) -> str:
        limit = self.max_neighbors
        filter_clause = ""
        if self.relation_filter:
            pids = " ".join(f"wdt:{p}" for p in self.relation_filter)
            filter_clause = f"FILTER(?p1 IN ({pids}) && ?p2 IN ({pids}))"
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
}} ORDER BY ?p1 ?p2 LIMIT {limit}
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
