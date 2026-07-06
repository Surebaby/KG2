"""Coverage and density helpers for KG subgraphs."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz


def coverage_score(
    query_entities: List[str],
    linked_entities: Dict[str, Optional[str]],
) -> float:
    """Fraction of query entities successfully linked to a Wikidata QID."""
    if not query_entities:
        return 0.0
    n = sum(1 for e in query_entities if linked_entities.get(e) is not None)
    return n / len(query_entities)


def graph_density(triples: List[Tuple[str, str, str]]) -> float:
    """``|E| / (|V| + ε)``; 0.0 for empty triple lists."""
    if not triples:
        return 0.0
    nodes: Set[str] = set()
    for h, _, t in triples:
        nodes.add(h)
        nodes.add(t)
    return len(triples) / (len(nodes) + 1e-6)


def triple_in_subgraph(
    triple: Tuple[str, str, str],
    subgraph: List[Tuple[str, str, str]],
    fuzzy_threshold: float = 85.0,
) -> bool:
    """Exact-then-fuzzy match check for a (h, r, t) triple inside ``subgraph``."""
    h, r, t = triple
    h_low, r_low, t_low = h.lower(), r.lower(), t.lower()
    for sh, sr, st in subgraph:
        if h_low == sh.lower() and r_low == sr.lower() and t_low == st.lower():
            return True
        if (
            fuzz.token_sort_ratio(h_low, sh.lower()) >= fuzzy_threshold
            and fuzz.token_sort_ratio(r_low, sr.lower()) >= fuzzy_threshold
            and fuzz.token_sort_ratio(t_low, st.lower()) >= fuzzy_threshold
        ):
            return True
    return False
