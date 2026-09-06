"""Store-first, historical-fallback edge retrieval for exact (QID, PID).

The historical-hybrid backend was previously *replacing* the versioned evidence
store's property edges with the historical cache, which regressed 2Wiki
property coverage (15 non-empty -> 11).  This retriever fixes that with a single
variable: for each (QID, PID) it queries the store first and only consults the
historical cache when the store has no edge for that PID.  It never concatenates
both sources for the same PID, so no duplicate or conflicting edges are emitted.

Every returned edge carries a ``source`` field ("store" or
"historical_fallback"); ``source_counts`` aggregates them for the run report.
Both underlying retrievers must already be offline (no network).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class StoreFirstCombinedRetriever:
    def __init__(self, store: Any, historical: Any) -> None:
        self.store = store
        self.historical = historical
        self.source_counts: Dict[str, int] = {"store": 0, "historical_fallback": 0}

    def _dedup_key(self, edge: Dict[str, Any], pid: str) -> tuple:
        return (
            str(edge.get("head_qid") or ""),
            str(pid),
            str(edge.get("tail_qid") or ""),
            str(edge.get("tail_value") or ""),
        )

    def fetch_edges(self, qid: str, pids: Sequence[str]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen: set = set()
        for pid in pids:
            store_edges = self.store.fetch_edges(qid, [pid])
            if store_edges:
                source, edges = "store", store_edges
            else:
                source, edges = "historical_fallback", self.historical.fetch_edges(qid, [pid])
            for edge in edges:
                tagged = dict(edge)
                tagged["source"] = source
                key = self._dedup_key(tagged, pid)
                if key in seen:
                    continue
                seen.add(key)
                result.append(tagged)
                self.source_counts[source] += 1
        return result
