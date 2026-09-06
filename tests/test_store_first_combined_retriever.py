"""Store-first, historical-fallback combined retriever tests."""

from __future__ import annotations

from kgproweight.kg.store_first_combined_retriever import StoreFirstCombinedRetriever


class _Fake:
    def __init__(self, edges):
        # edges: {(qid, pid): [edge, ...]}
        self.edges = edges
        self.calls = []

    def fetch_edges(self, qid, pids):
        out = []
        for pid in pids:
            out.extend(self.edges.get((str(qid), str(pid)), []))
        self.calls.append((qid, list(pids)))
        return out


def _edge(qid="Q1", pid="P17", label="X", relation="country", value="United States", tail="Q30"):
    return {
        "head_qid": qid, "head_label": label, "pid": pid,
        "relation": relation, "tail_qid": tail, "tail_value": value,
    }


def _make(store_edges, hist_edges):
    store = _Fake(store_edges)
    hist = _Fake(hist_edges)
    retriever = StoreFirstCombinedRetriever(store, hist)
    return retriever, store, hist


def test_store_hit_uses_store_without_historical():
    store_edge = _edge()
    retriever, store, hist = _make({("Q1", "P17"): [store_edge]}, {("Q1", "P17"): [_edge(value="Other")]})
    out = retriever.fetch_edges("Q1", ["P17"])
    assert len(out) == 1
    assert out[0]["source"] == "store"
    assert out[0]["tail_value"] == "United States"
    assert hist.calls == []  # historical never consulted
    assert retriever.source_counts == {"store": 1, "historical_fallback": 0}


def test_store_miss_falls_back_to_historical():
    hist_edge = _edge(value="Netherlands")
    retriever, store, hist = _make({}, {("Q1", "P17"): [hist_edge]})
    out = retriever.fetch_edges("Q1", ["P17"])
    assert len(out) == 1
    assert out[0]["source"] == "historical_fallback"
    assert out[0]["tail_value"] == "Netherlands"
    assert retriever.source_counts == {"store": 0, "historical_fallback": 1}


def test_multiple_pids_fallback_independently():
    store_edge = _edge(pid="P17", value="United States")
    hist_edge = _edge(pid="P569", relation="date of birth", value="1946")
    retriever, store, hist = _make(
        {("Q1", "P17"): [store_edge]},
        {("Q1", "P569"): [hist_edge]},
    )
    out = retriever.fetch_edges("Q1", ["P17", "P569"])
    by_pid = {e["pid"]: e for e in out}
    assert by_pid["P17"]["source"] == "store"
    assert by_pid["P569"]["source"] == "historical_fallback"
    assert retriever.source_counts == {"store": 1, "historical_fallback": 1}


def test_dedup_within_source():
    edge = _edge()
    retriever, store, hist = _make({("Q1", "P17"): [edge, dict(edge)]}, {})
    out = retriever.fetch_edges("Q1", ["P17"])
    assert len(out) == 1
    assert retriever.source_counts == {"store": 1, "historical_fallback": 0}


def test_deterministic_output():
    retriever, _, _ = _make(
        {("Q1", "P17"): [_edge()]},
        {("Q1", "P569"): [_edge(pid="P569", value="1946")]},
    )
    first = retriever.fetch_edges("Q1", ["P17", "P569"])
    second = retriever.fetch_edges("Q1", ["P17", "P569"])
    assert first == second


def test_empty_result_when_both_miss():
    retriever, store, hist = _make({}, {})
    out = retriever.fetch_edges("Q1", ["P17"])
    assert out == []
    assert retriever.source_counts == {"store": 0, "historical_fallback": 0}
    assert len(hist.calls) == 1  # store missed, historical was consulted once
