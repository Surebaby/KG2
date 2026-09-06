import json

from kgproweight.kg.historical_wikidata_retriever import (
    HISTORICAL_CACHE_VERSION,
    HistoricalWikidataPropertyRetriever,
)


def test_historical_retriever_reads_frozen_revision_cache(tmp_path) -> None:
    cutoff = "2020-12-09T23:59:59Z"
    cache = tmp_path / "history.jsonl"
    entity = {
        "labels": {"en": {"value": "Person"}},
        "claims": {
            "P569": [{
                "rank": "normal",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {
                        "type": "time",
                        "value": {"time": "+1900-01-02T00:00:00Z", "precision": 11},
                    },
                },
            }]
        },
    }
    cache.write_text(json.dumps({
        "schema_version": HISTORICAL_CACHE_VERSION,
        "key": f"{HISTORICAL_CACHE_VERSION}::{cutoff}::Q1",
        "qid": "Q1",
        "cutoff": cutoff,
        "revision": {"revid": 7, "timestamp": "2020-12-01T00:00:00Z"},
        "entity": entity,
    }) + "\n", encoding="utf-8")
    retriever = HistoricalWikidataPropertyRetriever(
        cache_path=cache, cutoff=cutoff, offline=True
    )
    edges = retriever.fetch_edges("Q1", ["P569"])
    assert len(edges) == 1
    assert edges[0]["tail_value"] == "2 January 1900"
    assert edges[0]["tail_raw_value"] == "1900-01-02"
    assert edges[0]["source_revision_id"] == "7"


def test_historical_retriever_preserves_tail_qid_from_cached_entities(tmp_path) -> None:
    cutoff = "2020-12-09T23:59:59Z"
    cache = tmp_path / "history.jsonl"
    rows = [
        {
            "schema_version": HISTORICAL_CACHE_VERSION,
            "key": f"{HISTORICAL_CACHE_VERSION}::{cutoff}::Q1",
            "qid": "Q1", "cutoff": cutoff, "revision": {"revid": 1},
            "entity": {
                "labels": {"en": {"value": "A"}},
                "claims": {"P26": [{
                    "rank": "normal",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q2"}},
                    },
                }]},
            },
        },
        {
            "schema_version": HISTORICAL_CACHE_VERSION,
            "key": f"{HISTORICAL_CACHE_VERSION}::{cutoff}::Q2",
            "qid": "Q2", "cutoff": cutoff, "revision": {"revid": 2},
            "entity": {"labels": {"en": {"value": "B"}}, "claims": {}},
        },
    ]
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    edges = HistoricalWikidataPropertyRetriever(
        cache_path=cache, cutoff=cutoff, offline=True
    ).fetch_edges("Q1", ["P26"])
    assert edges[0]["tail_qid"] == "Q2"
    assert edges[0]["tail_value"] == "B"
    assert edges[0]["tail_raw_value"] is None


def test_fetch_all_edges_scans_exact_entity_claims_without_expansion(tmp_path) -> None:
    cutoff = "2020-12-09T23:59:59Z"
    cache = tmp_path / "history.jsonl"
    cache.write_text(json.dumps({
        "schema_version": HISTORICAL_CACHE_VERSION,
        "key": f"{HISTORICAL_CACHE_VERSION}::{cutoff}::Q1",
        "qid": "Q1", "cutoff": cutoff, "revision": {"revid": 3},
        "entity": {
            "labels": {"en": {"value": "Work"}},
            "claims": {
                "P57": [{"rank": "normal", "mainsnak": {
                    "snaktype": "value", "datavalue": {
                        "type": "wikibase-entityid", "value": {"id": "Q2"},
                    },
                }}],
                "P577": [{"rank": "normal", "mainsnak": {
                    "snaktype": "value", "datavalue": {
                        "type": "time", "value": {
                            "time": "+2001-00-00T00:00:00Z", "precision": 9,
                        },
                    },
                }}],
            },
        },
    }) + "\n", encoding="utf-8")
    retriever = HistoricalWikidataPropertyRetriever(
        cache_path=cache, cutoff=cutoff, offline=True
    )
    edges = retriever.fetch_all_edges("Q1")
    assert {edge["pid"] for edge in edges} == {"P57", "P577"}
    assert {edge["head_qid"] for edge in edges} == {"Q1"}


def test_historical_retriever_renders_month_precision_without_gold(tmp_path) -> None:
    cutoff = "2020-12-09T23:59:59Z"
    cache = tmp_path / "history.jsonl"
    entity = {
        "labels": {"en": {"value": "Event"}},
        "claims": {
            "P571": [{
                "rank": "normal",
                "mainsnak": {
                    "snaktype": "value",
                    "datavalue": {
                        "type": "time",
                        "value": {"time": "+1900-07-00T00:00:00Z", "precision": 10},
                    },
                },
            }]
        },
    }
    cache.write_text(json.dumps({
        "schema_version": HISTORICAL_CACHE_VERSION,
        "key": f"{HISTORICAL_CACHE_VERSION}::{cutoff}::Q1",
        "qid": "Q1",
        "cutoff": cutoff,
        "revision": {"revid": 8},
        "entity": entity,
    }) + "\n", encoding="utf-8")
    edge = HistoricalWikidataPropertyRetriever(
        cache_path=cache, cutoff=cutoff, offline=True
    ).fetch_edges("Q1", ["P571"])[0]
    assert edge["tail_value"] == "July 1900"
    assert edge["tail_raw_value"] == "1900-07"


def test_request_delay_throttles_after_successful_fetch(tmp_path, monkeypatch) -> None:
    """--request_delay must sleep after every successful request (429 prevention).

    Regression guard for the Wikidata HTTP 429 that aborted the first prefetch
    round at 10/1500: a request_delay of 0 is a footgun, and the retriever must
    honour the configured per-request throttle.
    """
    cutoff = "2020-12-09T23:59:59Z"
    cache = tmp_path / "history.jsonl"
    cache.write_text("", encoding="utf-8")

    content = json.dumps({"labels": {"en": {"value": "X"}}, "claims": {}})
    payload = {"query": {"pages": [{"revisions": [{
        "slots": {"main": {"content": content}},
        "revid": 1,
        "timestamp": "2020-01-01T00:00:00Z",
    }]}]}}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return payload

    sleeps: list[float] = []
    monkeypatch.setattr(
        "kgproweight.kg.historical_wikidata_retriever.requests.get",
        lambda *a, **k: _FakeResponse(),
    )
    monkeypatch.setattr(
        "kgproweight.kg.historical_wikidata_retriever.time.sleep",
        lambda s: sleeps.append(s),
    )

    retriever = HistoricalWikidataPropertyRetriever(
        cache_path=cache, cutoff=cutoff, offline=False,
        request_delay=0.4, max_retries=1,
    )
    entity, _ = retriever._entity("Q1")
    assert entity is not None
    assert sleeps == [0.4]


def test_request_delay_backoff_on_retry(tmp_path, monkeypatch) -> None:
    """Retry sleep scales with attempt (request_delay * attempt)."""
    cutoff = "2020-12-09T23:59:59Z"
    cache = tmp_path / "history.jsonl"
    cache.write_text("", encoding="utf-8")

    sleeps: list[float] = []

    def _boom(*a, **k):
        raise __import__("requests").RequestException("429 Too Many Requests")

    monkeypatch.setattr(
        "kgproweight.kg.historical_wikidata_retriever.requests.get", _boom
    )
    monkeypatch.setattr(
        "kgproweight.kg.historical_wikidata_retriever.time.sleep",
        lambda s: sleeps.append(s),
    )

    retriever = HistoricalWikidataPropertyRetriever(
        cache_path=cache, cutoff=cutoff, offline=False,
        request_delay=0.4, max_retries=3,
    )
    import pytest
    with pytest.raises(Exception):
        retriever._entity("Q1")
    # attempt 1 fail -> sleep(0.4*1); attempt 2 fail -> sleep(0.4*2). No post-success sleep.
    assert sleeps == [0.4, 0.8]
