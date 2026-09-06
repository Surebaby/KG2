from kgproweight.kg.wikidata_property_retriever import WikidataPropertyRetriever


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_targeted_property_fetch_preserves_literals_and_entity_labels(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, *, params, headers, timeout):
        calls.append(dict(params))
        if params["props"] == "labels|claims":
            return _Response(
                {
                    "entities": {
                        "Q42": {
                            "labels": {"en": {"value": "Douglas Adams"}},
                            "claims": {
                                "P569": [
                                    {
                                        "rank": "normal",
                                        "mainsnak": {
                                            "snaktype": "value",
                                            "datavalue": {
                                                "type": "time",
                                                "value": {
                                                    "time": "+1952-03-11T00:00:00Z",
                                                    "precision": 11,
                                                },
                                            },
                                        },
                                    }
                                ],
                                "P27": [
                                    {
                                        "rank": "normal",
                                        "mainsnak": {
                                            "snaktype": "value",
                                            "datavalue": {
                                                "type": "wikibase-entityid",
                                                "value": {"id": "Q145"},
                                            },
                                        },
                                    }
                                ],
                            },
                        }
                    }
                }
            )
        return _Response(
            {"entities": {"Q145": {"labels": {"en": {"value": "United Kingdom"}}}}}
        )

    monkeypatch.setattr("kgproweight.kg.wikidata_property_retriever.requests.get", fake_get)
    cache_path = tmp_path / "properties.jsonl"
    retriever = WikidataPropertyRetriever(cache_path=cache_path, request_delay=0)
    triples = retriever.fetch_properties("Q42", ["P569", "P27"])
    assert triples == [
        ("Douglas Adams", "country of citizenship", "United Kingdom"),
        ("Douglas Adams", "date of birth", "1952-03-11"),
    ]
    assert len(calls) == 2
    # A fresh edge-aware cache retains the QID needed for iterative traversal.
    edge_cache_path = tmp_path / "edge_properties.jsonl"
    edge_retriever = WikidataPropertyRetriever(
        cache_path=tmp_path / "edge_triples.jsonl",
        edge_cache_path=edge_cache_path,
        request_delay=0,
    )
    edges = edge_retriever.fetch_edges("Q42", ["P27", "P569"])
    country = next(edge for edge in edges if edge["pid"] == "P27")
    assert country["tail_qid"] == "Q145"
    assert country["tail_value"] == "United Kingdom"
    offline_edges = WikidataPropertyRetriever(
        cache_path=tmp_path / "edge_triples.jsonl",
        edge_cache_path=edge_cache_path,
        offline=True,
    ).fetch_edges("Q42", ["P27", "P569"])
    assert offline_edges == edges

    def fail_get(*args, **kwargs):
        raise AssertionError("offline cache hit must not use network")

    monkeypatch.setattr("kgproweight.kg.wikidata_property_retriever.requests.get", fail_get)
    offline = WikidataPropertyRetriever(cache_path=cache_path, offline=True)
    assert offline.fetch_properties("Q42", ["P27", "P569"]) == triples


def test_successful_missing_property_is_cached(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, *, params, headers, timeout):
        calls.append(dict(params))
        return _Response(
            {"entities": {"Q1": {"labels": {"en": {"value": "Universe"}}, "claims": {}}}}
        )

    monkeypatch.setattr("kgproweight.kg.wikidata_property_retriever.requests.get", fake_get)
    cache_path = tmp_path / "properties.jsonl"
    retriever = WikidataPropertyRetriever(cache_path=cache_path, request_delay=0)
    assert retriever.fetch_properties("Q1", ["P569"]) == []
    assert retriever.fetch_properties("Q1", ["P569"]) == []
    assert len(calls) == 1
