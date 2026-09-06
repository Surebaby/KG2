from scripts.prepare.materialize_saeg_v1_training_graph_assets import make_wikidata_edges


def test_wikidata_edge_maps_exactly_to_runtime_hop_and_pid():
    record = {"qid": "q", "kg_subgraph": [["Ada", "place of birth", "London"]]}
    runtime = {
        "provenance": {"builder_version": "v"},
        "execution": {
            "hops": [{
                "hop_index": 1,
                "pids": ["P19"],
                "input_entities": [{"qid": "Q7259"}],
                "matches": [["Ada", "place of birth", "London"]],
                "output_entities": [{"surface": "London", "qid": "Q84"}],
            }]
        },
    }
    edges = make_wikidata_edges(record, runtime)
    assert len(edges) == 1
    assert edges[0]["provenance"]["pid"] == "P19"
    assert edges[0]["provenance"]["tail_qid"] == "Q84"
    assert edges[0]["construction_gold_access"] is False
