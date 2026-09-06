import pytest

from scripts.pilot.build_conservative_hybrid_overrides import build_hybrid_passages


def _passage(doc_id, title, body="body"):
    return {"id": doc_id, "contents": f"{title}\n{body}"}


def test_hybrid_preserves_old_prefix_and_deduplicates_bridge_pages():
    old = [_passage(str(i), f"Old {i}") for i in range(12)]
    bridge = [
        _passage("0", "Different title", "same id as stored"),
        _passage("b1", '"New Page"', "first chunk"),
        _passage("b2", "New Page", "second chunk"),
        _passage("b3", "Another Page"),
    ]

    result, counts = build_hybrid_passages(
        old, bridge, old_keep=2, bridge_keep=3, total=5
    )

    assert result[:2] == old[:2]
    assert [row["id"] for row in result] == ["0", "1", "b1", "b3", "2"]
    assert counts["bridge_added"] == 2
    assert counts["old_backfilled"] == 1
    assert counts["bridge_skipped_duplicate"] == 2


def test_hybrid_requires_quotas_to_equal_total():
    with pytest.raises(ValueError, match="must equal total"):
        build_hybrid_passages([], [], old_keep=10, bridge_keep=4, total=15)


def test_hybrid_does_not_mutate_inputs():
    old = [_passage("o1", "Old")]
    bridge = [_passage("b1", "New")]
    old_before = list(old)
    bridge_before = list(bridge)

    build_hybrid_passages(old, bridge, old_keep=1, bridge_keep=1, total=2)

    assert old == old_before
    assert bridge == bridge_before


def test_hybrid_fills_missing_old_prefix_slots_with_bridge_passages():
    old = [_passage("o1", "Old")]
    bridge = [_passage(f"b{i}", f"New {i}") for i in range(4)]

    result, counts = build_hybrid_passages(
        old, bridge, old_keep=3, bridge_keep=2, total=5
    )

    assert len(result) == 5
    assert counts["old_prefix"] == 1
    assert counts["bridge_target"] == 4
    assert counts["bridge_added"] == 4


def test_hybrid_backfills_duplicate_old_tail_when_bridge_is_exhausted():
    old = [
        _passage("o1", "Repeated Page", "prefix chunk"),
        _passage("o2", "Repeated Page", "tail chunk"),
    ]
    bridge = [_passage("b1", "Repeated Page", "bridge duplicate")]

    result, counts = build_hybrid_passages(
        old, bridge, old_keep=1, bridge_keep=1, total=2
    )

    assert [row["id"] for row in result] == ["o1", "o2"]
    assert counts["bridge_added"] == 0
    assert counts["old_backfilled"] == 1
