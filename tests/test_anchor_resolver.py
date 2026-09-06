"""Hotpot anchor resolver (passage-title supplement) tests."""

from __future__ import annotations

from kgproweight.kg.anchor_resolver import passage_titles, resolve_anchor


class _Result:
    def __init__(self, qid=None, abstained=False, score=1.0, margin=1.0):
        self.selected_qid = qid
        self.abstained = abstained
        self.score = score
        self.margin = margin


class _Resolver:
    def __init__(self, mapping):
        self.mapping = mapping  # surface -> qid or None (abstain)

    def resolve(self, surface):
        qid = self.mapping.get(surface, "__abstain__")
        if qid == "__abstain__" or qid is None:
            return _Result(abstained=True)
        return _Result(qid=qid)


class _Linker:
    def __init__(self, mapping, score=1.0, margin=1.0):
        self.mapping = mapping
        self.score = score
        self.margin = margin

    def link_single(self, surface, *, question):
        qid = self.mapping.get(surface, "__abstain__")
        if qid == "__abstain__" or qid is None:
            return _Result(abstained=True)
        return _Result(qid=qid, score=self.score, margin=self.margin)


def _passages(titles):
    return [{"title": t, "contents": "x"} for t in titles]


def test_planner_anchor_priority():
    resolver = _Resolver({"The Big Lebowski": "Q1"})
    linker = _Linker({})
    result, source = resolve_anchor("The Big Lebowski", "q", _passages(["Other"]), resolver, linker)
    assert result.selected_qid == "Q1"
    assert source == "planner_anchor"


def test_passage_title_exact_match_fallback():
    # anchor "The Big Lebowski" is disambiguated by the passage title
    # "The Big Lebowski (film)" (normalised prefix match).
    resolver = _Resolver({"The Big Lebowski (film)": "Q1"})
    linker = _Linker({})
    result, source = resolve_anchor("The Big Lebowski", "q", _passages(["The Big Lebowski (film)"]), resolver, linker)
    assert result.selected_qid == "Q1"
    assert source == "passage_title_fallback"


def test_passage_title_non_exact_no_fallback():
    resolver = _Resolver({})
    linker = _Linker({})
    result, source = resolve_anchor("The Big Lebowski", "q", _passages(["The Big Lebowski (film)"]), resolver, linker)
    assert source == "abstain"  # not exact normalised match -> no fallback


def test_conflict_abstains():
    resolver = _Resolver({})  # direct abstains
    linker = _Linker({})      # fallback also abstains
    result, source = resolve_anchor("Ambiguous", "q", _passages(["Ambiguous"]), resolver, linker)
    assert source == "abstain"


def test_low_confidence_reject():
    resolver = _Resolver({})
    linker = _Linker({"The Big Lebowski": "Q1"}, score=0.3, margin=0.05)  # below threshold
    result, source = resolve_anchor("The Big Lebowski", "q", _passages(["The Big Lebowski"]), resolver, linker)
    assert source == "abstain"


def test_duplicate_title_dedup():
    titles = passage_titles(_passages(["A", "A", "B", " a "]))
    assert titles == ["A", "B"]  # dedup + normalised order


def test_no_title_abstains():
    resolver = _Resolver({})
    linker = _Linker({})
    result, source = resolve_anchor("X", "q", _passages([]), resolver, linker)
    assert source == "abstain"
