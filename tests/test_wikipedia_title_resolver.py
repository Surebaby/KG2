from kgproweight.kg.wikipedia_title_resolver import (
    WikipediaTitleResolver,
    complete_question_surface_title,
    title_variants,
)


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "query": {
                "pages": [
                    {
                        "title": "Appointment with Death (film)",
                        "pageprops": {"wikibase_item": "Q618139"},
                    }
                ]
            }
        }


class _RedirectAgreementResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "query": {
                "pages": [
                    {"title": "Appointment with Death", "pageprops": {"wikibase_item": "Q618139"}},
                    {"title": "Appointment with Death (film)", "pageprops": {"wikibase_item": "Q618139"}},
                ]
            }
        }


def test_title_variants_normalize_dataset_title_case():
    assert "Appointment with Death (film)" in title_variants("Appointment With Death (Film)")
    assert "Robert Shaw (Royal Navy officer)" in title_variants(
        "Robert Shaw (Royal Navy Officer)"
    )
    assert "Giant (Stan Rogers song)" in title_variants("Giant (Stan Rogers Song)")
    assert "Shah Shuja (Mughal prince)" in title_variants(
        "Shah Shuja (Mughal Prince)"
    )
    assert "Stephen Marley (musician)" in title_variants(
        "Stephen Marley (Musician)"
    )


def test_title_resolver_persists_isolated_qid_cache(tmp_path, monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs["params"])
        return _Response()

    monkeypatch.setattr("kgproweight.kg.wikipedia_title_resolver.requests.get", fake_get)
    cache_path = tmp_path / "titles.jsonl"
    resolver = WikipediaTitleResolver(cache_path=cache_path, request_delay=0)
    result = resolver.resolve("Appointment With Death (Film)")
    assert result.selected_qid == "Q618139"
    assert result.selected_label == "Appointment with Death (film)"
    assert "Appointment with Death (film)" in calls[0]["titles"]

    monkeypatch.setattr(
        "kgproweight.kg.wikipedia_title_resolver.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must use cache")),
    )
    offline = WikipediaTitleResolver(cache_path=cache_path, offline=True)
    assert offline.resolve("Appointment With Death (Film)").selected_qid == "Q618139"


def test_title_variants_agreeing_on_one_qid_are_not_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kgproweight.kg.wikipedia_title_resolver.requests.get",
        lambda *args, **kwargs: _RedirectAgreementResponse(),
    )
    resolver = WikipediaTitleResolver(cache_path=tmp_path / "titles.jsonl", request_delay=0)
    result = resolver.resolve("Appointment With Death (Film)")
    assert result.selected_qid == "Q618139"
    assert not result.abstained


def test_complete_question_surface_title_restores_visible_disambiguator():
    assert complete_question_surface_title(
        "The Deer",
        "When was the director of The Deer (Film) born?",
    ) == "The Deer (Film)"


def test_complete_question_surface_title_is_conservative():
    assert complete_question_surface_title("The Deer", "Who directed The Deer?") == "The Deer"
    assert complete_question_surface_title(
        "The Deer (film)", "Who directed The Deer (film)?"
    ) == "The Deer (film)"
    assert complete_question_surface_title("Unknown", "Who directed The Deer (film)?") == "Unknown"
