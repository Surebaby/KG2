"""Resolve Wikipedia-like question anchors to Wikidata QIDs by page title.

Multi-hop QA entity surfaces frequently originate from Wikipedia titles.  An
exact page-title lookup is therefore safer than treating every title as a
free-form Wikidata search query.  This resolver is opt-in and uses an isolated
append-only cache so legacy entity-linking experiments remain reproducible.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List

import requests

from kgproweight.kg.cache import EntityCache
from kgproweight.kg.entity_linker import LinkResult, WIKIDATA_USER_AGENT


_PROXY_BASE = os.getenv("KGPW_WIKIDATA_PROXY_BASE", "").rstrip("/")
_PROXY_TOKEN = os.getenv("KGPW_WIKIDATA_PROXY_TOKEN", "")
_WIKIPEDIA_BASE = (
    f"{_PROXY_BASE}/https://en.wikipedia.org" if _PROXY_BASE else "https://en.wikipedia.org"
)
WIKIPEDIA_API_URL = f"{_WIKIPEDIA_BASE}/w/api.php"
_LOWER_WORDS = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
_PAREN_TYPE_WORDS = {
    "actor", "album", "book", "director", "emperor", "film", "instrumental",
    "leader", "movie", "musician", "novel", "officer", "prince", "series", "song",
}


def complete_question_surface_title(surface: str, question: str) -> str:
    """Restore an adjacent parenthetical disambiguator from the question.

    The helper only copies text already present in the question. It never
    consults supporting facts, answers, aliases, or an external knowledge
    source. Surfaces that are absent, already parenthesised, or not followed
    immediately by a parenthetical are returned unchanged.
    """
    clean = re.sub(r"\s+", " ", str(surface).strip())
    if not clean or re.search(r"\([^()]+\)\s*$", clean):
        return clean
    match = re.search(re.escape(clean), str(question), flags=re.IGNORECASE)
    if not match:
        return clean
    suffix = re.match(r"\s*(\([^()\n]{1,100}\))", str(question)[match.end():])
    if not suffix:
        return clean
    return f"{clean} {suffix.group(1)}"


def title_variants(surface: str) -> List[str]:
    clean = re.sub(r"\s+", " ", str(surface).strip())
    variants = [clean]
    words = clean.split(" ")
    normalized = " ".join(
        word.lower() if index and word.lower().strip("()") in _LOWER_WORDS else word
        for index, word in enumerate(words)
    )
    parenthetical = re.search(r"\(([^()]*)\)$", normalized)
    if parenthetical:
        inside = parenthetical.group(1)
        inside_words = inside.split(" ")
        if inside_words and inside_words[-1].lower() in _PAREN_TYPE_WORDS:
            inside_words[-1] = inside_words[-1].lower()
            normalized = normalized[:parenthetical.start()] + f"({' '.join(inside_words)})"
    # Do not remove a parenthetical disambiguator: ``Title (film)`` and
    # ``Title`` often denote a film and its source novel, respectively.
    if normalized and normalized not in variants:
        variants.append(normalized)
    return variants


class WikipediaTitleResolver:
    def __init__(
        self,
        *,
        cache_path: str | Path,
        offline: bool = False,
        timeout: float = 15.0,
        max_retries: int = 2,
        request_delay: float = 0.25,
    ) -> None:
        self.cache = EntityCache(cache_path)
        self.offline = bool(offline)
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.request_delay = float(request_delay)

    def resolve(self, surface: str) -> LinkResult:
        cached = self.cache.get(surface)
        if cached:
            return LinkResult(
                mention=surface, selected_qid=cached, selected_label=surface,
                score=1.0, margin=1.0,
            )
        if self.offline:
            return LinkResult(mention=surface, abstained=True, abstain_reason="title cache miss")
        variants = title_variants(surface)
        headers = {"User-Agent": WIKIDATA_USER_AGENT}
        if _PROXY_TOKEN:
            headers["X-Proxy-Token"] = _PROXY_TOKEN
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    WIKIPEDIA_API_URL,
                    params={
                        "action": "query",
                        "prop": "pageprops",
                        "ppprop": "wikibase_item|disambiguation",
                        "titles": "|".join(variants),
                        "redirects": "1",
                        "format": "json",
                        "formatversion": "2",
                    },
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                pages = ((response.json().get("query") or {}).get("pages") or [])
                candidates = [
                    page for page in pages
                    if not page.get("missing")
                    and "disambiguation" not in (page.get("pageprops") or {})
                    and (page.get("pageprops") or {}).get("wikibase_item")
                ]
                by_qid = {
                    str(page["pageprops"]["wikibase_item"]): page
                    for page in candidates
                }
                # Several close title variants may be redirects/aliases of the
                # same entity.  That is agreement, not ambiguity.
                if len(by_qid) == 1:
                    qid, page = next(iter(by_qid.items()))
                    label = str(page.get("title") or surface)
                    self.cache.set(surface, qid)
                    return LinkResult(
                        mention=surface, selected_qid=qid, selected_label=label,
                        score=1.0, margin=1.0,
                    )
                reason = "no exact non-disambiguation Wikipedia title"
                if len(by_qid) > 1:
                    reason = "Wikipedia title variants resolved to different QIDs"
                return LinkResult(mention=surface, abstained=True, abstain_reason=reason)
            except (requests.RequestException, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.request_delay * attempt)
        return LinkResult(mention=surface, abstained=True, abstain_reason=last_error or "title lookup failed")
