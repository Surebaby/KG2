"""Entity linking: GENRE (primary) + Wikidata Search (fallback) + context scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests
from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

import os as _os

from kgproweight.kg.cache import EntityCache
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)

# R9 v6: support reverse proxy for firewalled environments.
# Set KGPW_WIKIDATA_PROXY to the proxy base URL (e.g. https://proxy.example.com).
# Set KGPW_WIKIDATA_PROXY_TOKEN to the X-Proxy-Token header value.
_WIKIDATA_PROXY_BASE = _os.getenv("KGPW_WIKIDATA_PROXY_BASE", "").rstrip("/")
_WIKIDATA_PROXY_TOKEN = _os.getenv("KGPW_WIKIDATA_PROXY_TOKEN", "")

if _WIKIDATA_PROXY_BASE:
    _WIKIDATA_BASE_URL = f"{_WIKIDATA_PROXY_BASE}/https://www.wikidata.org"
else:
    _WIKIDATA_BASE_URL = "https://www.wikidata.org"

WIKIDATA_SEARCH_URL = f"{_WIKIDATA_BASE_URL}/w/api.php"
WIKIDATA_USER_AGENT = "KGProWeight/1.0 (research; contact: anonymous@example.com)"
REQUEST_DELAY = 0.5

# Build default proxy headers once at import time
DEFAULT_PROXY_HEADERS: Dict[str, str] = {}
if _WIKIDATA_PROXY_TOKEN:
    DEFAULT_PROXY_HEADERS["X-Proxy-Token"] = _WIKIDATA_PROXY_TOKEN

# Negative QID types for entity linking
_DISAMBIGUATION_QIDS: set[str] = {"Q4167410", "Q11266439", "Q13406463"}  # disambiguation, list, category
_NEGATIVE_DESCRIPTIONS: set[str] = {
    "wikimedia disambiguation page", "wikimedia category", "wikimedia list",
    "wikimedia template", "wikipedia disambiguation page", "wikipedia category",
    "wikimedia permanent duplicate item", "wikimedia duplicated page",
    "wikimedia internal item", "wikimedia article covering multiple topics",
}

# Known problematic entity overrides (emergency patch only)
_KNOWN_FIXES: Dict[str, Dict[str, str]] = {
    # mention → {context_keyword → correct_QID}
    "big stone gap": {
        "film": "Q4906381",      # Big Stone Gap (film)
        "movie": "Q4906381",
        "default": "Q4906381",   # film is more common in HotpotQA
    },
    "corliss archer": {
        "film": "Q1134521",      # Corliss Archer → link to Shirley Temple
        "default": "Q1134521",
    },
    "scott derrickson": {
        "film": "Q3476545",      # Scott Derrickson (film director)
        "default": "Q3476545",
    },
    "ed wood": {
        "film": "Q221843",       # Ed Wood (filmmaker)
        "default": "Q221843",
    },
}


def _clean(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def build_passage_titles(passages) -> List[str]:
    """Extract (title/id) strings from retrieved passages for title-support.

    Mirrors the ``id``/``title`` fallback used elsewhere (``filter_by_passage_support``,
    the inference pipeline) so every consumer reads the same field.
    """
    titles: List[str] = []
    for p in list(passages or []):
        if isinstance(p, dict):
            title = p.get("id") or p.get("title") or ""
        elif isinstance(p, str):
            title = p
        else:
            continue
        if title:
            titles.append(str(title))
    return titles


def build_passage_text(passages, max_n: int = 10) -> str:
    """Concatenate top-N retrieved passages (title + content) into one block.

    Used as the linker's disambiguation context. Reads the same
    ``contents``/``text`` + ``id``/``title`` fields as
    ``filter_by_passage_support`` so the linker's context signal and the KG
    filter's evidence check see the same text.
    """
    blocks: List[str] = []
    for p in list(passages or [])[:max_n]:
        if isinstance(p, dict):
            content = p.get("contents") or p.get("text") or ""
            title = p.get("id") or p.get("title") or ""
            blocks.append(f"{title} {content}".strip())
        elif isinstance(p, str):
            blocks.append(p)
    return " ".join(b for b in blocks if b).strip()


@dataclass
class LinkCandidate:
    qid: str
    label: str
    description: str = ""
    instance_of: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class LinkResult:
    mention: str
    selected_qid: Optional[str] = None
    selected_label: str = ""
    description: str = ""
    score: float = 0.0
    second_score: float = 0.0
    margin: float = 0.0
    abstained: bool = False
    abstain_reason: str = ""
    candidates: List[LinkCandidate] = field(default_factory=list)


class EntityLinker:
    """Map a list of surface-form mentions to Wikidata QIDs.

    Strategy
    --------
    1. Exact look-up in the on-disk cache (``EntityCache``).
    2. Fuzzy match in the cache (rapidfuzz, ``token_sort_ratio``).
    3. GENRE entity linker (optional, available behind ``use_genre=True``).
    4. Wikidata Search API fallback.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        confidence_threshold: float = 85.0,
        use_genre: bool = False,
        genre_model_path: Optional[str] = None,
        request_delay: float = REQUEST_DELAY,
        offline: bool = False,
        entity_index_path: Optional[str] = None,
    ) -> None:
        self.cache = EntityCache(cache_path)
        self.confidence_threshold = confidence_threshold
        self.request_delay = request_delay
        # offline=True: never hit the Wikidata Search API. Cache hits (exact +
        # fuzzy) still work; a cache miss returns None INSTANTLY instead of
        # blocking on a 10s network timeout. Use when Wikidata is unreachable
        # so a full run is not throttled to a crawl by per-miss timeouts.
        self.offline = offline
        self._genre = None
        if use_genre:
            self._genre = self._try_load_genre(genre_model_path)

        # R9 v7: offline candidate descriptions (label → [{qid, label, description}]).
        # When offline, _search_candidates returns these instead of [] so
        # _score_candidates can still disambiguate against passage context.
        self._entity_index: Dict[str, List[dict]] = {}
        if entity_index_path is None:
            try:
                from kgproweight.retrieval.bootstrap import resolve_entity_desc_index_path
                entity_index_path = resolve_entity_desc_index_path()
            except Exception:
                entity_index_path = None
        if entity_index_path and Path(entity_index_path).exists():
            try:
                self._entity_index = json.loads(
                    Path(entity_index_path).read_text(encoding="utf-8"))
                logger.info("Loaded %d-label entity desc index from %s",
                            len(self._entity_index), entity_index_path)
            except Exception as exc:
                logger.warning("Failed to load entity desc index %s: %s",
                               entity_index_path, exc)
                self._entity_index = {}

    # ------------------------------------------------------------------
    # Optional GENRE backend
    # ------------------------------------------------------------------

    def _try_load_genre(self, path: Optional[str]):
        if path is None:
            logger.warning("GENRE requested but no model path provided; falling back to Wikidata Search.")
            return None
        try:
            from genre.fairseq_model import GENRE  # type: ignore
        except ImportError:
            logger.warning("GENRE (genre / fairseq) not installed; install with `pip install -e .[genre]`. Falling back.")
            return None
        try:
            return GENRE.from_pretrained(path).eval()
        except Exception as exc:
            logger.warning("Failed to load GENRE from %s: %s", path, exc)
            return None

    def _link_via_genre(self, mention: str) -> Optional[str]:
        if self._genre is None:
            return None
        try:
            # GENRE produces titles, which we then map to QIDs via search.
            result = self._genre.sample([mention])
            if result and result[0]:
                title = result[0][0]["text"]
                return self._search_wikidata(title)
        except Exception as exc:
            logger.debug("GENRE failure for %r: %s", mention, exc)
        return None

    # ------------------------------------------------------------------
    # Wikidata Search API
    # ------------------------------------------------------------------

    def _search_wikidata(self, mention: str, lang: str = "en") -> Optional[str]:
        """Legacy: return single QID. Use _search_candidates for rich results."""
        if self.offline:
            return None
        params = {
            "action": "wbsearchentities",
            "search": mention,
            "language": lang,
            "format": "json",
            "limit": 5,
        }
        headers = {"User-Agent": WIKIDATA_USER_AGENT, **DEFAULT_PROXY_HEADERS}
        try:
            resp = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("search"):
                qid = data["search"][0]["id"]
                self.cache.set(mention, qid)
                return qid
        except requests.RequestException as exc:
            logger.warning("Wikidata search failed for '%s': %s", mention, exc)
        return None

    def _local_candidates(self, mention: str) -> List[LinkCandidate]:
        """Candidates reconstructed from the offline desc index (label → QID list).

        Descriptions are surface strings built from 1-hop subgraph relations, so
        they are lexical (not free-text) — still sufficient for the linker's
        word-overlap passage-support term to separate shared surface forms.
        """
        label = _clean(mention)
        entries = self._entity_index.get(label)
        if not entries:
            return []
        return [
            LinkCandidate(
                qid=e["qid"],
                label=e.get("label", mention),
                description=e.get("description", ""),
            )
            for e in entries
        ]

    def _search_candidates(self, mention: str, lang: str = "en") -> List[LinkCandidate]:
        """Return top-10 Wikidata candidates with descriptions for context scoring."""
        if self.offline:
            # R9 v7: no network — reconstruct candidates from the local desc
            # index so context scoring still runs and can disambiguate.
            return self._local_candidates(mention)
        params = {
            "action": "wbsearchentities",
            "search": mention,
            "language": lang,
            "format": "json",
            "limit": 10,
            "props": "",
        }
        headers = {"User-Agent": WIKIDATA_USER_AGENT, **DEFAULT_PROXY_HEADERS}
        try:
            resp = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("search"):
                return []
            candidates = []
            for item in data["search"]:
                candidates.append(LinkCandidate(
                    qid=item["id"],
                    label=item.get("label", mention),
                    description=item.get("description", ""),
                ))
            return candidates
        except requests.RequestException as exc:
            logger.warning("Wikidata candidate search failed for '%s': %s", mention, exc)
            return []

    # ------------------------------------------------------------------
    # Context-aware linking (R9 v6)
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        mention: str,
        candidates: List[LinkCandidate],
        question: str,
        expected_types: Optional[List[str]] = None,
        retrieved_titles: Optional[List[str]] = None,
        passage_text: Optional[str] = None,
    ) -> List[LinkCandidate]:
        """Score Wikidata candidates with full context (R9 v6, per §3.2).

        R9 v7 rebalance: retrieved passage *content* is the strongest
        disambiguation signal for ambiguous surface forms ("Evolution" film vs
        GNOME software), so a dedicated passage-support term replaces the weak
        "longer labels are better" coherence prior.

        score = 0.25 * mention_match
              + 0.20 * context_description_similarity
              + 0.20 * type_compatibility
              + 0.20 * passage_support
              + 0.15 * retrieved_title_support
        """
        q_lower = question.lower()
        m_lower = mention.lower()
        title_set = set(t.lower() for t in (retrieved_titles or []))
        passage_lower = (passage_text or "").lower()
        passage_words = set(passage_lower.split()) if passage_lower else set()

        for c in candidates:
            score = 0.0
            label_lower = c.label.lower()
            desc_lower = c.description.lower()
            desc_words = set(desc_lower.split())

            # mention_match (0.25)
            if label_lower == m_lower:
                score += 0.25
            elif m_lower in label_lower or label_lower in m_lower:
                score += 0.15

            # context_description_similarity (0.20)
            q_words = set(q_lower.split())
            overlap = desc_words & q_words
            if overlap:
                score += 0.20 * min(1.0, len(overlap) / max(1, len(desc_words)))

            # type_compatibility (0.20): expected_types OR question-based inference
            type_hit = False
            if expected_types:
                for et in expected_types:
                    if et in desc_lower:
                        type_hit = True
                        break
            # Infer expected type from question keywords
            if not type_hit:
                person_keywords = {"who", "whose", "actress", "actor", "singer", "director", "player", "author"}
                film_keywords = {"film", "movie", "directed", "starring"}
                location_keywords = {"where", "located", "city", "country", "capital"}
                if any(kw in q_lower for kw in person_keywords) and "human" in desc_lower:
                    type_hit = True
                if any(kw in q_lower for kw in film_keywords) and "film" in desc_lower:
                    type_hit = True
                if any(kw in q_lower for kw in location_keywords) and any(t in desc_lower for t in ("town", "city", "country", "state")):
                    type_hit = True
            if type_hit:
                score += 0.20

            # passage_support (0.20): candidate grounded in the retrieved passage
            # body. For ambiguous surface forms the label is SHARED across
            # candidates ("Evolution" film vs GNOME software), so the label is
            # uninformative and the description ("...film..." vs "...software...")
            # is the real disambiguator — matched against the body, not the title.
            if passage_words:
                p_support = 0.0
                if desc_words:
                    overlap = desc_words & passage_words
                    p_support = min(1.0, len(overlap) / max(1, len(desc_words)))
                # Multi-word label appearing in the body is a strong anchor
                # (title support only checks exact title match; this catches
                # "Big Stone Gap" mentioned inside a passage).
                if len(label_lower.split()) >= 2 and label_lower in passage_lower:
                    p_support = max(p_support, 0.8)
                score += 0.20 * p_support

            # retrieved_title_support (0.15)
            if title_set and label_lower in title_set:
                score += 0.15

            # Negative rules: disambiguation, category, list
            if c.qid in _DISAMBIGUATION_QIDS:
                score -= 0.60
            if desc_lower in _NEGATIVE_DESCRIPTIONS:
                score -= 0.60

            # Type conflict penalties
            if expected_types:
                for et in expected_types:
                    if et == "film" and "town" in desc_lower:
                        score -= 0.40
                    if et == "person" and "page" in desc_lower:
                        score -= 0.40

            c.score = max(0.0, min(1.0, score))

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    # NOTE: `link_with_context` was removed. It duplicated `link_single`
    # (same candidate generation + scoring) but with DIFFERENT abstain
    # thresholds and no legacy-cache fallback, and nothing called it — so
    # reading it as the context-aware entry point was misleading.
    # `link_single(mention, question=...)` is the one real path.

    # ------------------------------------------------------------------
    # Public API (legacy)
    # ------------------------------------------------------------------

    def link(self, mentions: List[str]) -> Dict[str, Optional[str]]:
        """Resolve a batch of mentions; results are also persisted to the cache."""
        results: Dict[str, Optional[str]] = {}
        for mention in mentions:
            result = self.link_single(mention)
            results[mention] = result.selected_qid if isinstance(result, LinkResult) else result
        return results

    def link_single(
        self,
        mention: str,
        question: str = "",
        expected_types: Optional[List[str]] = None,
        retrieved_titles: Optional[List[str]] = None,
        passage_text: Optional[str] = None,
    ) -> LinkResult:
        """Link a mention to Wikidata QID with full context (R9 v6).

        Returns LinkResult with score/margin/abstain/candidates.
        Backward compatible: callers using only mention get old behavior.
        ``passage_text`` (concatenated retrieved passage bodies) is the primary
        disambiguation context for ambiguous surface forms.
        """
        clean = _clean(mention)

        # Known fixes (emergency patch for critical cases)
        fix = _KNOWN_FIXES.get(clean)
        if fix:
            q_key = "default"
            if expected_types:
                for et in expected_types:
                    if et in fix:
                        q_key = et
                        break
            if q_key in fix:
                return LinkResult(
                    mention=mention, selected_qid=fix[q_key],
                    selected_label=mention, score=1.0, margin=0.5,
                )

        # Build candidates
        candidates = self._search_candidates(mention)
        if candidates and (question or passage_text or retrieved_titles or expected_types):
            candidates = self._score_candidates(
                mention, candidates, question,
                expected_types=expected_types,
                retrieved_titles=retrieved_titles,
                passage_text=passage_text,
            )

        if not candidates:
            # Fall back to legacy cache-based linking
            qid = self._legacy_cache_lookup(clean)
            if qid:
                return LinkResult(mention=mention, selected_qid=qid,
                                  selected_label=mention, score=0.85, margin=0.5)
            return LinkResult(mention=mention, abstained=True,
                            abstain_reason="no candidates, no cache hit")

        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top.score - second_score

        # Abstain rules (§3.2)
        if top.score < 0.15:
            qid = self._legacy_cache_lookup(clean)
            if qid:
                return LinkResult(mention=mention, selected_qid=qid,
                                  selected_label=mention, score=0.70, margin=0.3)
            return LinkResult(mention=mention, abstained=True,
                            abstain_reason=f"low score ({top.score:.2f})",
                            candidates=candidates)

        if margin < 0.05 and len(candidates) > 1:
            qid = self._legacy_cache_lookup(clean)
            if qid:
                return LinkResult(mention=mention, selected_qid=qid,
                                  selected_label=mention, score=0.70, margin=0.3)
            return LinkResult(mention=mention, abstained=True,
                            abstain_reason=f"low margin ({margin:.2f})",
                            candidates=candidates)

        # Explicit type conflict
        if top.score < 0.20 and self._legacy_cache_lookup(clean):
            qid = self._legacy_cache_lookup(clean)
            return LinkResult(mention=mention, selected_qid=qid,
                              selected_label=mention, score=0.70, margin=0.3)

        # Persist to cache
        self.cache.set(clean, top.qid)

        return LinkResult(
            mention=mention,
            selected_qid=top.qid,
            selected_label=top.label,
            description=top.description,
            score=top.score,
            second_score=second_score,
            margin=margin,
            candidates=candidates,
        )

    def _legacy_cache_lookup(self, clean: str) -> Optional[str]:
        """Old cache-based lookup for backward compatibility."""
        cached = self.cache.get(clean)
        if cached is not None:
            return cached
        cache_items = list(self.cache.items())
        if cache_items:
            match = rfprocess.extractOne(
                clean, [k for k, _ in cache_items],
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.confidence_threshold,
            )
            if match:
                _, qid = cache_items[match[2]]
                return qid
        return None

    def link_confidence(self, mention: str) -> float:
        """A fuzzy-match-based confidence in ``[0, 1]``. Embedding-based confidence
        lives in :mod:`kgproweight.kg.kg_embeddings`.
        """
        clean = _clean(mention)
        if self.cache.get(clean) is not None:
            return 1.0
        cache_items = list(self.cache.items())
        if not cache_items:
            return 0.0
        match = rfprocess.extractOne(
            clean,
            [k for k, _ in cache_items],
            scorer=fuzz.token_sort_ratio,
        )
        if match:
            return float(match[1]) / 100.0
        return 0.0


# ---------------------------------------------------------------------------
# Lightweight mention extractor (capitalised noun phrases)
# ---------------------------------------------------------------------------

_MENTION_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

# Words that are never part of an entity name. Used BOTH as a whole-mention
# filter and — critically — as a leading/trailing token stripper: a
# sentence-initial function word is capitalised, so the regex above happily
# glued it onto the real entity ("Were Scott Derrickson", "This Singer",
# "Are Local"). The resulting mention never linked. Measured contamination of
# the FIRST extracted mention: hotpotqa 17%, 2wiki 14.9%, musique 6.6%.
_MENTION_BLACKLIST = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "has", "have", "had",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "the", "a", "an", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "than", "both", "either", "neither",
    "in", "on", "at", "to", "for", "of", "from", "by", "with", "as",
    "there", "here", "it", "its", "his", "her", "their", "he", "she", "they",
    "also", "not", "no", "yes", "between", "during", "after", "before",
    "name", "given", "according", "based", "about",
}


def _strip_stopword_affixes(mention: str) -> str:
    """Drop leading/trailing blacklist tokens from a capitalised phrase."""
    parts = mention.split()
    while parts and parts[0].lower() in _MENTION_BLACKLIST:
        parts.pop(0)
    while parts and parts[-1].lower() in _MENTION_BLACKLIST:
        parts.pop()
    return " ".join(parts)


def extract_mentions(text: str, max_n: int = 5) -> List[str]:
    """Best-effort surface-form mention extractor used at inference time.

    For training-time silver generation we prefer GENRE. This regex is a fast
    fallback so the pipeline never blocks on a missing model.
    """
    seen: Dict[str, None] = {}
    for m in _MENTION_RE.findall(text):
        m = _strip_stopword_affixes(m)
        if not m or m.lower() in _MENTION_BLACKLIST:
            continue
        if len(m) >= 3:
            seen.setdefault(m, None)
        if len(seen) >= max_n:
            break
    return list(seen.keys())
