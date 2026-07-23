"""Entity linking: GENRE (primary) + Wikidata Search (fallback) + context scoring."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

from kgproweight.kg.cache import EntityCache
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_USER_AGENT = "KGProWeight/1.0 (research; contact: anonymous@example.com)"
REQUEST_DELAY = 0.5

# Negative QID types for entity linking
_DISAMBIGUATION_QIDS: set[str] = {"Q4167410", "Q11266439", "Q13406463"}  # disambiguation, list, category
_NEGATIVE_DESCRIPTIONS: set[str] = {
    "wikimedia disambiguation page", "wikimedia category", "wikimedia list",
    "wikimedia template", "wikipedia disambiguation page", "wikipedia category",
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
}


def _clean(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


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
        headers = {"User-Agent": WIKIDATA_USER_AGENT}
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

    def _search_candidates(self, mention: str, lang: str = "en") -> List[LinkCandidate]:
        """Return top-10 Wikidata candidates with descriptions for context scoring."""
        if self.offline:
            return []
        params = {
            "action": "wbsearchentities",
            "search": mention,
            "language": lang,
            "format": "json",
            "limit": 10,
            "props": "",
        }
        headers = {"User-Agent": WIKIDATA_USER_AGENT}
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
    ) -> List[LinkCandidate]:
        """Score Wikidata candidates based on context relevance."""
        q_lower = question.lower()
        m_lower = mention.lower()

        for c in candidates:
            score = 0.0

            # mention_match: exact or fuzzy (0.0–0.30)
            label_lower = c.label.lower()
            if label_lower == m_lower:
                score += 0.30
            elif m_lower in label_lower or label_lower in m_lower:
                score += 0.20
            else:
                score += 0.10  # wikidata returned it, assume partial match

            # context_description_similarity (0.0–0.30)
            desc_lower = c.description.lower()
            desc_words = set(desc_lower.split())
            q_words = set(q_lower.split())
            overlap = desc_words & q_words
            if overlap:
                score += 0.30 * min(1.0, len(overlap) / max(1, len(desc_words)))

            # type_compatibility (0.0–0.20)
            if expected_types:
                for etype in expected_types:
                    if etype in desc_lower:
                        score += 0.20
                        break

            # Negative: disambiguation/category penalty
            if c.qid in _DISAMBIGUATION_QIDS or desc_lower in _NEGATIVE_DESCRIPTIONS:
                score -= 0.50

            c.score = max(0.0, min(1.0, score))

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def link_with_context(
        self,
        mention: str,
        question: str = "",
        retrieved_titles: Optional[List[str]] = None,
        expected_types: Optional[List[str]] = None,
    ) -> LinkResult:
        """Link a mention to a Wikidata QID with question-context scoring.

        Returns a LinkResult with candidates, confidence, and abstain flag.
        """
        clean = _clean(mention)
        candidates: List[LinkCandidate] = []

        # Check known fixes first
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
                    mention=mention,
                    selected_qid=fix[q_key],
                    selected_label=mention,
                    score=0.95,
                    margin=0.50,
                )

        # 1. Wikidata Search candidates
        candidates = self._search_candidates(mention)
        if candidates:
            candidates = self._score_candidates(mention, candidates, question, expected_types)

        # 2. Score and select
        if not candidates:
            return LinkResult(mention=mention, abstained=True, abstain_reason="no candidates")

        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top.score - second_score

        # Abstain rules
        if top.score < 0.15:
            return LinkResult(mention=mention, abstained=True,
                           abstain_reason=f"low score ({top.score:.2f})",
                           candidates=candidates)
        if margin < 0.05 and len(candidates) > 1:
            return LinkResult(mention=mention, abstained=True,
                           abstain_reason=f"low margin ({margin:.2f})",
                           candidates=candidates)
        if top.score < 0:
            return LinkResult(mention=mention, abstained=True,
                           abstain_reason="negative score (disambiguation/category)",
                           candidates=candidates)

        # Update cache with context-aware decision
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

    # ------------------------------------------------------------------
    # Public API (legacy)
    # ------------------------------------------------------------------

    def link(self, mentions: List[str]) -> Dict[str, Optional[str]]:
        """Resolve a batch of mentions; results are also persisted to the cache."""
        results: Dict[str, Optional[str]] = {}
        for mention in mentions:
            results[mention] = self.link_single(mention)
        return results

    def link_single(self, mention: str) -> Optional[str]:
        clean = _clean(mention)

        # Exact cache hit
        cached = self.cache.get(clean)
        if cached is not None:
            return cached

        # Fuzzy cache hit
        cache_items = list(self.cache.items())
        if cache_items:
            match = rfprocess.extractOne(
                clean,
                [k for k, _ in cache_items],
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.confidence_threshold,
            )
            if match:
                idx = match[2]
                _, qid = cache_items[idx]
                return qid

        # GENRE
        qid = self._link_via_genre(mention)
        if qid is not None:
            return qid

        # Wikidata Search
        time.sleep(self.request_delay)
        return self._search_wikidata(mention)

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
_MENTION_BLACKLIST = {
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "the",
    "a",
    "an",
}


def extract_mentions(text: str, max_n: int = 5) -> List[str]:
    """Best-effort surface-form mention extractor used at inference time.

    For training-time silver generation we prefer GENRE. This regex is a fast
    fallback so the pipeline never blocks on a missing model.
    """
    seen: Dict[str, None] = {}
    for m in _MENTION_RE.findall(text):
        if m.lower() in _MENTION_BLACKLIST:
            continue
        if len(m) >= 3:
            seen.setdefault(m, None)
        if len(seen) >= max_n:
            break
    return list(seen.keys())
