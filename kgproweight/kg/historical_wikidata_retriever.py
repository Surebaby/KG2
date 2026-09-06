"""Targeted Wikidata property retrieval at a frozen historical cutoff."""

from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, List, Mapping, Protocol, Sequence

import requests

from kgproweight.kg.entity_linker import (
    DEFAULT_PROXY_HEADERS,
    WIKIDATA_SEARCH_URL,
    WIKIDATA_USER_AGENT,
)
from kgproweight.kg.wikidata_property_retriever import (
    _PID_TO_RELATION,
    _clean_ids,
    _literal_value,
)


HISTORICAL_CACHE_VERSION = "wikidata-historical-entity-revision-1"
_LOCK = threading.Lock()
_MONTH_NAMES = (
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _display_literal_value(datavalue: Mapping[str, Any], literal: str) -> str:
    """Render exact Wikidata dates for an English-language QA prompt.

    The cached revision and ``tail_raw_value`` retain the canonical ISO value.
    This is presentation-only and never consults a question, passage, or label.
    Unsupported/BCE values remain unchanged rather than being guessed.
    """
    if datavalue.get("type") != "time":
        return literal
    raw = datavalue.get("value")
    if not isinstance(raw, Mapping):
        return literal
    precision = int(raw.get("precision") or 0)
    if precision == 10:
        match = re.fullmatch(r"(\d{1,})-(\d{2})", literal)
        if match and 1 <= int(match.group(2)) <= 12:
            return f"{_MONTH_NAMES[int(match.group(2))]} {int(match.group(1))}"
    if precision >= 11:
        match = re.fullmatch(r"(\d{1,})-(\d{2})-(\d{2})", literal)
        if match and 1 <= int(match.group(2)) <= 12 and 1 <= int(match.group(3)) <= 31:
            return (
                f"{int(match.group(3))} {_MONTH_NAMES[int(match.group(2))]} "
                f"{int(match.group(1))}"
            )
    return literal


class _QidLabelResolver(Protocol):
    def label_for_qid(self, qid: str) -> str | None: ...


class HistoricalWikidataPropertyRetriever:
    """Fetch exact properties from the last entity revision before ``cutoff``."""

    def __init__(
        self,
        *,
        cache_path: str | Path,
        cutoff: str,
        offline: bool = False,
        timeout: float = 30.0,
        request_delay: float = 0.2,
        max_retries: int = 3,
        label_resolver: _QidLabelResolver | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cutoff = str(cutoff)
        self.offline = bool(offline)
        self.timeout = float(timeout)
        self.request_delay = float(request_delay)
        self.max_retries = max(1, int(max_retries))
        self.label_resolver = label_resolver
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("schema_version") == HISTORICAL_CACHE_VERSION:
                        self._cache[str(row.get("key"))] = row

    def cache_key(self, qid: str) -> str:
        return f"{HISTORICAL_CACHE_VERSION}::{self.cutoff}::{qid.upper()}"

    def _persist(self, qid: str, entity: Mapping[str, Any] | None, revision: Mapping[str, Any]) -> None:
        key = self.cache_key(qid)
        row = {
            "schema_version": HISTORICAL_CACHE_VERSION,
            "key": key,
            "qid": qid,
            "cutoff": self.cutoff,
            "revision": dict(revision),
            "entity": dict(entity) if entity is not None else None,
        }
        self._cache[key] = row
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_payload(payload: Mapping[str, Any]) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        pages = ((payload.get("query") or {}).get("pages") or [])
        revision = (pages[0].get("revisions") or [None])[0] if pages else None
        if not revision:
            return None, {"missing_before_cutoff": True}
        content = (((revision.get("slots") or {}).get("main") or {}).get("content"))
        if not isinstance(content, str):
            return None, {
                "revid": revision.get("revid"),
                "timestamp": revision.get("timestamp"),
                "missing_content": True,
            }
        entity = json.loads(content)
        return entity, {
            "revid": revision.get("revid"),
            "parentid": revision.get("parentid"),
            "timestamp": revision.get("timestamp"),
        }

    def _request_entity(self, qid: str) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    WIKIDATA_SEARCH_URL,
                    params={
                        "action": "query",
                        "prop": "revisions",
                        "titles": qid,
                        "rvstart": self.cutoff,
                        "rvdir": "older",
                        "rvlimit": "1",
                        "rvprop": "ids|timestamp|content",
                        "rvslots": "main",
                        "format": "json",
                        "formatversion": "2",
                    },
                    headers={"User-Agent": WIKIDATA_USER_AGENT, **DEFAULT_PROXY_HEADERS},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return self._parse_payload(response.json())
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.request_delay * attempt)
        assert last_error is not None
        raise last_error

    def _entity(self, qid: str) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        key = self.cache_key(qid)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.get("entity"), dict(cached.get("revision") or {})
        if self.offline:
            return None, {"offline_cache_miss": True}
        entity, revision = self._request_entity(qid)
        self._persist(qid, entity, revision)
        if self.request_delay:
            time.sleep(self.request_delay)
        return entity, revision

    def _label(self, qid: str) -> str:
        if self.label_resolver is not None:
            label = self.label_resolver.label_for_qid(qid)
            if label:
                return str(label)
        cached = self._cache.get(self.cache_key(qid)) or {}
        entity = cached.get("entity") or {}
        return str((((entity.get("labels") or {}).get("en") or {}).get("value") or qid))

    def fetch_edges(self, qid: str, pids: Sequence[str]) -> List[Dict[str, str | None]]:
        clean_qids = _clean_ids([qid], "Q")
        clean_pids = _clean_ids(pids, "P")
        if len(clean_qids) != 1 or not clean_pids:
            return []
        clean_qid = clean_qids[0]
        entity, revision = self._entity(clean_qid)
        if not entity:
            return []
        return self._edges_from_entity(clean_qid, entity, revision, clean_pids)

    def fetch_all_edges(self, qid: str) -> List[Dict[str, str | None]]:
        """Return all non-deprecated value claims on one exact historical item.

        This is deliberately an exact-entity operation, not neighbourhood
        expansion.  Callers are responsible for filtering the returned claims
        before any edge enters a prompt.  Keeping the full scan here avoids a
        planner-selected PID becoming an artificial coverage ceiling while the
        revision cutoff and source provenance remain identical to
        :meth:`fetch_edges`.
        """
        clean_qids = _clean_ids([qid], "Q")
        if len(clean_qids) != 1:
            return []
        clean_qid = clean_qids[0]
        entity, revision = self._entity(clean_qid)
        if not entity:
            return []
        clean_pids = _clean_ids(sorted((entity.get("claims") or {}).keys()), "P")
        return self._edges_from_entity(clean_qid, entity, revision, clean_pids)

    def _edges_from_entity(
        self,
        clean_qid: str,
        entity: Mapping[str, Any],
        revision: Mapping[str, Any],
        clean_pids: Sequence[str],
    ) -> List[Dict[str, str | None]]:
        head_label = str(((entity.get("labels") or {}).get("en") or {}).get("value") or clean_qid)
        edges: List[Dict[str, str | None]] = []
        for pid in clean_pids:
            for claim in (entity.get("claims") or {}).get(pid) or []:
                if claim.get("rank") == "deprecated":
                    continue
                snak = claim.get("mainsnak") or {}
                if snak.get("snaktype") != "value":
                    continue
                datavalue = snak.get("datavalue") or {}
                raw = datavalue.get("value")
                tail_qid = None
                if datavalue.get("type") == "wikibase-entityid" and isinstance(raw, Mapping):
                    candidate = str(raw.get("id") or "")
                    if re.fullmatch(r"Q[1-9][0-9]*", candidate):
                        tail_qid = candidate
                        tail_value = self._label(candidate)
                    else:
                        continue
                else:
                    literal = _literal_value(datavalue)
                    if not literal:
                        continue
                    tail_value = _display_literal_value(datavalue, literal)
                edge = {
                    "head_qid": clean_qid,
                    "head_label": head_label,
                    "pid": pid,
                    "relation": _PID_TO_RELATION.get(pid, pid),
                    "tail_qid": tail_qid,
                    "tail_value": tail_value,
                    "tail_raw_value": (
                        literal
                        if tail_qid is None and tail_value != literal
                        else None
                    ),
                    "source_revision_id": str(revision.get("revid") or ""),
                    "source_revision_timestamp": str(revision.get("timestamp") or ""),
                    "source_cutoff": self.cutoff,
                }
                if edge not in edges:
                    edges.append(edge)
        return edges
