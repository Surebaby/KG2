"""Targeted Wikidata property retrieval for per-question proof KGs.

Unlike the legacy neighbourhood retriever, this client requests only the
``(QID, PID)`` pairs named by a query plan.  Successful responses, including
confirmed missing properties, are persisted in an isolated append-only cache.
This prevents a broad neighbourhood budget from crowding out the one relation
needed by a question and preserves literal values such as dates and numbers.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import requests

from kgproweight.kg.cache import SubgraphCache
from kgproweight.kg.entity_linker import (
    DEFAULT_PROXY_HEADERS,
    WIKIDATA_SEARCH_URL,
    WIKIDATA_USER_AGENT,
)
from kgproweight.kg.kg_filter import _RELATION_LABEL_TO_PID


Triple = Tuple[str, str, str]
PROPERTY_CACHE_VERSION = "wikidata-property-v1"
PROPERTY_EDGE_CACHE_VERSION = "wikidata-property-edge-v1"
_PID_TO_RELATION = {pid: label for label, pid in _RELATION_LABEL_TO_PID.items()}
_EDGE_LOCK = threading.Lock()


def _clean_ids(values: Iterable[str], prefix: str) -> List[str]:
    pattern = re.compile(rf"^{prefix}\d+$")
    return sorted({str(value).strip().upper() for value in values if pattern.match(str(value).strip().upper())})


def _time_value(value: Mapping[str, Any]) -> str:
    raw = str(value.get("time") or "")
    match = re.match(r"^([+-]?\d+)-(\d\d)-(\d\d)T", raw)
    if not match:
        return raw
    year, month, day = match.groups()
    year = year.lstrip("+")
    precision = int(value.get("precision") or 0)
    if precision <= 9:
        return year
    if precision == 10:
        return f"{year}-{month}"
    return f"{year}-{month}-{day}"


def _literal_value(datavalue: Mapping[str, Any]) -> str | None:
    kind = str(datavalue.get("type") or "")
    value = datavalue.get("value")
    if kind == "time" and isinstance(value, Mapping):
        return _time_value(value)
    if kind == "quantity" and isinstance(value, Mapping):
        amount = str(value.get("amount") or "")
        return amount.lstrip("+") or None
    if kind == "monolingualtext" and isinstance(value, Mapping):
        return str(value.get("text") or "") or None
    if kind in {"string", "external-id", "url", "commonsMedia"}:
        return str(value or "") or None
    if kind == "globecoordinate" and isinstance(value, Mapping):
        latitude, longitude = value.get("latitude"), value.get("longitude")
        if latitude is not None and longitude is not None:
            return f"{latitude},{longitude}"
    return None


class WikidataPropertyRetriever:
    """Fetch exact properties with literal-safe parsing and isolated caching."""

    def __init__(
        self,
        *,
        cache_path: str | Path,
        offline: bool = False,
        timeout: float = 20.0,
        request_delay: float = 0.5,
        max_retries: int = 3,
        edge_cache_path: str | Path | None = None,
    ) -> None:
        self.cache = SubgraphCache(cache_path)
        self.offline = bool(offline)
        self.timeout = float(timeout)
        self.request_delay = float(request_delay)
        self.max_retries = max(1, int(max_retries))
        self.edge_cache_path = Path(edge_cache_path or f"{cache_path}.edges.jsonl")
        self._edge_cache: Dict[str, List[Dict[str, str | None]]] = {}
        if self.edge_cache_path.exists():
            with self.edge_cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        if row.get("schema_version") == PROPERTY_EDGE_CACHE_VERSION:
                            self._edge_cache[str(row["key"])] = list(row.get("edges") or [])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue

    @staticmethod
    def cache_key(qid: str, pid: str) -> str:
        return f"{PROPERTY_CACHE_VERSION}::{qid.upper()}::{pid.upper()}"

    @staticmethod
    def edge_cache_key(qid: str, pid: str) -> str:
        return f"{PROPERTY_EDGE_CACHE_VERSION}::{qid.upper()}::{pid.upper()}"

    def _set_edge_cache(self, qid: str, pid: str, edges: List[Dict[str, str | None]]) -> None:
        key = self.edge_cache_key(qid, pid)
        self._edge_cache[key] = edges
        self.edge_cache_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"schema_version": PROPERTY_EDGE_CACHE_VERSION, "key": key, "edges": edges}
        with _EDGE_LOCK, self.edge_cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def fetch_properties(self, qid: str, pids: Sequence[str]) -> List[Triple]:
        clean_qids = _clean_ids([qid], "Q")
        clean_pids = _clean_ids(pids, "P")
        if len(clean_qids) != 1 or not clean_pids:
            return []
        clean_qid = clean_qids[0]
        result: List[Triple] = []
        missing: List[str] = []
        for pid in clean_pids:
            cached = self.cache.get(self.cache_key(clean_qid, pid))
            if cached is None:
                missing.append(pid)
            else:
                result.extend(cached)
        if missing and not self.offline:
            fetched = self._fetch_remote(clean_qid, missing)
            for pid in missing:
                triples = fetched.get(pid, [])
                # A successful entity response proves that an empty PID is
                # absent; cache it so repeated questions do not re-query it.
                self.cache.set(self.cache_key(clean_qid, pid), triples)
                result.extend(triples)
            if self.request_delay:
                time.sleep(self.request_delay)
        return list(dict.fromkeys(result))

    def fetch_edges(self, qid: str, pids: Sequence[str]) -> List[Dict[str, str | None]]:
        """Return exact property edges while retaining entity-valued tail QIDs."""
        clean_qids = _clean_ids([qid], "Q")
        clean_pids = _clean_ids(pids, "P")
        if len(clean_qids) != 1 or not clean_pids:
            return []
        clean_qid = clean_qids[0]
        result: List[Dict[str, str | None]] = []
        missing: List[str] = []
        for pid in clean_pids:
            cached = self._edge_cache.get(self.edge_cache_key(clean_qid, pid))
            if cached is None:
                missing.append(pid)
            else:
                result.extend(cached)
        if missing and not self.offline:
            fetched = self._fetch_remote_edges(clean_qid, missing)
            for pid in missing:
                edges = fetched.get(pid, [])
                self._set_edge_cache(clean_qid, pid, edges)
                triples = [
                    (str(edge["head_label"]), str(edge["relation"]), str(edge["tail_value"]))
                    for edge in edges
                ]
                # Keep the legacy triple projection reusable as well.
                self.cache.set(self.cache_key(clean_qid, pid), triples)
                result.extend(edges)
            if self.request_delay:
                time.sleep(self.request_delay)
        elif missing:
            # Backward-compatible projection for old triple-only caches.  It
            # cannot recover tail QIDs, but still returns the known evidence.
            for pid in missing:
                triples = self.cache.get(self.cache_key(clean_qid, pid)) or []
                result.extend(
                    {
                        "head_qid": clean_qid,
                        "head_label": triple[0],
                        "pid": pid,
                        "relation": triple[1],
                        "tail_qid": None,
                        "tail_value": triple[2],
                    }
                    for triple in triples
                )
        return result

    def _request(self, params: Mapping[str, str]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    WIKIDATA_SEARCH_URL,
                    params=dict(params),
                    headers={"User-Agent": WIKIDATA_USER_AGENT, **DEFAULT_PROXY_HEADERS},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or "entities" not in payload:
                    raise ValueError("invalid wbgetentities response")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.request_delay * attempt)
        assert last_error is not None
        raise last_error

    def _labels(self, qids: Sequence[str]) -> Dict[str, str]:
        clean = _clean_ids(qids, "Q")
        if not clean:
            return {}
        payload = self._request(
            {
                "action": "wbgetentities",
                "ids": "|".join(clean),
                "props": "labels",
                "languages": "en",
                "languagefallback": "1",
                "format": "json",
            }
        )
        labels: Dict[str, str] = {}
        for qid, entity in (payload.get("entities") or {}).items():
            label = ((entity.get("labels") or {}).get("en") or {}).get("value")
            labels[str(qid)] = str(label or qid)
        return labels

    def _fetch_remote_edges(
        self, qid: str, pids: Sequence[str]
    ) -> Dict[str, List[Dict[str, str | None]]]:
        payload = self._request(
            {
                "action": "wbgetentities",
                "ids": qid,
                "props": "labels|claims",
                "languages": "en",
                "languagefallback": "1",
                "format": "json",
            }
        )
        entity = (payload.get("entities") or {}).get(qid) or {}
        if entity.get("missing") is not None:
            raise ValueError(f"Wikidata entity does not exist: {qid}")
        head = (((entity.get("labels") or {}).get("en") or {}).get("value") or qid)
        parsed: Dict[str, List[Tuple[str, str, str | None, str | None]]] = {
            pid: [] for pid in pids
        }
        tail_qids: List[str] = []
        claims = entity.get("claims") or {}
        for pid in pids:
            for claim in claims.get(pid) or []:
                if claim.get("rank") == "deprecated":
                    continue
                snak = claim.get("mainsnak") or {}
                if snak.get("snaktype") != "value":
                    continue
                datavalue = snak.get("datavalue") or {}
                raw_value = datavalue.get("value")
                if datavalue.get("type") == "wikibase-entityid" and isinstance(raw_value, Mapping):
                    tail_qid = str(raw_value.get("id") or "")
                    if tail_qid:
                        tail_qids.append(tail_qid)
                        parsed[pid].append((str(head), _PID_TO_RELATION.get(pid, pid), None, tail_qid))
                else:
                    literal = _literal_value(datavalue)
                    if literal:
                        parsed[pid].append((str(head), _PID_TO_RELATION.get(pid, pid), literal, None))
        labels = self._labels(tail_qids)
        result: Dict[str, List[Dict[str, str | None]]] = {}
        for pid, values in parsed.items():
            edges: List[Dict[str, str | None]] = []
            for head_label, relation, literal, tail_qid in values:
                tail = literal if literal is not None else labels.get(str(tail_qid), str(tail_qid))
                edge = {
                    "head_qid": qid,
                    "head_label": head_label,
                    "pid": pid,
                    "relation": relation,
                    "tail_qid": tail_qid,
                    "tail_value": tail,
                }
                if edge not in edges:
                    edges.append(edge)
            result[pid] = edges
        return result

    def _fetch_remote(self, qid: str, pids: Sequence[str]) -> Dict[str, List[Triple]]:
        edges = self._fetch_remote_edges(qid, pids)
        return {
            pid: [
                (str(edge["head_label"]), str(edge["relation"]), str(edge["tail_value"]))
                for edge in values
            ]
            for pid, values in edges.items()
        }
