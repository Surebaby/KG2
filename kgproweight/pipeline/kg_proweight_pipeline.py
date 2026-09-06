"""KG-ProWeight inference pipeline.

Uses the canonical SFT/inference schema from :mod:`kgproweight.data.prompts`
(``[Step N] ... [Final Answer]``) instead of FlashRAG's
:class:`ReasoningPipeline` ``<answer>`` protocol.

Workflow per sample:
  1. Hybrid RRF retrieval (top-K passages, configured in FlashRAG config).
  2. Optional Wikidata 2-hop subgraph injection (honours D_dropout overrides).
  3. Single-pass generation via :func:`build_inference_messages`.
  4. Answer extraction via :func:`extract_final_answer`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from kgproweight.data.parsers import parse_steps
from kgproweight.data.prompts import build_inference_messages
from kgproweight.eval.pred_processing import extract_kg_proweight_answer
from kgproweight.kg.entity_linker import (
    EntityLinker,
    build_passage_text,
    build_passage_titles,
    extract_mentions,
)
from kgproweight.kg.kg_filter import filter_and_rank_triples
from kgproweight.kg.wikidata_retriever import _QA_RELATION_FILTER, WikidataSubgraphRetriever
from kgproweight.retrieval.hybrid import DEFAULT_TOPK
from kgproweight.reward.alpha_gate import AlphaGate, compute_features
from kgproweight.reward.citation_features import citation_features
from kgproweight.utils.paths import index_dir
from kgproweight.reward.prm_annotator import PRMAnnotator
from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import get_logger

setup_flashrag()

from flashrag.pipeline.pipeline import BasicPipeline  # noqa: E402
from flashrag.utils import get_generator, get_retriever  # noqa: E402

logger = get_logger(__name__)


class KGProWeightPipeline(BasicPipeline):
    """Single-pass inference with KG context + α telemetry."""

    def __init__(
        self,
        config,
        alpha_gate_path: Optional[str] = None,
        entity_cache_path: Optional[str] = None,
        kg_cache_dir: Optional[str] = None,
        record_alpha: bool = True,
        inject_kg: bool = True,
        max_kg_triples: int = 12,
        max_mentions: int = 5,
        retrieval_topk: Optional[int] = None,
        rerank_topk: int = 0,  # 0 = disabled
        rerank_method: str = "cross-encoder",
        cross_encoder_model: str = "models/bge-reranker-v2-m3",
        alpha_bias_correction: Optional[float] = None,
        kg_supply_mode: str = "legacy",
        question_kg_records_path: Optional[str] = None,
        generator=None,
        retriever=None,
        **kwargs,
    ) -> None:
        super().__init__(config, prompt_template=kwargs.pop("prompt_template", None))
        self.record_alpha = record_alpha
        self.inject_kg = inject_kg
        # 2026-08-23: the default is now 0.0 (NO correction). None → 0.0; a float
        # → that exact additive bias. Pass 0.78 explicitly to reproduce any run
        # from before this date.
        #
        # +0.78 was never a derived quantity: b_trained = -1.7818 and
        # -1.7818 + 0.78 = -1.0018, i.e. it was reverse-engineered to land
        # b_effective on a round -1.0 (the comment below said so outright). Sized
        # against the mechanism it claims -- compensating a f_entropy regime shift
        # Δe through W2 = -0.5833, so needed bias = 0.5833·Δe -- +0.78 requires
        # Δe = 1.337, which is impossible: f_entropy's own ceiling here is ~0.62.
        # The defensible ceiling is +0.363 (taking e_tr = 0); the real post-D1
        # need is +0.129. See retraining_plan.md "P6 后续".
        self.alpha_bias_correction = alpha_bias_correction
        self.max_kg_triples = max_kg_triples
        self.max_mentions = max_mentions
        _cfg_topk = config["retrieval_topk"] if "retrieval_topk" in config else DEFAULT_TOPK
        self.retrieval_topk = retrieval_topk or int(_cfg_topk)
        self.rerank_topk = rerank_topk
        self.rerank_method = rerank_method
        self.cross_encoder_model = cross_encoder_model
        self._alpha_records: List[Dict] = []

        self.generator = generator if generator is not None else get_generator(config)
        self.retriever = retriever if retriever is not None else get_retriever(config)

        # KGPW_KG_OFFLINE=1 makes entity linking + subgraph fetch never touch the
        # network: cache hits still work, misses return empty INSTANTLY (no 10-100s
        # SPARQL/Search timeouts). Required when Wikidata is unreachable (e.g. CN host).
        _kg_offline = os.environ.get("KGPW_KG_OFFLINE", "").lower() in ("1", "true", "yes")
        if _kg_offline:
            logger.info("KG offline mode ON (KGPW_KG_OFFLINE) — cache-only, no network.")
        self.entity_linker = EntityLinker(cache_path=entity_cache_path, offline=_kg_offline)
        # Default to the project KG cache when the caller passes nothing — an
        # unset cache_dir means every offline fetch returns [] (no KG at all).
        _kg_cache_dir = kg_cache_dir or str(Path(index_dir()) / "kg_cache")
        self.kg_retriever = WikidataSubgraphRetriever(
            max_hops=2, max_neighbors=30, cache_dir=_kg_cache_dir,
            offline=_kg_offline, relation_filter=_QA_RELATION_FILTER,
        )
        self.prm_annotator = PRMAnnotator(entity_linker=self.entity_linker, verbose=False)
        # Where each sample's KG came from — logged after run() so a 0% index
        # hit rate (the R9 v6 dev-split gap) is visible instead of silent.
        self._kg_source_counts: Dict[str, int] = {"index": 0, "fallback": 0, "empty": 0}

        # ProofKG-v1 supply (optional, default off). When ``kg_supply_mode`` is
        # "proofkg_v1", the pipeline reads a versioned per-question ProofKG
        # record (keyed by ``dataset::qid``) instead of the legacy question-KG
        # index. The legacy index load below still runs so that falling back to
        # "legacy" reproduces every historical run bit-for-bit.
        self.kg_supply_mode = kg_supply_mode
        self._current_dataset_name: str = ""
        self._proofkg_records: Dict[str, List[Tuple[str, str, str]]] = {}
        self._qpeg_records: Dict[str, Dict] = {}
        self._selective_records: Dict[str, Dict] = {}
        if kg_supply_mode == "qpeg_v1":
            self._load_qpeg_records(question_kg_records_path)
        elif kg_supply_mode == "proofkg_v1":
            self._load_proofkg_records(question_kg_records_path)
        elif kg_supply_mode in ("legacy_plus_proofkg", "legacy_plus_complete_proofkg"):
            self._load_selective_records(question_kg_records_path)
        elif kg_supply_mode != "legacy":
            raise ValueError(f"unknown kg_supply_mode={kg_supply_mode!r}")

        # R9 v6: load pre-built question→KG index (v2 filtered cache) to align
        # inference KG quality with training. Falls back to live Wikidata on miss.
        self._q_kg_index: Dict[str, List[Tuple[str, str, str]]] = {}
        # R9 v6: use pre-built question→KG index. v2 is the original filtered
        # cache (30 triples/q, 3-layer filter). Runtime improvements
        # (max_kg_triples=12, bias correction, hard-delete) are applied.
        _q_kg_path = Path(index_dir()) / "kg_cache" / "question_kg_index_v2.json"
        if not _q_kg_path.exists():
            _q_kg_path = Path(index_dir()) / "kg_cache" / "question_kg_index.json"
        if _q_kg_path.exists():
            import json as _json
            _q_kg_raw = _json.loads(_q_kg_path.read_text(encoding="utf-8"))
            is_v2 = "builder_version" in (_q_kg_raw[0] if _q_kg_raw else {})
            for _entry in _q_kg_raw:
                _q = _entry.get("question", _entry.get("q", ""))
                if is_v2:
                    self._q_kg_index[_q] = [(t["h"], t["r"], t["t"]) for t in _entry["triples"]]
                else:
                    self._q_kg_index[_q] = [tuple(t) for t in _entry["t"]]
            logger.info("Loaded %d question→KG entries from %s (v%s, inference)",
                        len(self._q_kg_index), _q_kg_path.name, "2" if is_v2 else "1")

        self.alpha_gate = AlphaGate()
        if alpha_gate_path and Path(alpha_gate_path).exists():
            self.alpha_gate.load_state_dict(torch.load(alpha_gate_path, map_location="cpu"))
            logger.info("Loaded AlphaGate from %s", alpha_gate_path)
            # HISTORY (R9 v6): a +0.78 bias correction was applied here to
            # compensate f_entropy being hardcoded to 1.0 at inference (logprobs
            # were unavailable), which shifted the α input distribution right of
            # the regime the trained b≈-1.78 was calibrated for.
            #
            # 2026-08-23: DEFAULT CHANGED TO 0.0 (no correction). Three reasons,
            # in decreasing order of how hard they are to argue with:
            #
            # 1. The premise is gone. D1 (commit e6b2198) made inference use REAL
            #    per-token logprobs (return_scores=True below), and the measured
            #    f_entropy is ~0.62, not 1.0. alpha_diagnose.py further shows
            #    f_entropy=1.0 has NO solution in the feasible domain -- it would
            #    require f_confidence = +1.135 > 1. Correcting the bias on top of
            #    D1's feature-level fix double-counts the same shift.
            # 2. Training never applied it. phase3_ppo.py:787-791 loads the gate
            #    raw and calls eval(). So +0.78 made train and inference use two
            #    DIFFERENT α functions -- a train/eval mismatch, not a fix for one.
            # 3. Measured harm: with the correction, α's sd is 0.0217/0.0224/0.0301
            #    (hotpotqa/2wiki/musique); without it, 0.0523/0.0529/0.0676. It
            #    pushes α into sigmoid saturation and suppresses the gate's already
            #    weak discriminative range -- the opposite of what it was for.
            #
            # This does NOT move EM/F1: α is eval-time telemetry and does not enter
            # the answer path. It matters for PPO's r_kg weighting and for not
            # claiming in §3.4 that +0.78 had a derivation.
            _correction = 0.0 if self.alpha_bias_correction is None else self.alpha_bias_correction
            if _correction:
                _b_corrected = self.alpha_gate.b.data + _correction
                logger.info("AlphaGate inference bias correction: %.3f → %.3f",
                            self.alpha_gate.b.item(), _b_corrected.item())
                self.alpha_gate.b.data = _b_corrected
            else:
                logger.info("AlphaGate inference bias correction disabled (b=%.3f)",
                            self.alpha_gate.b.item())
        elif inject_kg:
            logger.warning("No AlphaGate checkpoint provided — using initial weights.")
        self.alpha_gate.eval()

    # ------------------------------------------------------------------
    # KG context construction
    # ------------------------------------------------------------------

    def _get_dropout_kg(self, item) -> Optional[List[Tuple[str, str, str]]]:
        """Return the severed subgraph if this item is from D_dropout."""
        meta = getattr(item, "metadata", None) or {}
        if isinstance(meta, dict):
            dropout = meta.get("dropout")
            if isinstance(dropout, dict):
                mod = dropout.get("modified_kg")
                if isinstance(mod, list) and mod:
                    return [tuple(t) for t in mod if len(t) == 3]
        return None

    def _load_proofkg_records(self, path: Optional[str]) -> None:
        import json as _json
        from kgproweight.kg.question_kg import load_question_kg_index
        if not path:
            raise ValueError("kg_supply_mode=proofkg_v1 requires question_kg_records_path")
        _p = Path(path)
        if not _p.is_file():
            raise ValueError(f"question_kg_records_path not found: {_p}")
        _records = [
            _json.loads(_line)
            for _line in _p.read_text(encoding="utf-8").splitlines()
            if _line.strip()
        ]
        # load_question_kg_index enforces: canonical schema, question_key ==
        # dataset::qid (no auto-join), question hash matches the stored question,
        # triples are non-empty 3-tuples, and no duplicate keys.
        _index = load_question_kg_index(_records)
        self._proofkg_records = {
            _key: [tuple(t) for t in _rec["kg_subgraph"]]
            for _key, _rec in _index.items()
        }
        logger.info("Loaded %d ProofKG-v1 records (kg_supply_mode=%s) from %s",
                    len(self._proofkg_records), self.kg_supply_mode, _p.name)

    def _load_qpeg_records(self, path: Optional[str]) -> None:
        import json as _json
        from kgproweight.kg.qpeg import validate_qpeg_record
        if not path:
            raise ValueError("kg_supply_mode=qpeg_v1 requires question_kg_records_path")
        _p = Path(path)
        if not _p.is_file():
            raise ValueError(f"QPEG records not found: {_p}")
        self._qpeg_records = {}
        for _line in _p.read_text(encoding="utf-8").splitlines():
            if not _line.strip():
                continue
            _record = _json.loads(_line)
            validate_qpeg_record(_record)
            _key = str(_record["question_key"])
            if _key in self._qpeg_records:
                raise ValueError(f"duplicate QPEG record key: {_key}")
            self._qpeg_records[_key] = _record
        logger.info("Loaded %d QPEG-v1 records from %s", len(self._qpeg_records), _p.name)

    def _load_selective_records(self, path: Optional[str]) -> None:
        import json as _json
        from kgproweight.kg.question_kg import question_key
        from kgproweight.kg.selective_proofkg import validate_selective_proofkg_record
        if not path:
            raise ValueError(f"kg_supply_mode={self.kg_supply_mode} requires question_kg_records_path (selective records)")
        _p = Path(path)
        if not _p.is_file():
            raise ValueError(f"selective records not found: {_p}")
        _records = [
            _json.loads(_line)
            for _line in _p.read_text(encoding="utf-8").splitlines()
            if _line.strip()
        ]
        for _rec in _records:
            # fail-fast on a corrupt/mismatched record; never a silent legacy fallback
            validate_selective_proofkg_record(
                _rec, dataset=str(_rec.get("dataset") or ""), qid=str(_rec.get("qid") or "")
            )
        self._selective_records = {}
        for _rec in _records:
            _key = question_key(str(_rec["dataset"]), str(_rec["qid"]))
            if _key in self._selective_records:
                raise ValueError(f"duplicate selective record key: {_key}")
            self._selective_records[_key] = _rec
        logger.info("Loaded %d selective ProofKG records (kg_supply_mode=%s) from %s",
                    len(self._selective_records), self.kg_supply_mode, _p.name)

    def _augment_with_selective(self, legacy, item, arm) -> List[Tuple[str, str, str]]:
        from kgproweight.kg.question_kg import question_key, question_sha256
        from kgproweight.kg.selective_proofkg import (
            merge_legacy_and_proof_edges,
            select_selective_proof_edges,
        )
        _key = question_key(self._current_dataset_name, str(item.id))
        rec = self._selective_records.get(_key)
        if rec is None:
            # join must be 1.0; a missing record is a corrupt asset, not a fallback
            raise ValueError(f"selective ProofKG record missing for {_key} (join < 1.0)")
        if rec.get("question_sha256") != question_sha256(str(item.question)):
            raise ValueError(f"selective record question hash mismatch for {_key}")
        proof_edges = select_selective_proof_edges(rec, arm=arm)
        if not proof_edges:
            # legal record but no trusted edges (or not complete for C): exact legacy
            return list(legacy)
        merged, counters = merge_legacy_and_proof_edges(legacy, proof_edges, cap=self.max_kg_triples)
        self._selective_counters = getattr(self, "_selective_counters", [])
        self._selective_counters.append(counters)
        return merged

    def _enforce_proofkg_join(self, dataset) -> None:
        """Fail-fast identity join for ``kg_supply_mode=proofkg_v1``.

        A partial ProofKG-v1 materialisation must never look like a complete
        experiment: in this mode the pipeline requires every eval item to have a
        versioned record keyed by ``dataset::qid`` *before* any generation runs,
        instead of silently supplying an empty KG on a miss (``_build_kg_context``
        keeps its empty-on-miss branch as a defensive per-item fallback, but the
        run-level gate below makes that branch unreachable for the configured
        question set).
        """
        from kgproweight.kg.question_kg import question_key
        if not self._current_dataset_name:
            raise ValueError(
                "kg_supply_mode=proofkg_v1 requires a known dataset_name to join "
                "records by dataset::qid; got an empty dataset_name."
            )
        qids = list(getattr(dataset, "id", []) or [])
        missing = [
            qid for qid in qids
            if question_key(self._current_dataset_name, str(qid)) not in self._proofkg_records
        ]
        if missing:
            raise ValueError(
                f"kg_supply_mode=proofkg_v1 identity join < 1.0 on "
                f"{self._current_dataset_name!r}: {len(missing)}/{len(qids)} questions "
                f"have no ProofKG-v1 record (first missing qids: {missing[:5]!r}). "
                f"Materialise question_kg_records for this exact question set before "
                f"running; empty-KG-on-miss is not permitted."
            )

    def _enforce_qpeg_join(self, dataset) -> None:
        """Require an identity- and question-hash-exact QPEG record for every item."""
        from kgproweight.kg.question_kg import question_key, question_sha256
        if not self._current_dataset_name:
            raise ValueError("kg_supply_mode=qpeg_v1 requires a known dataset_name")
        missing: List[str] = []
        mismatched: List[str] = []
        for item in dataset:
            key = question_key(self._current_dataset_name, str(item.id))
            record = self._qpeg_records.get(key)
            if record is None:
                missing.append(str(item.id))
            elif record.get("question_sha256") != question_sha256(str(item.question)):
                mismatched.append(str(item.id))
        if missing or mismatched:
            raise ValueError(
                "kg_supply_mode=qpeg_v1 identity/hash join < 1.0 on "
                f"{self._current_dataset_name!r}: missing={len(missing)}, "
                f"hash_mismatch={len(mismatched)}; first missing={missing[:5]!r}, "
                f"first mismatched={mismatched[:5]!r}"
            )

    def _build_kg_context(self, item, passages=None) -> List[Tuple[str, str, str]]:
        if not self.inject_kg:
            return []

        if self.kg_supply_mode == "qpeg_v1":
            from kgproweight.kg.question_kg import question_key, question_sha256
            from kgproweight.kg.qpeg import compute_passages_sha256
            _key = question_key(self._current_dataset_name, str(item.id))
            _record = self._qpeg_records.get(_key)
            if _record is None:
                raise ValueError(f"QPEG record missing for {_key}")
            if _record.get("question_sha256") != question_sha256(str(item.question)):
                raise ValueError(f"QPEG question hash mismatch for {_key}")
            if passages is None or _record.get("passages_sha256") != compute_passages_sha256(passages):
                raise ValueError(f"QPEG passages hash mismatch for {_key}")
            self._kg_source_counts["index"] += 1
            return [tuple(value) for value in _record["kg_subgraph"][: self.max_kg_triples]]

        # ProofKG-v1 supply: look up the versioned per-question record by
        # dataset::qid, exactly as training consumed it. No live-Wikidata fallback
        # in this mode — a missing record is an empty KG, not a silent legacy mix.
        if self.kg_supply_mode == "proofkg_v1":
            from kgproweight.kg.question_kg import question_key
            _key = question_key(self._current_dataset_name, str(item.id))
            if _key in self._proofkg_records:
                self._kg_source_counts["index"] += 1
                return list(self._proofkg_records[_key][: self.max_kg_triples])
            self._kg_source_counts["empty"] += 1
            return []

        if self.kg_supply_mode in ("legacy_plus_proofkg", "legacy_plus_complete_proofkg"):
            legacy = self._build_legacy_kg_context(item, passages)
            arm = "complete" if self.kg_supply_mode == "legacy_plus_complete_proofkg" else "partial"
            return self._augment_with_selective(legacy, item, arm)

        return self._build_legacy_kg_context(item, passages)

    def _build_legacy_kg_context(self, item, passages=None) -> List[Tuple[str, str, str]]:
        dropout = self._get_dropout_kg(item)
        if dropout is not None:
            return list(dropout)

        # R9 v6: prefer pre-built filtered KG cache (aligned with training).
        # Falls back to live Wikidata lookup only on cache miss.
        # The index builder stores keys STRIPPED (q.strip()); the eval question
        # string is raw, so ~10% of questions carry a trailing/leading space and
        # silently miss the curated index, falling back to lower-quality (or
        # empty) live KG. Strip here to match the builder's key normalization.
        cached = self._q_kg_index.get(item.question) or self._q_kg_index.get(item.question.strip())
        if cached:
            self._kg_source_counts["index"] += 1
            return list(cached[:self.max_kg_triples])

        # R9 v6 fix: extract passage titles for context-aware entity linking.
        # R9 v7: also pass the passage *bodies* (build_passage_text) so the
        # linker's passage-support term can disambiguate surface forms that
        # share a label — "Evolution" film vs GNOME software, etc. The title
        # alone is uninformative for such cases; the disambiguating signal
        # ("film", "directed", "software") lives in the body.
        _passage_titles: Optional[List[str]] = build_passage_titles(passages) if passages else None
        _passage_text: Optional[str] = build_passage_text(passages) if passages else None

        mentions = extract_mentions(item.question, max_n=self.max_mentions)
        qids = []
        for m in mentions:
            result = self.entity_linker.link_single(
                m, question=item.question,
                retrieved_titles=_passage_titles,
                passage_text=_passage_text,
            )
            if not result.abstained and result.selected_qid:
                qids.append(result.selected_qid)
        if not qids:
            self._kg_source_counts["empty"] += 1
            return []
        raw = self.kg_retriever.fetch(qids)
        if not raw:
            self._kg_source_counts["empty"] += 1
            return []
        # R9 v6 fix: the fallback previously returned RAW SPARQL-order triples,
        # bypassing the three-layer filter entirely. Since the v2 index only
        # covers the questions it was built from, most eval questions took this
        # path — so the measured "74.6% noise removed" never reached inference.
        # Apply the same scoring/quota policy here.
        self._kg_source_counts["fallback"] += 1
        filtered = filter_and_rank_triples(
            raw, question=item.question, max_keep=self.max_kg_triples, min_keep=5,
        )
        # R9 v6 Phase C: passage-verified filtering. Triples whose entities
        # never appear in any retrieved passage have no textual grounding for
        # the model — they're floating facts that the model can't verify.
        if passages and filtered:
            from kgproweight.kg.kg_filter import filter_by_passage_support
            filtered = filter_by_passage_support(filtered, passages)
        return filtered

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, dataset, do_eval: bool = True, pred_process_fun=None):
        self._current_dataset_name = getattr(dataset, "dataset_name", "")
        questions = list(dataset.question)
        # ProofKG-v1 supply: refuse to run on a partial question set rather than
        # silently emitting empty KG for the uncovered questions.
        if self.kg_supply_mode == "proofkg_v1":
            self._enforce_proofkg_join(dataset)
        elif self.kg_supply_mode == "qpeg_v1":
            self._enforce_qpeg_join(dataset)
        logger.info("KGProWeight inference on %d samples (top_k=%d, inject_kg=%s)",
                    len(questions), self.retrieval_topk, self.inject_kg)

        retrieval_results = self.retriever.batch_search(questions)

        # R9 v6: two-stage retrieval with reranker
        if self.rerank_topk > 0 and retrieval_results and len(retrieval_results[0]) > self.rerank_topk:
            from kgproweight.retrieval.reranker import (
                RetrievalConfig, pack_passages_by_token_budget, rerank_passages,
            )
            rcfg = RetrievalConfig(
                rrf_candidate_topk=len(retrieval_results[0]),
                rerank_topk=self.rerank_topk,
                prompt_passage_token_budget=3860,
                rerank_method=self.rerank_method,
                cross_encoder_model=self.cross_encoder_model,
            )
            logger.info("R9 v6 retrieval: %s", rcfg.log_string())
            # Was hardcoded to the hand-rolled BM25 scorer even though
            # bge-reranker-v2-m3 is present locally and Phase 1 already uses it —
            # so train and eval reranked with DIFFERENT models. Now both go
            # through one dispatcher, cross-encoder by default.
            retrieval_results = rerank_passages(
                questions, retrieval_results,
                topk=self.rerank_topk,
                method=rcfg.rerank_method,
                cross_encoder_model=rcfg.cross_encoder_model,
            )
            # Apply token budget to each question's passages
            retrieval_results = [
                pack_passages_by_token_budget(passages, rcfg.prompt_passage_token_budget)
                for passages in retrieval_results
            ]

        dataset.update_output("retrieval_result", retrieval_results)

        prompts: List[str] = []
        kg_subgraphs: List[List[Tuple[str, str, str]]] = []
        used_dropout: List[bool] = []

        for item, passages in zip(dataset, retrieval_results):
            kg_sub = self._build_kg_context(item, passages=passages)
            kg_subgraphs.append(kg_sub)
            used_dropout.append(self._get_dropout_kg(item) is not None)
            msgs = build_inference_messages(
                question=item.question,
                retrieved_passages=passages,
                kg_triples=kg_sub,
                top_k=self.retrieval_topk,
                max_kg_triples=self.max_kg_triples,
            )
            prompts.append(self.prompt_template.get_string(messages=msgs))

        _src = self._kg_source_counts
        _tot = max(1, sum(_src.values()))
        logger.info(
            "KG source: prebuilt index %d (%.0f%%), live fallback %d (%.0f%%), empty %d (%.0f%%) "
            "| mean triples/question=%.1f",
            _src["index"], 100 * _src["index"] / _tot,
            _src["fallback"], 100 * _src["fallback"] / _tot,
            _src["empty"], 100 * _src["empty"] / _tot,
            sum(len(k) for k in kg_subgraphs) / max(1, len(kg_subgraphs)),
        )
        if self.inject_kg and _src["index"] == 0 and _src["fallback"] + _src["empty"] > 0:
            logger.warning(
                "question_kg_index covered 0/%d eval questions — the pre-built v2 KG is "
                "NOT being used. Rebuild it for this split with "
                "scripts/prepare/06_build_question_kg_index.py --datasets <ds> --split <split>.",
                _tot,
            )

        dataset.update_output("prompt", prompts)
        dataset.update_output("kg_subgraphs", kg_subgraphs)
        dataset.update_output("used_dropout_kg", used_dropout)

        # D1: request per-token scores so f_entropy uses real logprobs instead
        # of the hardcoded 1.0 default. The generator already computes these;
        # we just need return_scores=True to get them back.
        try:
            raw_outputs, token_scores = self.generator.generate(prompts, return_scores=True)
        except (TypeError, ValueError):
            raw_outputs = self.generator.generate(prompts)
            token_scores = None
        dataset.update_output("raw_output", raw_outputs)

        preds: List[str] = []
        alpha_stats: List[Dict] = []
        ihr_stats: List[Dict] = []

        for i, (item, raw_output, kg_sub) in enumerate(zip(dataset, raw_outputs, kg_subgraphs)):
            pred = extract_kg_proweight_answer(raw_output)
            preds.append(pred)
            # D1: compute per-question token logprobs from real generation scores
            _logprobs = None
            if token_scores is not None and i < len(token_scores):
                import math
                _scores = [max(s, 1e-9) for s in token_scores[i]]
                _logprobs = [math.log(s) for s in _scores]
            alpha_stats.append(self._compute_alpha_stats(raw_output, kg_sub, item.question,
                                                          logprobs=_logprobs))
            ihr_stats.append(self._compute_ihr(raw_output, kg_sub))

        dataset.update_output("pred", preds)
        dataset.update_output("alpha_stats", alpha_stats)
        dataset.update_output("ihr_flags", ihr_stats)

        if self.record_alpha:
            self._alpha_records.extend(alpha_stats)

        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def _compute_alpha_stats(self, generated_text: str, kg_subgraph, query: str,
                              logprobs=None) -> Dict:
        # Telemetry/parser change (registered separately from the KG-source
        # switch): passing known_kg makes citation parsing match the prompt's
        # exact "(h, r, t)" rendering instead of comma-splitting. It changes
        # cited_triples → α (cite_any/cite_match) and heuristic IHR only; it does
        # NOT enter prompt construction, answer extraction, or EM/F1.
        steps = parse_steps(generated_text, known_kg=kg_subgraph) if generated_text else []
        alphas: List[float] = []
        for step in steps:
            # D1: use real per-token logprobs from generation when available.
            # Falls back to None (→ f_entropy=1.0) for backward compatibility.
            f_density, f_confidence, f_entropy = compute_features(
                step_entities=step.mentioned_entities,
                kg_subgraph=kg_subgraph,
                logprobs=logprobs,  # D1: was hardcoded None
                entity_linker=self.entity_linker,
            )
            # §14: the two per-step citation features, through the same shared
            # helper Phase 2 and the PPO reward use. Omitting them here would make
            # eval-time α a DIFFERENT function from training-time α -- the exact
            # class of train/inference mismatch the +0.78 bias hack turned out to be.
            f_cite_any, f_cite_match = citation_features(step.cited_triples, kg_subgraph)
            alphas.append(self.alpha_gate.forward_single(
                f_density, f_confidence, f_entropy, f_cite_any, f_cite_match
            ))

        mean_alpha = sum(alphas) / len(alphas) if alphas else 0.0
        var = (sum((a - mean_alpha) ** 2 for a in alphas) / len(alphas)) if len(alphas) > 1 else 0.0
        return {
            "query": query,
            "num_steps": len(steps),
            "alpha_mean": mean_alpha,
            "alpha_std": var ** 0.5,
            "alpha_values": alphas,
        }

    def _compute_ihr(self, generated_text: str, kg_subgraph) -> Dict:
        if not generated_text or not kg_subgraph:
            return {"ihr_heuristic": None, "n_steps": 0, "n_hallucinated": 0}
        # Same telemetry/parser change as _compute_alpha_stats: known_kg affects
        # cited_triples → heuristic IHR, not the answer/EM/F1 path.
        steps = parse_steps(generated_text, known_kg=kg_subgraph)
        labels = self.prm_annotator.annotate_trajectory(steps, kg_subgraph)
        n_neg = sum(1 for x in labels if x == -1)
        total = len(labels)
        return {
            "ihr_heuristic": (n_neg / total) if total else 0.0,
            "n_steps": total,
            "n_hallucinated": n_neg,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_alpha_distribution(self, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for record in self._alpha_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved α distribution to %s", output_path)

    def print_alpha_summary(self) -> None:
        if not self._alpha_records:
            logger.info("No α records collected.")
            return
        all_alphas = [a for r in self._alpha_records for a in r.get("alpha_values", [])]
        if not all_alphas:
            return
        mean_alpha = sum(all_alphas) / len(all_alphas)
        std_alpha = (sum((a - mean_alpha) ** 2 for a in all_alphas) / len(all_alphas)) ** 0.5
        logger.info(
            "α summary: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
            len(all_alphas),
            mean_alpha,
            std_alpha,
            min(all_alphas),
            max(all_alphas),
        )
