#!/usr/bin/env python3
"""Freeze and materialize train-only SFT-v3 canonical Wiki18 contexts in batches.

Only an already-selected, hash-bound question-only pool is accepted.  No QA
labels, teacher responses or support annotations are read.  The production
backend calls the loaded BGE model directly, so BM25 fallback is impossible.
Each completed batch is an immutable self-contained JSON artifact; --resume
verifies every existing batch before reusing it.  This does not select data,
construct graphs, accept teacher supervision or start SFT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VERSION = "sft-v3-canonical-retrieval-v1"
STACK = "E5@100+BM25@100->RRF60@50->bge-reranker-v2-m3@10->pack3860"
DATASETS = {"hotpotqa", "2wikimultihopqa", "musique"}
ALLOWED = {"dataset", "qid", "question", "question_key", "question_sha256", "family_sha256", "family_version", "role", "gold_access", "split", "selection_rank", "schema_version", "source_split", "within_split_dataset_rank", "consumption_order"}
FORBIDDEN = {"answer", "answers", "gold_answer", "gold_answers", "gold_answer_aliases", "golden_answers", "support", "supporting_facts", "decomposition", "question_decomposition", "teacher_output", "target", "labels", "evidences", "reasoning", "steps"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stat(path: Path) -> dict:
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "size_bytes": path.stat().st_size}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def assert_gold_free(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN & set(value):
            raise ValueError(f"forbidden Gold/supervision fields: {sorted(FORBIDDEN & set(value))}")
        for v in value.values():
            assert_gold_free(v)
    elif isinstance(value, list):
        for v in value:
            assert_gold_free(v)


def validate_requests(rows: list[dict]) -> None:
    from kgproweight.kg.question_kg import question_key, question_sha256
    from scripts.prepare.freeze_qpeg_v1_protocol import FAMILY_VERSION, family_sha256
    if not rows:
        raise ValueError("empty frozen question pool")
    seen, hashes, families = set(), set(), set()
    for r in rows:
        if not isinstance(r, dict) or set(r) - ALLOWED:
            raise ValueError("request must contain only allowlisted question/identity fields")
        assert_gold_free(r)
        ds, qid, q = r.get("dataset"), r.get("qid"), r.get("question")
        if ds not in DATASETS or not isinstance(qid, str) or not qid.strip() or not isinstance(q, str) or not q.strip():
            raise ValueError("invalid dataset/qid/question")
        expected = {"question_key": question_key(ds, qid), "question_sha256": question_sha256(q), "family_sha256": family_sha256(q), "family_version": FAMILY_VERSION, "gold_access": False}
        if any(r.get(k) != v for k, v in expected.items()) or not isinstance(r.get("role"), str) or not r["role"]:
            raise ValueError("question identity or gold_access mismatch")
        if "split" in r and r["split"] not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if "source_split" in r and r["source_split"] != "train":
            raise ValueError("source_split must be raw train")
        for field in ("within_split_dataset_rank", "consumption_order"):
            if field in r and (type(r[field]) is not int or r[field] < 1):
                raise ValueError(f"{field} must be a positive integer")
        if "selection_rank" in r and (type(r["selection_rank"]) is not int or r["selection_rank"] < 0):
            raise ValueError("selection_rank must be a nonnegative integer")
        if r["question_key"] in seen or r["question_sha256"] in hashes or r["family_sha256"] in families:
            raise ValueError("duplicate qid/question/global family in frozen pool")
        seen.add(r["question_key"]); hashes.add(r["question_sha256"]); families.add(r["family_sha256"])


def asset_paths(root: Path = ROOT) -> list[Path]:
    wiki = root / "indexes_wiki18"
    paths = [wiki / n for n in ("corpus_flashrag.jsonl", "corpus_flashrag.mmindex.json", "e5_fp16.dat")]
    paths += [wiki / "bm25" / n for n in ("data.csc.index.npy", "indices.csc.index.npy", "indptr.csc.index.npy", "params.index.json", "stopwords.tokenizer.json", "vocab.index.json", "vocab.tokenizer.json")]
    for name in ("e5-base-v2", "bge-reranker-v2-m3"):
        model = root / "models" / name
        if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            raise FileNotFoundError(f"local retrieval model incomplete: {model}")
        paths += [p for p in sorted(model.rglob("*")) if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".model"}]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("complete full Wiki18 corpus/index assets required")
    return paths


def code_paths(root: Path = ROOT) -> list[Path]:
    paths = [Path(__file__).resolve()]
    paths += [root / n for n in ("kgproweight/retrieval/hybrid.py", "kgproweight/retrieval/reranker.py", "kgproweight/retrieval/np_search.py", "kgproweight/retrieval/bootstrap.py", "kgproweight/utils/paths.py", "kgproweight/utils/flashrag_bootstrap.py", "kgproweight/data/flashrag_loader.py", "kgproweight/kg/question_kg.py", "scripts/prepare/freeze_qpeg_v1_protocol.py", "scripts/prepare/materialize_mixed3_v4_expansion_retrieval.py", "scripts/prepare/materialize_qpeg_v1_retrieval.py")]
    for directory in ("retriever", "config", "utils"):
        paths += sorted((root / "flashrag_src/flashrag" / directory).glob("*.py"))
        paths += sorted((root / "flashrag_src/flashrag" / directory).glob("*.yaml"))
    return paths


def prepare(requests: Path, expected_sha: str, directory: Path, experiment_id: str, batch_size: int, *, bind_assets: bool = True) -> dict:
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite: {directory}; use --resume for a verified frozen run")
    if not experiment_id.strip() or batch_size < 1 or sha(requests) != expected_sha:
        raise ValueError("nonempty Experiment ID, positive batch and matching request SHA required")
    rows = read_rows(requests); validate_requests(rows)
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(requests, directory / "requests.question_only.jsonl")
    if sha(directory / "requests.question_only.jsonl") != expected_sha:
        raise ValueError("request file changed while freezing")
    assets = []
    if bind_assets:
        for path in asset_paths():
            before = stat(path); bound = {**identity(path), "stat": before}
            if stat(path) != before:
                raise ValueError(f"asset changed during full SHA: {path}")
            assets.append(bound)
            print(f"BOUND {path.name}", flush=True)
    protocol = {"schema_version": VERSION, "status": "FROZEN_RETRIEVAL_NOT_STARTED", "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "requests": identity(directory / "requests.question_only.jsonl"),
        "source_requests": identity(requests), "request_counts": dict(Counter(r["dataset"] for r in rows)), "rows": len(rows),
        "batch_size": batch_size, "retrieval": STACK, "full_wiki18_documents": 21015324,
        "execution": {"seed": 42, "torch_threads": 4, "retrieval_query_max_length": 128, "cross_encoder_max_chars": 1200,
            "reranker_fallback": False, "one_original_question_query_per_row": True, "query_expansion": False,
            "asset_verification": "full SHA at freeze; device/inode/bytes/mtime_ns checked before and after run; code SHA always rechecked"},
        "assets": assets, "code": [identity(p) for p in code_paths()], "test_double_only": not bind_assets,
        "scientific_boundary": {"gold_access": False, "teacher_calls": 0, "selection_performed": False, "kg_constructed": False, "model_updates": 0}}
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "prepared.json", {"protocol": identity(directory / "protocol.json")})
    return protocol


def verify(directory: Path) -> tuple[dict, list[dict]]:
    prepared = json.loads((directory / "prepared.json").read_text())
    if identity(directory / "protocol.json") != prepared["protocol"]:
        raise ValueError("frozen protocol drift")
    p = json.loads((directory / "protocol.json").read_text())
    for bound in [p["requests"], *p["code"]]:
        if identity(Path(bound["path"])) != bound:
            raise ValueError(f"frozen input/code drift: {bound['path']}")
    for bound in p["assets"]:
        if stat(Path(bound["path"])) != bound["stat"]:
            raise ValueError(f"frozen asset stat drift: {bound['path']}")
    rows = read_rows(Path(p["requests"]["path"])); validate_requests(rows)
    return p, rows


class StrictCanonicalBackend:
    def __init__(self, directory: Path):
        # Explicit full-index paths bypass project smoke-index defaults.
        os.environ.update({"KGPW_CORPUS_MMAP": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        import torch
        from kgproweight.data.flashrag_loader import flashrag_config
        from kgproweight.retrieval.hybrid import build_flashrag_config, build_rrf_setting
        from kgproweight.retrieval.reranker import get_cross_encoder
        from kgproweight.utils.flashrag_bootstrap import setup_flashrag
        torch.set_num_threads(4); torch.manual_seed(42)
        if not torch.cuda.is_available():
            raise RuntimeError("local CUDA is required for frozen E5/BGE retrieval")
        setup_flashrag()
        from flashrag.utils import get_retriever
        wiki = ROOT / "indexes_wiki18"
        cfg = build_flashrag_config("hotpotqa", "sft_v3_canonical_retrieval", str(directory / "runtime"), topk=50, split="train", corpus_path=str(wiki / "corpus_flashrag.jsonl"), seed=42)
        cfg.update({"index_path": str(wiki / "e5_fp16.dat"), "bm25_index_path": str(wiki / "bm25"), "use_retrieval_cache": False, "save_retrieval_cache": False,
            "multi_retriever_setting": build_rrf_setting(topk=50, dense_index_path=str(wiki / "e5_fp16.dat"), sparse_index_path=str(wiki / "bm25"), dense_model_path=str(ROOT / "models/e5-base-v2"), corpus_path=str(wiki / "corpus_flashrag.jsonl"))})
        self.router = get_retriever(flashrag_config(cfg))
        if [r.retrieval_method for r in self.router.retriever_list] != ["e5", "bm25"] or any(len(r.corpus) != 21015324 for r in self.router.retriever_list):
            raise ValueError("full Wiki18 E5 and BM25 branches required; smoke corpus forbidden")
        self.ce = get_cross_encoder(str(ROOT / "models/bge-reranker-v2-m3"))
        if self.ce is None:
            raise RuntimeError("BGE could not load; no fallback permitted")
        self.attestation = {"mode": "real_full_wiki18", "fallback": False, "load_succeeded": True, "retrieval_config": cfg,
            "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__, "corpus_documents": 21015324,
            "bge_path": str(ROOT / "models/bge-reranker-v2-m3"), "bge_max_length": self.ce.max_length,
            "bge_dtype": str(next(self.ce.model.parameters()).dtype), "bge_device": str(self.ce.device),
            "dataset_config_note": "one shared corpus-only retriever; the dataset tag is bookkeeping and no raw dataset is loaded"}

    def __call__(self, rows: list[dict]) -> tuple[list[dict], list[dict]]:
        from kgproweight.retrieval.reranker import pack_passages_by_token_budget
        from scripts.prepare.materialize_qpeg_v1_retrieval import _sha_json
        questions = [r["question"] for r in rows]; branches = []
        for retriever in self.router.retriever_list:
            values = retriever.batch_search(questions, num=100)
            if len(values) != len(rows) or any(len(v) != 100 for v in values):
                raise ValueError("each branch must return exactly 100 documents per request")
            branches.append(self.router.add_source(values, retriever))
        merged, rrf_scores = self.router.rrf_merge([d + s for d, s in zip(*branches)], topk=50, k=60)
        if len(merged) != len(rows):
            raise ValueError("RRF request count mismatch")
        contexts, traces = [], []
        for i, (r, candidates) in enumerate(zip(rows, merged)):
            assert_gold_free(candidates)
            scores = [float(s) for s in self.ce.predict([(r["question"], str(c.get("contents") or c.get("text") or "")[:1200]) for c in candidates], show_progress_bar=False)]
            if len(candidates) != 50 or len(scores) != 50 or not all(math.isfinite(s) for s in scores):
                raise ValueError("BGE must score exactly 50 candidates with finite scores")
            order = sorted(range(50), key=lambda j: scores[j], reverse=True)
            passages = pack_passages_by_token_budget([candidates[j] for j in order[:10]], 3860)
            context = {**r, "schema_version": VERSION, "passages": passages, "passages_sha256": _sha_json(passages), "retrieval_source": STACK}
            if "schema_version" in r:
                context["request_schema_version"] = r["schema_version"]
            contexts.append(context)
            traces.append({"question_key": r["question_key"], "dense_ids": [str(d["id"]) for d in branches[0][i]], "sparse_ids": [str(d["id"]) for d in branches[1][i]],
                "rrf_ids": [str(d["id"]) for d in candidates], "rrf_scores": [float(s) for s in rrf_scores[i]], "bge_scores": scores, "bge_sorted_indices": order,
                "gold_access": False})
        return contexts, traces


def validate_contexts(requests: list[dict], contexts: list[dict]) -> None:
    from scripts.prepare.materialize_qpeg_v1_retrieval import _sha_json
    if len(requests) != len(contexts):
        raise ValueError("batch returned incorrect request count")
    for r, c in zip(requests, contexts):
        assert_gold_free(c)
        if any(c.get(k) != v for k, v in r.items() if k != "schema_version") or c.get("retrieval_source") != STACK:
            raise ValueError("materialized context identity/order/stack drift")
        passages = c.get("passages", [])
        if len(passages) != 10 or any(not isinstance(p, dict) or any(not str(p.get(k) or "").strip() for k in ("id", "contents", "source")) for p in passages):
            raise ValueError("exactly ten nonempty sourced passages required")
        if len({str(p["id"]) for p in passages}) != 10 or _sha_json(passages) != c.get("passages_sha256"):
            raise ValueError("duplicate passages or bad passage hash")


def run(directory: Path, *, backend: Any = None, max_batches: int | None = None) -> dict:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    p, rows = verify(directory); start = time.monotonic()
    if (directory / "manifest.json").exists():
        raise FileExistsError("retrieval release is complete; no overwrite")
    if p["test_double_only"] != (backend is not None):
        raise ValueError("test and production modes cannot be mixed")
    batch_dir = directory / "batches"; batch_dir.mkdir(exist_ok=True)
    attempt = len(list(directory.glob("attempt_*.json"))) + 1
    attempt_path = directory / f"attempt_{attempt:04d}.json"
    write_json(attempt_path, {"protocol": identity(directory / "protocol.json"), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    batch_refs, all_contexts, reused, new_batches = [], [], 0, 0
    expected_files = {f"{start:08d}{suffix}" for start in range(0, len(rows), p["batch_size"]) for suffix in (".json", ".seal.json")}
    if any(f.name not in expected_files for f in batch_dir.iterdir()):
        raise ValueError("unexpected or incomplete batch artifact; retain and diagnose before resume")
    try:
        for offset in range(0, len(rows), p["batch_size"]):
            request_batch = rows[offset:offset + p["batch_size"]]; path = batch_dir / f"{offset:08d}.json"
            request_hash = hashlib.sha256(canonical(request_batch).encode()).hexdigest()
            if not path.exists() and max_batches is not None and new_batches >= max_batches:
                verify(directory)
                progress = {"schema_version": VERSION, "experiment_id": p["experiment_id"], "status": "PARTIAL_CHECKPOINT_PAUSED",
                    "frozen_requests": len(rows), "completed_contexts": len(all_contexts), "reused_contexts": reused, "new_batches_this_attempt": new_batches,
                    "next_request_offset": offset, "batch_limit_is_operational_only": True, "elapsed_seconds_this_attempt": time.monotonic() - start,
                    "protocol": identity(directory / "protocol.json"), "batches": batch_refs, "gold_access": False, "teacher_calls": 0, "model_updates": 0}
                write_json(directory / f"progress_{attempt:04d}.json", progress)
                return progress
            seal = path.with_name(path.stem + ".seal.json")
            if path.exists():
                if not seal.exists() or json.loads(seal.read_text()).get("batch") != identity(path):
                    raise ValueError("immutable cached batch seal mismatch or missing")
                b = json.loads(path.read_text())
                if b["request_sha256"] != request_hash or b["protocol_sha256"] != sha(directory / "protocol.json") or hashlib.sha256(canonical(b["contexts"]).encode()).hexdigest() != b["contexts_sha256"]:
                    raise ValueError("immutable cached batch drift")
                contexts = b["contexts"]; reused += len(contexts)
            else:
                if backend is None:
                    backend = StrictCanonicalBackend(directory)
                batch_start = time.monotonic(); contexts, traces = backend(request_batch)
                validate_contexts(request_batch, contexts)
                attestation = getattr(backend, "attestation", {})
                if attestation.get("fallback") is not False or attestation.get("load_succeeded") is not True:
                    raise ValueError("actual backend load/fallback attestation required")
                write_json(path, {"schema_version": VERSION, "request_sha256": request_hash, "protocol_sha256": sha(directory / "protocol.json"), "contexts_sha256": hashlib.sha256(canonical(contexts).encode()).hexdigest(),
                    "contexts": contexts, "retrieval_trace": traces, "backend_attestation": attestation, "elapsed_seconds": time.monotonic() - batch_start})
                write_json(seal, {"batch": identity(path), "protocol_sha256": sha(directory / "protocol.json")})
                new_batches += 1
            validate_contexts(request_batch, contexts); all_contexts.extend(contexts); batch_refs.append(identity(path))
            print(f"BATCH_COMPLETE {len(all_contexts)}/{len(rows)} reused={reused}", flush=True)
        verify(directory)
        output = directory / "retrieval_contexts.jsonl"
        with output.open("x", encoding="utf-8") as f:
            for c in all_contexts:
                f.write(canonical(c) + "\n")
        report = {"schema_version": VERSION, "experiment_id": p["experiment_id"], "status": "COMPLETE_TEST_DOUBLE_ONLY" if p["test_double_only"] else "COMPLETE_CANONICAL_RETRIEVAL_NOT_TEACHER_DATA_NOT_TRAINED",
            "requests": len(rows), "contexts": len(all_contexts), "counts": dict(Counter(r["dataset"] for r in rows)), "reused_contexts": reused,
            "hybrid_queries": len(rows), "dense_queries": len(rows), "sparse_queries": len(rows), "bge_pairs": 50 * len(rows),
            "elapsed_seconds_this_attempt": time.monotonic() - start, "retrieval": STACK, "gold_access": False, "teacher_calls": 0, "model_updates": 0, "all_exactly_ten": True,
            "asset_verification": p["execution"]["asset_verification"], "protocol": identity(directory / "protocol.json"), "batches": batch_refs, "outputs": {"retrieval_contexts": identity(output)}}
        write_json(directory / "report.json", report)
        write_json(directory / "manifest.json", {**report, "outputs": {**report["outputs"], "report": identity(directory / "report.json")}})
        return report
    except Exception as exc:
        write_json(directory / f"failure_{attempt:04d}.json", {"exception_type": type(exc).__name__, "message": str(exc), "completed_batches": batch_refs, "retained": True})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--requests_sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_batches", type=int, help="pause after this many NEW batches; frozen pool/order remain unchanged")
    args = parser.parse_args()
    if args.resume:
        p, _ = verify(args.out)
        if p["source_requests"] != identity(args.requests) or p["requests"]["sha256"] != args.requests_sha256 or p["experiment_id"] != args.experiment_id or p["batch_size"] != args.batch_size:
            raise ValueError("resume arguments differ from frozen protocol")
    else:
        prepare(args.requests, args.requests_sha256, args.out, args.experiment_id, args.batch_size)
    if not args.prepare_only:
        print(json.dumps(run(args.out, max_batches=args.max_batches), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
