"""One frozen, Gold-free pseudo-relevance evidence expansion on consumed train data.

This is a data-preparation pilot, not a Reader search controller.  It never
opens QA labels, official supports, generated answers, or evidence annotations.
The old 10-passage release and every baseline remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PARENT = ROOT / "outputs/audits/generation_length_384_512_paired_probe_20260906_v1"
DEFAULT_OUTPUT = ROOT / "outputs/audits/evidence_supply_v1_consumed20_20260906_v1"
VERSION = "evidence-supply-visible-entity-expansion-v1"
EXPERIMENT = "EVIDENCE-SUPPLY-CONSUMED-MUSIQUE20-20260906-V1"
FORBIDDEN = {"answer", "answers", "gold_answer", "gold_answers", "gold_answer_aliases",
             "golden_answers", "supporting_facts", "decomposition", "question_decomposition",
             "teacher_output", "target", "labels", "evidence_review", "semantic_annotation"}
STOP = set("The A An He She It They His Her Its Their This That These Those In On At To From By For As Of With After Before During Following When Where Which What Who How Although Born January February March April May June July August September October November December".casefold().split())
TOKEN = r"(?:[A-Z](?:\.[A-Z])+\.?|[A-Z][A-Za-zÀ-ÖØ-öø-ÿ’'\-]*)"
PHRASE = re.compile(rf"(?<![\w]){TOKEN}(?:[ \t]+(?:(?:of|the|de|van|von)[ \t]+)?{TOKEN}){{0,5}}")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def stat_signature(path: Path) -> dict:
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def assert_gold_free(value: Any) -> None:
    if isinstance(value, dict):
        if set(value) & FORBIDDEN:
            raise ValueError("forbidden Gold/support/annotation field in evidence pilot")
        for item in value.values():
            assert_gold_free(item)
    elif isinstance(value, list):
        for item in value:
            assert_gold_free(item)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold()))


def passage_text(passage: dict) -> str:
    return str(passage.get("contents") or passage.get("text") or "")


def entity_mentions(text: str) -> list[dict]:
    found = []
    for match in PHRASE.finditer(text):
        raw = match.group()
        words = list(re.finditer(r"\S+", raw))
        while words and words[0].group().casefold() in STOP:
            words.pop(0)
        if not words:
            continue
        phrase = raw[words[0].start():].rstrip(".")
        offset = match.start() + words[0].start()
        if len(normalize(phrase)) >= 3 and normalize(phrase) not in STOP:
            found.append({"entity": phrase, "offset": offset})
    return found


def build_query_plan(question: str, passages: list[dict]) -> dict:
    """Freeze at most three visible novel name strings; no learned selection."""
    if len(passages) != 10:
        raise ValueError("exactly ten original passages required")
    assert_gold_free(passages)
    question_norm = " " + normalize(question) + " "
    question_entities = [normalize(m["entity"]) for m in entity_mentions(question)]
    names = {}
    for rank, passage in enumerate(passages[:3]):
        for mention in entity_mentions(passage_text(passage)):
            norm = normalize(mention["entity"])
            if " " + norm + " " in question_norm:
                continue
            # A longer surface containing the already mentioned entity is not
            # a novel bridge (e.g. an expanded full name of the same person).
            if any(" " + q + " " in " " + norm + " " for q in question_entities):
                continue
            item = names.setdefault(norm, {"entity": mention["entity"], "normalized": norm,
                                          "first_passage_rank": rank + 1,
                                          "first_offset": mention["offset"], "occurrences": []})
            item["occurrences"].append({"passage_rank": rank + 1, "offset": mention["offset"],
                                         "passage_id": str(passage["id"])})
    ranked = sorted(names.values(), key=lambda v: (-len({m["passage_rank"] for m in v["occurrences"]}),
                                                   v["first_passage_rank"], v["first_offset"], v["normalized"]))
    for item in ranked:
        item["passage_frequency"] = len({m["passage_rank"] for m in item["occurrences"]})
    queries = [{"query_index": i, "query": question + " " + item["entity"], **item}
               for i, item in enumerate(ranked[:3])]
    return {"queries": queries, "ranked_entity_candidates": ranked,
            "upstream_hybrid_queries": len(queries), "source_passage_ranks": [1, 2, 3]}


def select_passages(original: list[dict], ranked_expansions: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    """Retain first four distinct contents, then cycle expanded rankings."""
    if len(original) != 10 or len({str(p["id"]) for p in original}) != 10:
        raise ValueError("original must have ten distinct document IDs")
    seen, seen_contents = set(), set()
    selected, trace = [], []

    def add(passage, origin, rank):
        did = str(passage["id"])
        content = normalize(passage_text(passage))
        if not content:
            raise ValueError("empty passage")
        if did in seen or content in seen_contents or len(selected) == 10:
            return False
        seen.add(did)
        seen_contents.add(content)
        selected.append(deepcopy(passage))
        trace.append({"id": did, "origin": origin, "origin_rank": rank,
                      "passage_sha256": digest(passage)})
        return True

    for rank, passage in enumerate(original, 1):
        add(passage, "legacy", rank)
        if len(selected) == 4:
            break
    # New documents are required for the expansion slots; an already visible
    # document found again cannot consume one of the six discovery slots.
    old_ids = {str(p["id"]) for p in original}
    old_contents = {normalize(passage_text(p)) for p in original}
    queues = [[(i + 1, p) for i, p in enumerate(ps) if str(p["id"]) not in old_ids and normalize(passage_text(p)) not in old_contents]
              for ps in ranked_expansions]
    while len(selected) < 10 and any(queues):
        for qi, queue in enumerate(queues):
            while queue:
                rank, passage = queue.pop(0)
                if add(passage, f"expanded_query_{qi}", rank):
                    break
    for rank, passage in enumerate(original, 1):
        add(passage, "legacy_backfill", rank)
    if len(selected) != 10:
        raise ValueError("could not retain final ten distinct passages")
    return selected, trace


def rebind_input(old: dict, passages: list[dict], tokenizer) -> dict:
    from kgproweight.data.prompts import build_rl_messages
    assert_gold_free(old)
    if old["m_graph"] != 0 or old["kg_subgraph"] or old["dataset"] != "musique":
        raise ValueError("this pilot only supports the frozen ordinary MuSiQue slice")
    if len(passages) != 10:
        raise ValueError("reader must receive exactly ten passages")
    new = deepcopy(old)
    new["retrieved_passages"] = deepcopy(passages)
    new["spec"]["retrieved_passages"] = deepcopy(passages)
    new["messages"] = build_rl_messages(old["question"], passages, old["kg_subgraph"], top_k=10, max_kg_triples=12)
    if new["messages"][0] != old["messages"][0]:
        raise ValueError("legacy system prompt drift")
    new["prompt"] = tokenizer.apply_chat_template(new["messages"], tokenize=False, add_generation_prompt=True)
    new["prompt_tokens"] = len(tokenizer(new["prompt"], add_special_tokens=False, truncation=False)["input_ids"])
    if new["prompt_tokens"] > 6144:
        raise ValueError("new evidence exceeds unchanged reader input token budget")
    new["input_sha256"] = digest({k: v for k, v in new.items() if k != "input_sha256"})
    return new


def require_bindings(bindings: dict) -> None:
    for name, bound in bindings.items():
        path = Path(bound["path"])
        if not path.is_file() or file_sha(path) != bound["sha256"]:
            raise ValueError(f"frozen artifact drift: {name}: {path}")


def require_asset_stats(bindings: dict) -> None:
    """Assets are fully hashed at freeze; guard their identity/stat at runtime.

    This avoids repeatedly streaming the 50+ GB readonly retrieval assets.
    It is explicitly weaker than independently rehashing every asset each run.
    """
    for name, bound in bindings.items():
        path = Path(bound["path"])
        if not path.is_file() or stat_signature(path) != bound["stat_signature"]:
            raise ValueError(f"frozen retrieval asset stat drift: {name}")


def asset_paths() -> list[Path]:
    wiki = ROOT / "indexes_wiki18"
    paths = [wiki / "corpus_flashrag.jsonl", wiki / "corpus_flashrag.mmindex.json", wiki / "e5_fp16.dat"]
    paths += [wiki / "bm25" / n for n in ("data.csc.index.npy", "indices.csc.index.npy", "indptr.csc.index.npy",
              "params.index.json", "stopwords.tokenizer.json", "vocab.index.json", "vocab.tokenizer.json")]
    for name in ("e5-base-v2", "bge-reranker-v2-m3"):
        paths += [p for p in sorted((ROOT / "models" / name).rglob("*")) if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".model")]
    return paths


def prepare(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    parent_prepared = json.loads((PARENT / "prepared.json").read_text())
    require_bindings({"parent_protocol": parent_prepared["protocol"]})
    parent = json.loads((PARENT / "protocol.json").read_text())
    require_bindings({n: parent["frozen_artifacts"][n] for n in ("inputs.jsonl", "selection.question_only.jsonl")})
    rows = [r for r in read_rows(PARENT / "inputs.jsonl") if r["dataset"] == "musique"]
    selection = [r for r in read_rows(PARENT / "selection.question_only.jsonl") if r["dataset"] == "musique"]
    if len(rows) != 20 or {r["question_key"] for r in rows} != {r["question_key"] for r in selection}:
        raise ValueError("must retain exact consumed twenty MuSiQue questions")
    for row in rows:
        assert_gold_free(row)
        if digest({k: v for k, v in row.items() if k != "input_sha256"}) != row["input_sha256"]:
            raise ValueError("old input hash mismatch")
    plans = [{"question_key": r["question_key"], "question": r["question"], "legacy_input_sha256": r["input_sha256"],
              "original_passages_sha256": digest(r["retrieved_passages"]), **build_query_plan(r["question"], r["retrieved_passages"])} for r in rows]
    write_rows(directory / "legacy_inputs.jsonl", rows)
    write_rows(directory / "selection.question_only.jsonl", selection)
    write_rows(directory / "query_plan.jsonl", plans)
    shutil.copyfile(Path(__file__), directory / "probe.executed.py")
    code_paths = [ROOT / p for p in ("kgproweight/data/prompts.py", "kgproweight/retrieval/hybrid.py", "kgproweight/retrieval/reranker.py",
        "kgproweight/retrieval/np_search.py", "kgproweight/retrieval/bootstrap.py", "kgproweight/utils/paths.py",
        "kgproweight/utils/flashrag_bootstrap.py", "kgproweight/data/flashrag_loader.py")]
    code_paths += sorted((ROOT / "flashrag_src/flashrag/retriever").glob("*.py"))
    code_paths += sorted((ROOT / "flashrag_src/flashrag/config").glob("*.py"))
    code_paths += sorted((ROOT / "flashrag_src/flashrag/config").glob("*.yaml"))
    code_paths += sorted((ROOT / "flashrag_src/flashrag/utils").glob("*.py"))
    code_bindings = {str(p.relative_to(ROOT)): identity(p) for p in code_paths}
    print("Binding full Wiki18 corpus, dense/sparse indexes, E5/BGE files; no retrieval yet", flush=True)
    assets = {}
    paths = asset_paths() + [Path(parent["policy_path"]) / n for n in parent["models"]["policy_tokenizer"]["files"]]
    for path in paths:
        before = stat_signature(path)
        assets[str(path.relative_to(ROOT))] = {**identity(path), "stat_signature": before}
        if stat_signature(path) != before:
            raise ValueError("asset changed while calculating full SHA")
        print(f"BOUND {path.relative_to(ROOT)}", flush=True)
    from transformers import AutoTokenizer
    query_tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/e5-base-v2", local_files_only=True)
    query_lengths = [len(query_tokenizer("query: " + q["query"], truncation=False)["input_ids"]) for p in plans for q in p["queries"]]
    if query_lengths and max(query_lengths) > 128:
        raise ValueError("expanded query exceeds existing E5 query budget; refusing silent truncation")
    protocol = {"schema_version": VERSION, "experiment_id": EXPERIMENT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "consumed_train_development_supply_pilot_only", "root": str(ROOT), "parent_protocol": parent_prepared["protocol"],
        "parent_manifest": identity(PARENT / "manifest.json"), "policy_path": parent["policy_path"], "models": parent["models"],
        "generation_reference": parent["generation_384"], "frozen_runtime_snapshot_for_reader": parent["runtime_snapshot"],
        "frozen_artifacts": {n: identity(directory / n) for n in ("legacy_inputs.jsonl", "selection.question_only.jsonl", "query_plan.jsonl", "probe.executed.py")},
        "code_bindings": code_bindings, "retrieval_assets": assets,
        "algorithm": {"candidate_passages": "original top3 only; English capitalized name-string regex", "query": "original question + space + selected entity",
          "selection": "novel normalized phrase; exclude original question substrings and phrases containing existing question entities; descending passage-frequency, then first passage rank, first character offset, normalized string",
          "max_queries": 3, "dense_topk": 100, "sparse_topk": 100, "rrf_k": 60, "rrf_topk": 50,
          "reranker": "bge-reranker-v2-m3 on each expanded query, same max_chars=1200", "reranker_fallback": False,
          "final_selection": "preserve original first4 distinct document IDs and normalized full contents; round-robin unseen ID+content from each expanded ranking until10; fill shortage using original remaining order",
          "final_passages": 10, "passage_pack": "unchanged canonical pack_passages_by_token_budget3860; require10", "reader_max_input_tokens": 6144},
        "execution": {"seed": 42, "query_batch_size": 60, "e5_fp16": True, "bge_default_precision": "float32", "dense_index": "CPU readonly fp16 mmap", "corpus": "Wiki18 readonly JSONL mmap", "network": "offline", "torch_threads": 4,
          "asset_verification": "Full SHA256 once at protocol freeze; device/inode/bytes/mtime_ns checked at execution start/end; code SHA256 checked start/end. Not an independent full rehash of 50+GB at each stage."},
        "planned_counts": {"questions": 20, "expanded_queries": len(query_lengths), "max_e5_query_tokens": max(query_lengths, default=0),
                           "hybrid_query_count_histogram": dict(Counter(p["upstream_hybrid_queries"] for p in plans)), "reader_candidates_later": 40},
        "scientific_boundary": {"gold_access": False, "raw_qa_read": False, "official_support_read": False, "semantic_annotation_read": False,
          "generated_answer_read": False, "per_question_manual_rules": False, "llm_query_planner": False, "final_reader_budget_matched": True,
          "upstream_retrieval_budget_matched": False, "baseline_evaluation_changed": False, "original_data_modified": False,
          "kg_changed": False, "reward_changed": False, "optimizer_updates": 0, "independent_confirmation": False, "ppo_launch_clearance": False,
          "scope_limit": "One heuristic on already consumed20, not a general retrieval fix or post-PPO evaluation; all20 retained, no answer-driven retries or question substitution."}}
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "prepared.json", {"protocol": identity(directory / "protocol.json"), "status": "FROZEN_RETRIEVAL_NOT_STARTED"})
    return protocol


def verify(directory: Path, *, assets: bool) -> dict:
    prepared = json.loads((directory / "prepared.json").read_text())
    require_bindings({"protocol": prepared["protocol"]})
    p = json.loads((directory / "protocol.json").read_text())
    for group in ("frozen_artifacts", "code_bindings"):
        require_bindings(p[group])
    require_bindings({"parent_protocol": p["parent_protocol"], "parent_manifest": p["parent_manifest"]})
    if assets:
        require_asset_stats(p["retrieval_assets"])
    return p


def build_backend(directory: Path):
    os.environ.update({"KGPW_CORPUS_MMAP": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    from kgproweight.retrieval.hybrid import build_flashrag_config, build_rrf_setting
    from kgproweight.data.flashrag_loader import flashrag_config
    from kgproweight.retrieval.reranker import get_cross_encoder
    from flashrag.utils import get_retriever
    wiki = ROOT / "indexes_wiki18"
    config = build_flashrag_config("musique", "evidence_supply_v1", str(directory / "runtime"), topk=50,
                 corpus_path=str(wiki / "corpus_flashrag.jsonl"), seed=42)
    config.update({"index_path": str(wiki / "e5_fp16.dat"), "bm25_index_path": str(wiki / "bm25"),
                   "use_retrieval_cache": False, "save_retrieval_cache": False,
                   "multi_retriever_setting": build_rrf_setting(topk=50, dense_index_path=str(wiki / "e5_fp16.dat"),
                       sparse_index_path=str(wiki / "bm25"), dense_model_path=str(ROOT / "models/e5-base-v2"),
                       corpus_path=str(wiki / "corpus_flashrag.jsonl"))})
    router = get_retriever(flashrag_config(config))
    if [r.retrieval_method for r in router.retriever_list] != ["e5", "bm25"]:
        raise ValueError("unexpected retrieval branches")
    if len(router.retriever_list[0].corpus) != 21015324:
        raise ValueError("must execute full Wiki18; smoke index is forbidden")
    ce = get_cross_encoder(str(ROOT / "models/bge-reranker-v2-m3"))
    if ce is None:
        raise ValueError("BGE unavailable; fallback is forbidden")
    return router, ce, config


def run(directory: Path) -> dict:
    start = time.monotonic()
    os.environ.update({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    p = verify(directory, assets=True)
    if identity(Path(__file__)) != p["frozen_artifacts"]["probe.executed.py"]:
        raise ValueError("execute the frozen producer copy")
    if (directory / "retrieval_started.json").exists():
        raise FileExistsError("no overwrite or retry of retrieval execution")
    write_json(directory / "retrieval_started.json", {"protocol": identity(directory / "protocol.json"), "started_at_utc": datetime.now(timezone.utc).isoformat()})
    import torch
    import numpy as np
    import transformers
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA required for the frozen E5/BGE pilot")
    from kgproweight.retrieval.reranker import pack_passages_by_token_budget
    plans = read_rows(directory / "query_plan.jsonl")
    inputs = read_rows(directory / "legacy_inputs.jsonl")
    if any(build_query_plan(r["question"], r["retrieved_passages"]) != {k: v for k, v in plan.items()
          if k not in ("question_key", "question", "legacy_input_sha256", "original_passages_sha256")} for r, plan in zip(inputs, plans)):
        raise ValueError("frozen query plan cannot be reproduced from allowed evidence")
    router, ce, config = build_backend(directory)
    flat = [{"question_key": plan["question_key"], **q} for plan in plans for q in plan["queries"]]
    questions = [q["query"] for q in flat]
    write_json(directory / "execution_environment.json", {"torch": torch.__version__, "transformers": transformers.__version__,
         "numpy": np.__version__, "gpu": torch.cuda.get_device_name(), "python": sys.executable,
         "packages": {n: importlib.metadata.version(n) for n in ("sentence-transformers", "bm25s", "tokenizers")},
         "bge_max_length": ce.max_length, "bge_parameter_dtype": str(next(ce.model.parameters()).dtype),
         "bge_device": str(ce.device), "bge_tokenizer_model_max_length": ce.tokenizer.model_max_length,
         "retrieval_config": config, "loaded_reranker_module": str(Path(sys.modules["kgproweight.retrieval.reranker"].__file__).resolve()),
         "loaded_flashrag_retriever_module": str(Path(sys.modules["flashrag.retriever.retriever"].__file__).resolve()),
         "branches": [{"method": r.retrieval_method, "index_path": r.index_path, "corpus_path": r.corpus_path} for r in router.retriever_list],
         "fallback": False, "gold_access": False})
    branches = []
    for retriever in router.retriever_list:
        t = time.monotonic()
        print(f"SEARCH {retriever.retrieval_method} queries={len(questions)}", flush=True)
        values = retriever.batch_search(questions, num=100)
        if len(values) != len(questions) or any(len(ds) != 100 for ds in values):
            raise ValueError("each retrieval branch must return 100 for every frozen query")
        branches.append(router.add_source(values, retriever))
        print(f"SEARCH_DONE {retriever.retrieval_method} seconds={time.monotonic()-t:.3f}", flush=True)
    merged, rrf_scores = router.rrf_merge([d + s for d, s in zip(*branches)], topk=50, k=60)
    by_question = {r["question_key"]: [] for r in inputs}
    with (directory / "retrieval_calls.jsonl").open("x") as trace:
        for i, (query, candidates) in enumerate(zip(flat, merged)):
            assert_gold_free(candidates)
            pairs = [(query["query"], passage_text(c)[:1200]) for c in candidates]
            scores = [float(s) for s in ce.predict(pairs, show_progress_bar=False)]
            if len(scores) != 50 or any(not np.isfinite(s) for s in scores):
                raise ValueError("reranker must return 50 finite values")
            order = sorted(range(50), key=lambda j: scores[j], reverse=True)
            ranked = [deepcopy(candidates[j]) for j in order]
            by_question[query["question_key"]].append(ranked)
            record = {"question_key": query["question_key"], "query_index": query["query_index"], "query": query["query"],
                      "query_sha256": digest(query), "gold_access": False, "dense": branches[0][i], "sparse": branches[1][i],
                      "rrf_candidates": candidates, "rrf_scores": [float(s) for s in rrf_scores[i]],
                      "bge_scores": scores, "bge_sorted_indices": order}
            trace.write(canonical(record) + "\n"); trace.flush()
            print(f"RERANK_DONE {i+1}/{len(flat)}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(p["policy_path"], local_files_only=True)
    contexts, rebound, changes = [], [], []
    for old in inputs:
        passages, selected = select_passages(old["retrieved_passages"], by_question[old["question_key"]])
        packed = pack_passages_by_token_budget(passages, 3860)
        if len(packed) != 10:
            raise ValueError("canonical pack violates final ten-passage budget")
        new = rebind_input(old, packed, tokenizer)
        rebound.append(new)
        contexts.append({"question_key": old["question_key"], "question": old["question"], "dataset": old["dataset"], "qid": old["qid"],
              "family_sha256": old["family_sha256"], "question_sha256": old["question_sha256"], "gold_access": False,
              "passages": packed, "passages_sha256": digest(packed), "selected_passage_origins": selected})
        changes.append({"question_key": old["question_key"], "legacy_input_sha256": old["input_sha256"], "new_input_sha256": new["input_sha256"],
              "legacy_passages_sha256": digest(old["retrieved_passages"]), "new_passages_sha256": digest(packed),
              "preserved_first_four_unique": len([v for v in selected if v["origin"] == "legacy"]) == 4,
              "retained_original_ranks": [i for i,p in enumerate(old["retrieved_passages"], 1) if str(p["id"]) in {str(p2["id"]) for p2 in packed}],
              "removed_original_ranks": [i for i,p in enumerate(old["retrieved_passages"], 1) if str(p["id"]) not in {str(p2["id"]) for p2 in packed}],
              "system_unchanged": new["messages"][0] == old["messages"][0],
              "kg_unchanged": new["kg_subgraph"] == old["kg_subgraph"], "legacy_prompt_tokens": old["prompt_tokens"],
              "new_prompt_tokens": new["prompt_tokens"], "new_document_count": sum(v["origin"].startswith("expanded") for v in selected)})
    write_rows(directory / "retrieval_contexts.jsonl", contexts)
    write_rows(directory / "inputs.jsonl", rebound)
    write_rows(directory / "input_changes.jsonl", changes)
    print("Verifying frozen code SHA and full-hash-bound corpus/index/model file stats after retrieval", flush=True)
    verify(directory, assets=True)
    report = {"schema_version": VERSION, "experiment_id": EXPERIMENT, "status": "COMPLETE_DEVELOPMENT_ONLY",
              "questions": len(rebound), "hybrid_queries": len(flat), "dense_query_calls": len(flat), "sparse_query_calls": len(flat),
              "reranker_pairs": len(flat)*50, "new_document_count": sum(r["new_document_count"] for r in changes),
              "new_document_count_histogram": dict(Counter(r["new_document_count"] for r in changes)),
              "old_prompt_tokens": sum(r["legacy_prompt_tokens"] for r in changes), "new_prompt_tokens": sum(r["new_prompt_tokens"] for r in changes),
              "max_prompt_tokens": max(r["new_prompt_tokens"] for r in changes), "all_exactly_ten": all(len(r["retrieved_passages"]) == 10 for r in rebound),
              "all_original_first4_unique_retained": all(r["preserved_first_four_unique"] for r in changes), "all_system_and_kg_unchanged": all(r["system_unchanged"] and r["kg_unchanged"] for r in changes),
              "full_assets_hashed_at_freeze_stat_verified_start_end": True, "code_sha_verified_start_end": True, "elapsed_seconds": time.monotonic()-start,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated()/1024**3, "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
              "gold_access": False, "optimizer_updates": 0, "ppo_launch_clearance": False, "upstream_budget_matched": False}
    write_json(directory / "report.json", report)
    files = ["protocol.json", "prepared.json", "retrieval_started.json", "execution_environment.json", "retrieval_calls.jsonl",
             "retrieval_contexts.jsonl", "inputs.jsonl", "input_changes.jsonl", "report.json"]
    write_json(directory / "manifest.json", {**report, "outputs": {n: identity(directory / n) for n in files}})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.out) if args.stage == "prepare" else run(args.out) if args.stage == "run" else verify(args.out, assets=True)
    print(json.dumps({"status": result.get("status", "FROZEN"), "questions": result.get("questions", 20), "experiment_id": EXPERIMENT}, indent=2), flush=True)


if __name__ == "__main__":
    main()
