"""Check the actual dense top100 ordering before attributing a retrieval defect.

This readonly, Gold-free diagnostic reuses the frozen evidence-supply assets.
It only queries the twenty already consumed original questions.  No proposed
ranking repair, expanded query, reranker or Reader is executed by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ.get("KGPW_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SUPPLY = ROOT / "outputs/audits/evidence_supply_v1_consumed20_20260906_v1"
OUTPUT = ROOT / "outputs/audits/dense_rank_contract_consumed20_20260906_v1"
VERSION = "dense-rank-contract-consumed20-v1"
EXPERIMENT = "DENSE-RANK-CONTRACT-CONSUMED20-20260906-V1"


def helper():
    path = SUPPLY / "probe.executed.py"
    spec = importlib.util.spec_from_file_location("frozen_evidence_supply_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rank_diagnostic(documents, scores):
    import math
    if len(documents) != len(scores) or not documents or any(not math.isfinite(s) for s in scores):
        raise ValueError("same nonempty number of documents and finite scores required")
    if len({str(d["id"]) for d in documents}) != len(documents):
        raise ValueError("duplicate dense document ID")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], str(documents[i]["id"])))
    strict_inversions = [i + 1 for i in range(len(scores) - 1) if scores[i] < scores[i + 1]]
    return {"returned_descending": not strict_inversions, "adjacent_inversion_after_ranks": strict_inversions,
            "adjacent_exact_score_ties": sum(a == b for a,b in zip(scores, scores[1:])),
            "stable_score_desc_docid_tie_order": order, "stable_sort_changes_order": order != list(range(len(scores))),
            "returned_top1_is_best_score": scores[0] == max(scores)}


def prepare(directory):
    h = helper()
    p = h.verify(SUPPLY, assets=True)
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SUPPLY / "legacy_inputs.jsonl", directory / "legacy_inputs.jsonl")
    shutil.copyfile(Path(__file__), directory / "probe.executed.py")
    rows = h.read_rows(directory / "legacy_inputs.jsonl")
    if len(rows) != 20:
        raise ValueError("exact consumed20 required")
    questions = []
    for row in rows:
        h.assert_gold_free(row)
        questions.append({"question_key": row["question_key"], "question": row["question"], "legacy_input_sha256": row["input_sha256"]})
    h.write_rows(directory / "queries.question_only.jsonl", questions)
    protocol = {"schema_version": VERSION, "experiment_id": EXPERIMENT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
       "supply_protocol": h.identity(SUPPLY / "protocol.json"), "helper_code": h.identity(Path(h.__file__)),
       "asset_verification": p["execution"]["asset_verification"],
       "frozen_artifacts": {n:h.identity(directory/n) for n in ("legacy_inputs.jsonl", "queries.question_only.jsonl", "probe.executed.py")},
       "protocol": {"queries": "all20 original questions verbatim", "retrieval": "existing fullWiki18 E5 fp16-memmap top100, return_score=True",
                    "repair_applied": False, "counterfactual_order": "score descending then string docid ascending; only compare returned100 ranks, no new retrieval candidates",
                    "scope": "Current platform numerical contract diagnostic. No claim that historical baseline or formal data were affected absent direct evidence.",
                    "next_if_all_monotone_and_unchanged": "Record issue not reproduced in current environment; return to frozen evidence expansion pilot.",
                    "next_if_changed": "Record exact score/rank changes; separately preregister RRF/BGE and reader comparison before a scientific repair claim."},
       "seed":42, "dtype":"existing E5 use_fp16=True; index fp16 casttofloat32 dot product", "query_max_tokens":128,
       "gold_access":False, "reranker_loaded":False, "reader_loaded":False, "optimizer_updates":0, "ppo_launch_clearance":False}
    h.write_json(directory / "protocol.json", protocol)
    h.write_json(directory / "prepared.json", {"protocol":h.identity(directory / "protocol.json")})
    return protocol


def verify(directory):
    h = helper()
    h.require_bindings(json.loads((directory / "prepared.json").read_text()))
    p = json.loads((directory / "protocol.json").read_text())
    h.require_bindings(p["frozen_artifacts"])
    h.require_bindings({k:p[k] for k in ("supply_protocol", "helper_code")})
    h.verify(SUPPLY, assets=True)
    return p


def run(directory):
    start = time.monotonic()
    os.environ.update({"KGPW_CORPUS_MMAP":"1", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                       "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"4"})
    h = helper()
    p = verify(directory)
    if h.identity(Path(__file__)) != p["frozen_artifacts"]["probe.executed.py"]:
        raise ValueError("must execute frozen producer")
    h.write_json(directory / "started.json", {"protocol": h.identity(directory/"protocol.json"), "started_at_utc":datetime.now(timezone.utc).isoformat()})
    import torch
    import numpy as np
    import transformers
    torch.set_num_threads(4); torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen E5")
    from kgproweight.retrieval.hybrid import build_flashrag_config
    from kgproweight.data.flashrag_loader import flashrag_config
    from flashrag.utils import get_retriever
    wiki=ROOT/"indexes_wiki18"
    config=build_flashrag_config("musique", "dense_rank_contract_v1", str(directory/"runtime"),topk=100,
        use_multi_retriever=False,corpus_path=str(wiki/"corpus_flashrag.jsonl"),seed=42,
        extra={"index_path":str(wiki/"e5_fp16.dat"),"retrieval_model_path":str(ROOT/"models/e5-base-v2"),
               "use_retrieval_cache":False,"save_retrieval_cache":False})
    retriever=get_retriever(flashrag_config(config))
    if retriever.retrieval_method!="e5" or len(retriever.corpus)!=21015324:
        raise ValueError("must use E5 over fullWiki18")
    questions=h.read_rows(directory/"queries.question_only.jsonl")
    print("DENSE_RANK_AUDIT_SEARCH original_queries=20 topk=100", flush=True)
    documents,scores=retriever.batch_search([q["question"] for q in questions],num=100,return_score=True)
    if len(documents)!=20 or len(scores)!=20 or any(len(ds)!=100 for ds in documents):
        raise ValueError("must retain all20 top100 lists")
    rows=[]
    for query,docs,values in zip(questions,documents,scores):
        h.assert_gold_free(docs)
        vals=[float(v) for v in values]
        rows.append({**query,"documents":docs,"dense_scores":vals,**rank_diagnostic(docs,vals)})
    h.write_rows(directory/"dense_rank_diagnostics.jsonl",rows)
    h.write_json(directory/"execution_environment.json",{"python":sys.executable,"numpy":np.__version__,"torch":torch.__version__,
        "transformers":transformers.__version__,"gpu":torch.cuda.get_device_name(),"retrieval_config":config,
        "e5_dtype":str(next(retriever.encoder.model.parameters()).dtype),"index_type":type(retriever.index).__name__,
        "reranker_loaded":False,"reader_loaded":False,"gold_access":False})
    verify(directory)
    inversions=sum(not r["returned_descending"] for r in rows)
    changes=sum(r["stable_sort_changes_order"] for r in rows)
    report={"schema_version":VERSION,"experiment_id":EXPERIMENT,"status":"COMPLETE_DEVELOPMENT_ONLY",
        "result":"NO_ORDERING_DEFECT_REPRODUCED_ON_CURRENT20" if not(inversions or changes) else "RETURNED_ORDER_DIFFERS_FROM_STABLE_SORT",
        "questions":20,"topk":100,"nonmonotone_questions":inversions,"changed_order_questions":changes,
        "adjacent_inversion_count":sum(len(r["adjacent_inversion_after_ranks"]) for r in rows),
        "top1_not_max_score_questions":sum(not r["returned_top1_is_best_score"] for r in rows),
        "elapsed_seconds":time.monotonic()-start,"peak_cuda_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
        "full_asset_SHA_reused_from_frozen_supply":True,"asset_stats_and_code_SHA_verified_start_end":True,
        "gold_access":False,"original_retriever_modified":False,"reranker_loaded":False,"reader_loaded":False,"optimizer_updates":0}
    h.write_json(directory/"report.json",report)
    files=["protocol.json","prepared.json","started.json","dense_rank_diagnostics.jsonl","execution_environment.json","report.json"]
    h.write_json(directory/"manifest.json",{**report,"outputs":{n:h.identity(directory/n) for n in files}})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run"))
    parser.add_argument("--out",type=Path,default=OUTPUT)
    args=parser.parse_args()
    result=prepare(args.out) if args.stage=="prepare" else run(args.out)
    print(json.dumps(result if args.stage=="run" else {"status":"FROZEN","experiment_id":EXPERIMENT},indent=2),flush=True)


if __name__=="__main__":
    main()
