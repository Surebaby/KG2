"""Synthetic CPU tests. No fresh generations or Gold are read by this suite."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.pilot import analyze_source_credit_v2_fresh_confirmation_v1 as a
from scripts.prepare.score_source_credit_v2_fresh_confirmation_v1 import rank_questions


def estimate(point, low=None, high=None):
    return {"point": point, "ci95": [point if low is None else low, point if high is None else high], "families": 96}


def decision_fixture():
    pair = {**estimate(.70, .50, .86), "mixed_outcome_families": 30}
    return {"all132_ITT": {"metrics": {"sampled_valid": estimate(.92,.86,.98)}},
            "three_dataset_macro": {"sampled_valid": estimate(.93,.85,.99)},
            "graph96_ITT": {"metrics": {"oracle_minus_greedy_em": estimate(.1,.02,.18),
                "features_v2_A_minus_greedy_em": estimate(.01,-.06,.08),
                "features_v2_A_minus_F_em": estimate(0,-.05,.05)}},
            "source_PASS79": {"pairwise": {"features_v2": {"A": pair}}}}


@pytest.mark.parametrize("point,low,high,target,expected", [
    (.9,.8,.98,.9,"PASS"), (0,-.1,.1,0,"PASS"), (.8,.7,.89,.9,"FAIL"),
    (.8,.7,.9,.9,"INCONCLUSIVE"), (.8,.7,.95,.9,"INCONCLUSIVE"),
    (None,None,None,.9,"INCONCLUSIVE")])
def test_predeclared_point_ci_three_states(point, low, high, target, expected):
    assert a.target_decision(estimate(point,low,high),target)["status"] == expected


def test_no_significant_advantage_required_and_no_norm_only_selection():
    result = a.decide(decision_fixture())
    assert result["overall_status"] == result["independent_utility_status"] == "PASS"
    assert result["engineering_probe_eligibility"]
    assert result["matched600_investment_clearance"]
    assert result["secondary_cannot_rescue_primary"]


def test_health_failure_is_separate_from_confirmed_utility():
    summary = decision_fixture()
    summary["all132_ITT"]["metrics"]["sampled_valid"] = estimate(.8,.7,.88)
    result = a.decide(summary)
    assert result["overall_status"] == result["health_status"] == "FAIL"
    assert result["independent_utility_status"] == "PASS"
    assert result["engineering_probe_eligibility"]
    assert not result["matched600_investment_clearance"]


def test_health_uncertain_does_not_forge_full_investment_clearance():
    summary = decision_fixture()
    summary["all132_ITT"]["metrics"]["sampled_valid"] = estimate(.88,.80,.96)
    result = a.decide(summary)
    assert result["overall_status"] == "INCONCLUSIVE"
    assert result["independent_utility_status"] == "PASS"
    assert result["engineering_probe_eligibility"] and not result["matched600_investment_clearance"]


@pytest.mark.parametrize("kind", ["mixed_families", "oracle"])
def test_information_shortfall_suppresses_degenerate_utility_failure(kind):
    summary = decision_fixture()
    if kind == "mixed_families":
        summary["source_PASS79"]["pairwise"]["features_v2"]["A"]["mixed_outcome_families"] = 24
    else:
        summary["graph96_ITT"]["metrics"]["oracle_minus_greedy_em"] = estimate(.02,.01,.025)
    summary["source_PASS79"]["pairwise"]["features_v2"]["A"].update(estimate(.1,0,.2))
    result = a.decide(summary)
    assert result["independent_utility_status"] == "INCONCLUSIVE"
    assert result["utility"]["source_pass_A_pairwise"]["point_ci_status_diagnostic"] == "FAIL"
    assert not result["engineering_probe_eligibility"]


def test_primary_utility_failure_cannot_be_rescued_by_other_variant():
    summary = decision_fixture()
    summary["graph96_ITT"]["metrics"]["features_v2_A_minus_F_em"] = estimate(-.1,-.2,-.01)
    summary["source_PASS79"]["pairwise"]["norm_only"] = {"A": estimate(1)}
    assert a.decide(summary)["independent_utility_status"] == "FAIL"


def test_rules_return_defensive_copy():
    rules = a.decision_rules()
    rules["bootstrap"]["seed"] = 99
    assert a.decision_rules()["bootstrap"]["seed"] == 42


def test_pairwise_ties_and_family_before_pair_micro():
    rows = [{"candidate_id": f"c{i}", "variants": {"features_v2": {"A": {"process": p}}}} for i,p in enumerate([.5,.5,.2,.9])]
    scores = {f"c{i}": {"em": int(i in [0,2])} for i in range(4)}
    out = a.pairwise(rows,scores,"features_v2","A")
    assert out == {"family_accuracy": .125,"correct_incorrect_pairs":4,"wins":0,"ties":1}
    assert a.pairwise(rows,{f"c{i}":{"em":1} for i in range(4)},"features_v2","A")["family_accuracy"] is None


def test_bootstrap_is_family_paired_deterministic_and_rejects_duplicates():
    rows = [{"family_sha256": str(i), "bootstrap_stratum": f"s{i//2}", "dataset": "x", "A": float(i), "F":float(i), "delta":0.} for i in range(8)]
    one = a.bootstrap_estimates(rows,["A","F","delta"],replicates=2000)
    assert one == a.bootstrap_estimates(rows,["A","F","delta"],replicates=2000)
    assert one["A"] == one["F"]
    assert one["delta"]["ci95"] == [0.,0.]
    with pytest.raises(ValueError,match="unique family"):
        a.bootstrap_estimates(rows+[rows[0]],["A"])


def test_dataset_macro_weights_do_not_equal_graph_heavy_micro():
    rows = []
    for domain,n,value in (("hotpotqa",1,0.),("musique",1,0.),("2wikimultihopqa",8,1.)):
        rows.extend({"family_sha256":f"{domain}{i}","bootstrap_stratum":domain,"dataset":domain,"valid":value} for i in range(n))
    assert a.bootstrap_estimates(rows,["valid"],replicates=100)["valid"]["point"] == pytest.approx(.8)
    assert a.bootstrap_estimates(rows,["valid"],replicates=100,macro_dataset=True)["valid"]["point"] == pytest.approx(1/3)


@pytest.mark.parametrize("generation,surfaces,em,f1", [
    ("[Final Answer]\nThe New-York.",["NewYork"],1.,1.),
    ("[Final Answer]\nAlias",["Primary","alias"],1.,1.),
    ("[Final Answer]\n",["Something"],0.,0.),
    ("[Final Answer]\nno",["yes"],0.,0.)])
def test_canonical_double_extractor_and_max_alias(generation,surfaces,em,f1):
    assert a.answer_scores(generation,surfaces) == {"em":em,"f1":f1}


def fixture132():
    cohort,inputs,checks,generations,processes = [],[],[],[],[]
    specs = []
    for kind,counts in (("bridge_comparison",(18,9,5)),("comparison",(30,1,1)),("compositional",(31,1,0))):
        specs.extend(("2wikimultihopqa","graph",kind,status) for status,n in zip(("PASS","UNVERIFIED","FAIL"),counts) for _ in range(n))
    specs.extend((d,"ordinary","", "ORDINARY") for d in a.DATASETS for _ in range(12))
    for n,(domain,role,kind,status) in enumerate(specs):
        qid=str(n); key=f"{domain}::{qid}"; question=f"Synthetic question {n}?"
        identity={"dataset":domain,"qid":qid,"question_key":key,"question":question,
                  "question_sha256":hashlib.sha256(question.encode()).hexdigest(),"family_sha256":hashlib.sha256(f"family{n}".encode()).hexdigest(),"family_version":"answer-free-lexical-family-v1"}
        cohort.append({**identity,"proposal_role":role,"question_type":kind,"gold_access":False})
        original={**identity,"input_sha256":f"input{n}","source_record_sha256":f"source{n}","m_graph":int(role=="graph"),"spec":{}}
        inputs.append(original)
        if role=="graph": checks.append({"question_key":key,"status":status,"input_sha256":original["input_sha256"],"gold_used":False,"original_m_graph":1,"clearance":status=="PASS"})
        for i in range(5):
            pred={"candidate_id":f"{key}::k{i}","candidate_index":i,"generation_kind":"sampled" if i<4 else "greedy",
                  "generation":"[Final Answer]\n"+("Gamma" if i in (0,2) else "Wrong"),"seed":42,"n_response_tokens":30,"reached_max_new_tokens":False}
            generations.append(pred)
            features={v:{"masked":{"m_graph":int(status=="PASS")}} for v in a.VARIANTS}
            variants={}
            for v in a.VARIANTS:
                variants[v]={}
                for arm in a.ARMS:
                    alpha=.5 if status=="PASS" and arm!="T" else 0.
                    tn=[.8,.2,.6,.1,.99][i]; text=.3*(1-alpha)*tn
                    variants[v][arm]={"alpha_effective":alpha,"text_component":text,"graph_component":0.,"process":text,
                        "text_step_components":[text],"rank_eligible":True,"text_normalized_mean":tn,"text_normalized_steps":[tn],"graph_normalized":0.,"graph_normalized_unclipped":0.}
            row={"schema_version":"source-credit-v2-fresh-process-row-v1",**original,**pred,
                 "protocol_sha256":"protocol","generation_sha256":a.digest(pred),"trajectory_valid":True,"rank_eligible":True,
                 "raw_graph":.4,"raw_text":[.4],"variants":variants,"features":features,"gold_access":False,"outcome_in_process":False,"model_updates":0}
            row["process_row_sha256"]=a.digest(row); processes.append(row)
    bykey, checkmap=a.verify_population(cohort,inputs,checks)
    grouped=a.verify_process_rows(processes,generations,inputs,bykey,checkmap,protocol_sha256="protocol")
    rankings=rank_questions(processes)
    ranked=a.verify_rankings(rankings,grouped)
    return {"cohort":bykey,"checks":checkmap,"grouped":grouped,"rankings":ranked},processes,generations,inputs,cohort,checks


@pytest.fixture(scope="module")
def complete():
    return fixture132()


def test_complete132_and_660_membership_and_process_ranking(complete):
    context,*_=complete
    assert len(context["cohort"])==132
    assert sum(len(rows) for rows in context["grouped"].values())==660
    assert all(rank["rankings"]["features_v2"]["A"]["selected_candidate_id"].endswith("::k0") for rank in context["rankings"].values())


def test_resealed_wrong_rank_or_greedy_never_passes(complete):
    context,*_=complete
    ranks=deepcopy(list(context["rankings"].values()))
    ranks[0]["rankings"]["features_v2"]["A"]["selected_candidate_id"]=ranks[0]["greedy_candidate_id"]
    with pytest.raises(ValueError,match="independent process-only"):
        a.verify_rankings(ranks,context["grouped"])


@pytest.mark.parametrize("change",["nan","extra_outcome","nonpass_alpha","bad_hash","duplicate"])
def test_process_integrity_fail_closed(complete,change):
    context,processes,generations,inputs,_,_=complete
    p=deepcopy(processes)
    target=p[18*5] if change=="nonpass_alpha" else p[0]
    if change=="nan":
        target["variants"]["features_v2"]["A"]["process"]=float("nan")
    elif change=="extra_outcome": target["outcome_in_process"]=True
    elif change=="nonpass_alpha": target["variants"]["features_v2"]["A"]["alpha_effective"]=.3
    elif change=="bad_hash": target["generation"]="changed"
    else: p[-1]=deepcopy(p[0])
    if change not in ("bad_hash","nan","duplicate"):
        target["process_row_sha256"]=a.digest({k:v for k,v in target.items() if k!="process_row_sha256"})
    with pytest.raises(ValueError):
        a.verify_process_rows(p,generations,inputs,context["cohort"],context["checks"],protocol_sha256="protocol")


def test_all_invalid_retained_and_raw_greedy_included(complete):
    context=deepcopy(complete[0]); key=next(iter(context["cohort"]))
    for row in context["grouped"][key][:4]: row["trajectory_valid"]=False
    for variant in a.VARIANTS:
        for arm in a.ARMS: context["rankings"][key]["rankings"][variant][arm]["selected_candidate_id"]=None
    context["grouped"][key][4]["trajectory_valid"]=False
    context["grouped"][key][4]["generation"]="[Final Answer]\nGamma"
    questions,_=a.question_metrics(context,{k:["Gamma"] for k in context["cohort"]})
    q=next(r for r in questions if r["question_key"]==key)
    assert q["all_sampled_invalid"]==1 and q["features_v2_A_em"]==q["valid_oracle_em"]==0
    assert q["raw_greedy_em"]==1 and q["format_gated_greedy_em"]==0


def test_full_statistical_analysis_uses_primary_and_all_strata(complete):
    context=complete[0]
    questions,candidates,summaries,decision=a.analyze(context,{k:["Gamma"] for k in context["cohort"]})
    assert len(questions)==132 and len(candidates)==660
    assert decision["independent_utility_status"]=="PASS" and decision["overall_status"]=="PASS"
    assert summaries["source_PASS79"]["pairwise"]["features_v2"]["A"]["mixed_outcome_families"]==79
    assert summaries["graph_type_bridge_comparison_source_PASS"]["questions"]==18
    assert summaries["graph_source_FAIL"]["questions"]==6
    assert summaries["ordinary36_diagnostic"]["questions"]==36


def make_gold_context(tmp_path,complete):
    context=deepcopy(complete[0]); refs={}
    for role in ("graph","ordinary"):
        path=tmp_path/f"{role}.jsonl"
        with path.open("w") as f:
            for r in context["cohort"].values():
                if r["proposal_role"]!=role: continue
                f.write(json.dumps({k:r[k] for k in ("dataset","qid","question")} | {"metadata":{"gold_answer":"Gamma","gold_answer_aliases":["Alias"],**({"source_split":"train"} if role=="ordinary" else {})}})+"\n")
        refs[role]=a.binding(path)
    sentinel=tmp_path/"sealed.txt"; sentinel.write_text("before-labels")
    context.update({"seal":{"status":"PASS_BEFORE_GOLD"},"frozen_files":{"sentinel":a.binding(sentinel)},"protocol":{"analysis":{"gold_sources":refs}}})
    return context


def test_gold_requires_seal_and_exact_readonly_join(tmp_path,complete):
    context=make_gold_context(tmp_path,complete)
    labels=a.load_gold_after_seal(context)
    assert len(labels)==132 and all(v==["Gamma","Alias"] for v in labels.values())
    context["seal"]["status"]="NOT_SEALED"
    with pytest.raises(ValueError,match="rank seal"): a.load_gold_after_seal(context)


@pytest.mark.parametrize("fault",["mutation","duplicate","question","ordinary_split"])
def test_gold_source_and_seal_changes_reject(tmp_path,complete,fault):
    context=make_gold_context(tmp_path,complete)
    if fault=="mutation":
        Path(context["frozen_files"]["sentinel"]["path"]).write_text("modified")
    else:
        role="ordinary" if fault=="ordinary_split" else "graph"
        path=Path(context["protocol"]["analysis"]["gold_sources"][role]["path"])
        rows=a.read_rows(path)
        if fault=="duplicate": rows.append(rows[0])
        elif fault=="question": rows[0]["question"]+=" changed"
        else: rows[0]["metadata"]["source_split"]="dev"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        context["protocol"]["analysis"]["gold_sources"][role]=a.binding(path)
    with pytest.raises(ValueError): a.load_gold_after_seal(context)


def test_run_records_start_and_never_opens_gold_before_failed_seal(tmp_path,monkeypatch):
    opened=[]
    monkeypatch.setattr(a,"verify_before_gold",lambda *args: (_ for _ in ()).throw(ValueError("broken seal")))
    monkeypatch.setattr(a,"load_gold_after_seal",lambda *args: opened.append(True))
    out=tmp_path/"release"
    with pytest.raises(ValueError): a.run(protocol="unused",scoring="unused",out=out)
    assert not opened
    assert json.loads((out/"started.json").read_text())["automatic_reanalysis_allowed"] is False
    assert json.loads((out/"failed.json").read_text())["gold_boundary_entered"] is False
    with pytest.raises(ValueError,match="never overwrite"): a.run(protocol="unused",scoring="unused",out=out)


def test_full_output_io_reads_synthetic_gold_only_after_saved_seal(tmp_path,complete,monkeypatch):
    context=make_gold_context(tmp_path,complete)
    context["protocol_sha256"]="synthetic-protocol"
    context["protocol"]["experiment_id"]="SYNTHETIC-CPU-TEST"
    monkeypatch.setattr(a,"verify_before_gold",lambda *args:context)
    original=a.load_gold_after_seal
    out=tmp_path/"analysis"
    def read_after_marker(ctx):
        assert (out/"before_gold.json").exists() and (out/"started.json").exists()
        return original(ctx)
    monkeypatch.setattr(a,"load_gold_after_seal",read_after_marker)
    result=a.run(protocol="synthetic",scoring="synthetic",out=out)
    assert result["status"]=="PASS"
    report=json.loads((out/"report.json").read_text())
    assert report["independent_utility_status"]=="PASS" and report["engineering_probe_eligibility"]
    assert report["model_updates"]==0 and not report["parent_gate_modified"]
    for path in out.iterdir():
        assert "Gamma" not in path.read_text() and '"Alias"' not in path.read_text()
    manifest=json.loads((out/"manifest.json").read_text())
    for ref in manifest["outputs"].values(): a.checked(ref)
