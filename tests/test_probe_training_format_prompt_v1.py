from copy import deepcopy
from types import SimpleNamespace

import pytest

from kgproweight.data import training_format_prompt_v1 as prompt
from kgproweight.data.prompts import build_sft_messages
from scripts.pilot import probe_training_format_prompt_v1 as probe


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["role"]+":"+m["content"] for m in messages)
    def __call__(self,text,**kwargs):
        return {"input_ids":list(range(len(text)))}


def row():
    old = {"question_key":"hotpotqa::synthetic","dataset":"hotpotqa","qid":"synthetic",
           "messages":build_sft_messages(question="Which city?",retrieved_passages=["Literal evidence.\nLine two."],kg_triples=[]),
           "spec":{"query":"Which city?","retrieved_passages":["Literal evidence.\nLine two."],"kg_subgraph":[],"metadata":{}},
           "kg_subgraph":[],"prompt":"old prompt","prompt_tokens":20,"m_graph":0,"source_bindings":{"safe":"frozen"}}
    old["input_sha256"]=probe.bank.input_hash(old)
    return old


def test_new_input_identity_changes_but_user_and_evidence_do_not():
    old=row();before=deepcopy(old)
    new,change=probe.make_input(old,prompt,Tokenizer())
    assert old==before
    assert new["messages"][1]==old["messages"][1]
    assert new["spec"]==old["spec"] and new["source_bindings"]==old["source_bindings"]
    assert new["input_sha256"]!=old["input_sha256"]
    assert probe.bank.input_hash(new)==new["input_sha256"]
    assert change["required_steps"]==3
    assert change["legacy_input_sha256"]==old["input_sha256"]
    assert change["new_input_sha256"]==new["input_sha256"]


def test_user_mutating_candidate_rejected():
    def bad(messages,**kwargs):
        result=deepcopy(messages);result[1]["content"]+="changed evidence";return result
    with pytest.raises(ValueError,match="user/evidence"):
        probe.make_input(row(),SimpleNamespace(clarify_training_format_messages_v1=bad),Tokenizer())


def test_repetition_diagnostic_does_not_repair_invalid_output():
    step="[Step 1]\nReasoning: This is a complete reason with sufficient characters.\nKnowledge Used: []\nConclusion: A factual conclusion.\n"
    result=probe.format_diagnostic(row(),step+step+"[Final Answer]\nCity\n[Final Answer]\nCity")
    assert result["duplicate_step_indices"]==1
    assert result["duplicate_exact_step_bodies"]==1
    assert result["final_marker_count"]==2
    assert result["valid"] is False


def test_paired_recovery_and_regression_are_both_retained():
    records=[]
    for i,(old,new) in enumerate([(False,True),(True,False),(True,True),(False,False)]):
        item={"question_key":str(i//2)}
        for arm,valid,tokens in [("legacy",old,100),("prompt_v1",new,110)]:
            item[f"format_{arm}"]={"valid":valid,"all_step_count":3,"violations":[] if valid else ["invalid"],
                                    "duplicate_step_indices":0,"duplicate_exact_step_bodies":0,"repeated_step_fields":0,"final_marker_count":1}
            item.update({f"tokens_{arm}":tokens,f"cap_{arm}":False,f"eos_{arm}":True})
        records.append(item)
    summary=probe.summarize(records)
    assert summary["invalid_to_valid"]==summary["valid_to_invalid"]==1
    assert summary["valid_to_valid"]==summary["invalid_to_invalid"]==1
    assert summary["response_token_delta"]==40
    assert summary["legacy"]["valid"]==summary["prompt_v1"]["valid"]==2
