"""Frozen ProofKG process reward v2.1 (config-gated in mixed PPO-TK).

Rewards covering the question's planned Proof hops in dependency order, and
verifies the final derived answer.  Fixes the four v2 formula/code mismatches:

  P  precise required-hop citations / ALL unique structured citations
     (unknown citations enter the denominator, no separate heavy penalty);
  O  dependency order: parent hop's first-cited step < child hop's step;
  G  conclusion grounding: output_entities first, matched-triple tail as fallback;
  A  full derivation: terminal bridge / same-different yes-no / temporal-numeric
     min-max, mapped back to the root entity.

Formula (frozen):  R = (0.05P + 0.15G + 0.35*(H*O) + 0.45*m_A*A) / (0.55 + 0.45*m_A)
No gold answer enters this module.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from kgproweight.data.parsers import extract_final_answer, parse_steps

SCORER_VERSION = "proofkg-process-v2-1-frozen-1"

_QUESTION_SAME_DIFF = re.compile(r"\b(same|different|common|share|both)\b", re.IGNORECASE)
_EARLIER = re.compile(r"\b(first|earlier|later|older|younger|more|less|higher|lower|longer|shorter|before|after|oldest|youngest)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{3,4}$")  # a bare year


def _norm(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _phrase_in(phrase: object, text: object) -> bool:
    needle = _norm(phrase)
    haystack = _norm(text)
    return bool(needle and re.search(rf"(?:^| ){re.escape(needle)}(?: |$)", haystack))


def _dynamic_min_steps(planned_hops: int) -> int:
    return max(2, min(3, int(planned_hops)))


def build_execution_trace(plan: Mapping[str, Any], execution: Mapping[str, Any]) -> List[Dict[str, Any]]:
    plan_hops = list(plan.get("hops") or [])
    exec_hops = list(execution.get("hops") or [])
    trace: List[Dict[str, Any]] = []
    for ph in plan_hops:
        slot = str(ph.get("output_slot") or "")
        deps: List[str] = []
        subject = str(ph.get("subject") or "")
        if subject.startswith("$"):
            deps.append(subject[1:])
        hop_no = _hop_number(slot)
        eh = next((h for h in exec_hops if str(h.get("hop_index") or "") == str(hop_no)), None)
        trace.append({
            "output_slot": slot,
            "dependencies": deps,
            "relation_role": str(ph.get("relation_role") or "answer_operand"),
            "pids": list(ph.get("pids") or []),
            "matched_triples": [tuple(t) for t in (eh.get("matches") or []) if len(t) == 3] if eh else [],
            "output_entities": list(eh.get("output_entities") or []) if eh else [],
        })
    return trace


def _hop_number(output_slot: str) -> int:
    m = re.search(r"(\d+)$", str(output_slot))
    return int(m.group(1)) if m else 0


def _precise_citation(steps, req: set) -> Tuple[float, int, int]:
    """P = cited required triples / all unique structured citations (known+unknown)."""
    known = {t for step in steps for t in step.cited_triples}
    unknown = {str(s).strip() for step in steps for s in step.unknown_citation_surfaces if str(s).strip()}
    denom = len(known) + len(unknown)
    return (len(known & req) / denom if denom else 0.0), len(known), len(unknown)


def _dependency_order(trace: List[Dict[str, Any]], steps) -> float:
    """O = fraction of dependency hops whose parent was cited before the child."""
    first: Dict[str, int] = {}
    for hop in trace:
        hop_triples = set(hop["matched_triples"])
        for i, step in enumerate(steps):
            if hop_triples & set(step.cited_triples):
                first[hop["output_slot"]] = i
                break
    ordered = total = 0
    for hop in trace:
        if not hop["dependencies"]:
            continue
        total += 1
        child = first.get(hop["output_slot"])
        parent = first.get(hop["dependencies"][0])
        if child is not None and parent is not None and parent < child:
            ordered += 1
    return ordered / total if total else 1.0


def _conclusion_grounding(trace: List[Dict[str, Any]], steps) -> float:
    grounded = 0
    for hop in trace:
        hop_triples = set(hop["matched_triples"])
        if not hop_triples:
            continue
        outs = [str(e.get("label") or e.get("qid") or "") for e in hop.get("output_entities") or []]
        outs = [o for o in outs if o]
        if not outs:
            outs = [str(t[2]) for t in hop["matched_triples"] if len(t) == 3]
            outs = [o for o in outs if o]
        for step in steps:
            if hop_triples & set(step.cited_triples):
                conclusion = step.intermediate_conclusion or ""
                if any(_phrase_in(o, conclusion) for o in outs):
                    grounded += 1
                break
    return grounded / max(1, len(trace))


def _question_operator(question: str) -> str:
    if _QUESTION_SAME_DIFF.search(question):
        return "same_different"
    if _EARLIER.search(question):
        return "temporal"
    return "terminal_bridge"


def _answer_consistency(
    question: str, kg_triples: Sequence[Sequence[str]], trace: List[Dict[str, Any]], answer: str
) -> Tuple[float, int, str, str]:
    """Return (A, m_A, operator, derived_answer)."""
    op = _question_operator(question)
    answer_operands = [hop for hop in trace if hop["relation_role"] == "answer_operand"]
    tails = [str(t[2]) for hop in answer_operands for t in hop["matched_triples"] if len(t) == 3]
    tails = [t for t in tails if t]

    if op == "terminal_bridge":
        derived = tails[-1] if tails else ""
        m_A = 1 if derived else 0
        a = float(bool(answer) and any(_phrase_in(answer, d) or _phrase_in(d, answer) for d in tails))
        return a, m_A, op, derived

    # same/different: derive yes/no from operand equality
    if op == "same_different":
        if len(tails) < 2:
            return 0.0, 0, op, ""
        equal = _norm(tails[0]) == _norm(tails[-1])
        derived = "yes" if equal else "no"
        return float(bool(answer) and _norm(answer) in ("yes", "no") and _norm(answer) == _norm(derived)), 1, op, derived

    # temporal/numeric: parse bare years (or numbers) on tails, select min/max, map to head
    heads: List[str] = []
    values: List[Tuple[str, int]] = []
    for h, r, t in kg_triples:
        if len(t) != 3:
            continue
        tail = str(t[2])
        if _DATE_RE.match(tail):
            heads.append(str(t[0]))
            values.append((str(t[0]), int(tail)))
    if not values:
        return 0.0, 0, op, ""
    earlier = bool(re.search(r"\b(first|earlier|older|before|oldest)\b", question, re.IGNORECASE))
    chosen = min(values, key=lambda x: x[1])[0] if earlier else max(values, key=lambda x: x[1])[0]
    return float(bool(answer) and (_phrase_in(answer, chosen) or _phrase_in(chosen, answer))), 1, op, chosen


def score_proofkg_v2(
    *,
    question: str,
    generation: str,
    kg_triples: Sequence[Sequence[str]],
    execution_trace: List[Dict[str, Any]],
    planned_hops: int,
) -> Dict[str, Any]:
    triples = {tuple(str(x).strip() for x in e) for e in kg_triples if len(e) == 3}
    steps = parse_steps(generation, known_kg=list(triples))
    answer = (extract_final_answer(generation) or "").split("\n", 1)[0].strip()

    min_steps = _dynamic_min_steps(planned_hops)
    valid = bool(steps) and len(steps) >= min_steps and extract_final_answer(generation) is not None
    if valid:
        expected = 1
        for s in steps:
            if s.index != expected or not s.raw_text.strip():
                valid = False
                break
            body = s.raw_text.strip()
            m = re.search(r"(?is)\breasoning\s*:\s*(.*?)(?:knowledge used\s*:|conclusion\s*:|$)", body)
            if not m or len(m.group(1).strip()) < 20:
                valid = False
                break
            expected += 1

    if not valid:
        return {"scorer_version": SCORER_VERSION, "score": -1.0, "trajectory_valid": False,
                "prediction": answer, "n_steps": len(steps), "components": {}}

    req = {t for hop in execution_trace for t in hop["matched_triples"]}
    n_hops = max(1, len(execution_trace))

    P, n_known, n_unknown = _precise_citation(steps, req)
    O = _dependency_order(execution_trace, steps)
    G = _conclusion_grounding(execution_trace, steps)

    cited_set = {t for step in steps for t in step.cited_triples}
    covered = [bool(set(hop["matched_triples"]) & cited_set) for hop in execution_trace]
    H = sum(covered) / n_hops

    A, m_A, operator, derived = _answer_consistency(question, kg_triples, execution_trace, answer)

    C = H * O
    score = (0.05 * P + 0.15 * G + 0.35 * C + 0.45 * m_A * A) / (0.55 + 0.45 * m_A)
    return {
        "scorer_version": SCORER_VERSION,
        "score": float(score),
        "trajectory_valid": True,
        "prediction": answer,
        "n_steps": len(steps),
        "components": {
            "P_precise_citation": P,
            "H_hop_coverage": H,
            "O_dependency_order": O,
            "G_conclusion_grounding": G,
            "A_answer_consistency": A,
            "m_A_deterministic": float(m_A),
            "C": C,
            "operator": operator,
            "derived_answer": derived,
        },
        "telemetry": {"n_known_citations": n_known, "n_unknown_citations": n_unknown},
    }
