"""Prediction post-processing for evaluation."""

from __future__ import annotations

from kgproweight.data.parsers import extract_final_answer


def kg_proweight_pred_process(dataset):
    """FlashRAG ``pred_process_fun`` — extract ``[Final Answer]`` for EM/F1."""
    for item in dataset:
        raw = item.pred or ""
        item.pred = extract_final_answer(raw) or raw.strip()
    return dataset


def extract_kg_proweight_answer(raw_output: str) -> str:
    """Return the concise answer string from a full model trace."""
    if not raw_output:
        return ""
    return extract_final_answer(raw_output) or raw_output.strip()
