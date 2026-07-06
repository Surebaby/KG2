"""GPT-4o (or any OpenAI-compatible chat model) as IHR judge.

The IHR (Intermediate Hallucination Rate) signal scores each *reasoning
step* on a binary hallucination scale. We use an OpenAI JSON-mode prompt
to elicit a structured response, then aggregate per-trajectory.

A tiny utility :func:`compute_cohen_kappa` is included to compute the
inter-rater agreement between the LLM judge and a human-labelled subset
(paper §5.5: κ ≥ 0.7 reproducibility check).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


JUDGE_SYSTEM_PROMPT = """You are a strict factual evaluator for multi-hop QA reasoning chains.

You will receive:
  • a user question Q
  • the gold answer A* (may be empty)
  • a single reasoning step S extracted from a longer trace.

Decide whether S contains a "hallucination": an unsupported factual claim
that contradicts the gold answer or asserts a relation between entities
that is unlikely to be true. Pure paraphrase, decomposition, or summary
without a falsifiable claim is NOT a hallucination.

Respond in compact JSON with EXACTLY these keys:
{"hallucination": <true|false>, "confidence": <0..1>, "reason": "<≤120 chars>"}
"""


JUDGE_USER_TEMPLATE = """Question: {question}
Gold answer: {gold_answer}

Reasoning step:
{step_text}

Respond as JSON only.
"""


@dataclass
class JudgeResult:
    step_index: int
    is_hallucination: bool
    confidence: float
    reason: str


class IHRJudge:
    """Thin wrapper around an OpenAI-compatible client.

    Parameters
    ----------
    model:
        The OpenAI model id. Recommended: ``gpt-4o-2024-08-06``.
    api_key, base_url:
        Override the default ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI  # local import; keeps `pip install` light

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def judge_step(
        self,
        question: str,
        gold_answer: str,
        step_text: str,
        step_index: int = 0,
    ) -> JudgeResult:
        user = JUDGE_USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer or "",
            step_text=step_text.strip(),
        )
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                data = json.loads(content)
                return JudgeResult(
                    step_index=step_index,
                    is_hallucination=bool(data.get("hallucination", False)),
                    confidence=float(data.get("confidence", 0.0)),
                    reason=str(data.get("reason", "")),
                )
            except Exception as exc:
                logger.warning("IHR judge attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)
                time.sleep(2 * (attempt + 1))
        return JudgeResult(step_index=step_index, is_hallucination=False, confidence=0.0, reason="judge_error")

    def judge_trajectory(
        self,
        question: str,
        gold_answer: str,
        step_texts: Sequence[str],
    ) -> List[JudgeResult]:
        return [
            self.judge_step(question, gold_answer, step_text=s, step_index=i)
            for i, s in enumerate(step_texts)
        ]

    @staticmethod
    def aggregate_ihr(results: Iterable[JudgeResult]) -> float:
        results = list(results)
        if not results:
            return 0.0
        return sum(1 for r in results if r.is_hallucination) / len(results)


# ---------------------------------------------------------------------------
# Cohen κ
# ---------------------------------------------------------------------------

def compute_cohen_kappa(human: Sequence[int], llm: Sequence[int]) -> float:
    """Cohen's κ for two binary annotators (0/1 labels).

    Uses the standard formula: ``κ = (p_o - p_e) / (1 - p_e)``.
    """
    if len(human) != len(llm) or not human:
        return 0.0
    n = len(human)
    agree = sum(1 for a, b in zip(human, llm) if a == b)
    p_o = agree / n
    p_yes_h = sum(human) / n
    p_yes_l = sum(llm) / n
    p_e = p_yes_h * p_yes_l + (1 - p_yes_h) * (1 - p_yes_l)
    if abs(1 - p_e) < 1e-12:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)
