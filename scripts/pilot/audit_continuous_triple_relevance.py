#!/usr/bin/env python
"""Score cited/control triples without changing silver labels.

The cross-encoder emits raw relevance logits.  They are intentionally not
converted into continuous labels until an independent human-labelled
calibration set supplies Platt parameters ``b`` and ``T``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from kgproweight.retrieval.reranker import get_cross_encoder, resolve_cross_encoder_path
from kgproweight.utils.logging import dump_manifest


Triple = Tuple[str, str, str]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triple(value: Sequence[Any]) -> Triple:
    return tuple(str(part).strip() for part in value)  # type: ignore[return-value]


def _key(value: Sequence[Any]) -> Triple:
    return tuple(str(part).strip().casefold() for part in value)  # type: ignore[return-value]


def _reasoning_without_citation_echo(text: str) -> str:
    """Remove the rendered citation block so the scorer cannot copy-match it."""
    return re.sub(
        r"\n?\s*Knowledge Used\s*:.*?(?=\n\s*Conclusion\s*:|\Z)",
        "\n",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _stats(values: Iterable[float]) -> Dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}

    def quantile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": quantile(0.25),
        "median": median(ordered),
        "p75": quantile(0.75),
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="models/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    records: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)
    for row in rows:
        split_value = int(
            hashlib.sha256(f"{args.seed}:{row.get('qid')}".encode("utf-8")).hexdigest()[:8],
            16,
        ) % 5
        review_split = "eval" if split_value == 0 else "fit"
        kg = [_triple(t) for t in row.get("kg_subgraph") or [] if len(t) == 3]
        for step_pos, step in enumerate(row.get("steps") or []):
            cited = [_triple(t) for t in step.get("cited_triples") or [] if len(t) == 3]
            if not cited:
                continue
            cited_keys = {_key(t) for t in cited}
            controls = [t for t in kg if _key(t) not in cited_keys]
            rng.shuffle(controls)
            query_text = (
                f"Question: {row.get('question') or ''}\n"
                f"Step: {_reasoning_without_citation_echo(str(step.get('text') or ''))}"
            )
            for citation_pos, triple in enumerate(cited):
                records.append({
                    "pair_id": f"{row.get('qid')}:{step_pos}:cited:{citation_pos}",
                    "qid": row.get("qid"),
                    "dataset": row.get("dataset"),
                    "step_position": step_pos,
                    "step_index": step.get("index"),
                    "pair_type": "cited",
                    "review_split": review_split,
                    "query_text": query_text,
                    "triple": list(triple),
                    "human_relevance": None,
                })
            for control_pos, triple in enumerate(controls[: len(cited)]):
                records.append({
                    "pair_id": f"{row.get('qid')}:{step_pos}:control:{control_pos}",
                    "qid": row.get("qid"),
                    "dataset": row.get("dataset"),
                    "step_position": step_pos,
                    "step_index": step.get("index"),
                    "pair_type": "noncited_control",
                    "review_split": review_split,
                    "query_text": query_text,
                    "triple": list(triple),
                    "human_relevance": None,
                })

    if not records:
        raise SystemExit("no cited/control pairs found")

    model = get_cross_encoder(args.model)
    # CrossEncoder may otherwise apply a model-dependent activation.  Identity
    # guarantees that the persisted values are raw logits, not probabilities.
    import torch

    model.model.to(args.device)
    pairs = [
        (record["query_text"], " | ".join(record["triple"]))
        for record in records
    ]
    scores = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        activation_fn=torch.nn.Identity(),
        convert_to_numpy=True,
    )
    for record, score in zip(records, scores):
        record["raw_cross_encoder_logit"] = float(score)
        record["calibration_status"] = "UNCALIBRATED"

    pairs_path = output_dir / "relevance_pairs_for_human_review.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    cited_scores = [r["raw_cross_encoder_logit"] for r in records if r["pair_type"] == "cited"]
    control_scores = [r["raw_cross_encoder_logit"] for r in records if r["pair_type"] == "noncited_control"]
    report = {
        "status": "UNCALIBRATED",
        "reason": "No independent human relevance labels were supplied; raw logits are not probabilities.",
        "source": {"path": str(input_path), "md5": _md5(input_path), "rows": len(rows)},
        "protocol": {
            "model": str(Path(resolve_cross_encoder_path(args.model)).resolve()),
            "device": args.device,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "citation_echo_removed": True,
            "control": "same-step, same-KG, non-cited triple; diagnostic only",
            "gold_used": False,
        },
        "accounting": {
            "pairs": len(records),
            "cited_pairs": len(cited_scores),
            "noncited_control_pairs": len(control_scores),
        },
        "raw_logit_distribution": {
            "cited": _stats(cited_scores),
            "noncited_control": _stats(control_scores),
        },
        "proposed_formula_after_calibration": {
            "per_triple": "u_i = sigmoid((raw_logit_i - b) / T)",
            "positive_step": "r_kg = mean(u_i over verified cited triples)",
            "no_citation": "r_kg = 0",
            "verified_contradiction": "r_kg = -1 (unchanged in this pilot)",
            "constraint": "Fit b and positive T only on an independent human-labelled calibration split.",
        },
        "output": str(pairs_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dump_manifest(
        output_dir / "run",
        extra={
            "experiment": "continuous_triple_relevance_raw_score_audit",
            "report": str(report_path),
            "status": "UNCALIBRATED",
            "pairs": len(records),
            "gold_used": False,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
