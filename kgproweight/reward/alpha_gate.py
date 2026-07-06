"""Dynamic Confidence Gating Network (α-Gate).

  α_t = σ( (W^T · x_t + b) / τ )
  x_t = [f_density, f_confidence, f_entropy]

The crucial difference from the legacy implementation:

- ``compute_semantic_entropy`` takes the *real* token log-probabilities now
  (was hardcoded to 0.5 during PPO).
- ``compute_link_confidence`` accepts an optional ``kg_embedding_model``
  to switch from fuzzy matching to cosine similarity between TransE entity
  embeddings and the LM context embedding.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from kgproweight.kg.coverage import graph_density


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class AlphaGate(nn.Module):
    """Learnable gate ``α ∈ (0, 1)`` over the 3-feature vector ``x_t``."""

    def __init__(
        self,
        init_weights: Sequence[float] = (1.0, 1.5, -0.8),
        init_bias: float = -2.0,
        init_tau: float = 0.5,
        min_tau: float = 0.1,
    ) -> None:
        super().__init__()
        self.min_tau = min_tau
        self.W = nn.Parameter(torch.tensor(list(init_weights), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), dtype=torch.float32))

    @property
    def tau(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_tau), min=self.min_tau)

    def forward(
        self,
        graph_density_t: torch.Tensor,
        link_confidence_t: torch.Tensor,
        semantic_entropy_t: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack(
            [graph_density_t, link_confidence_t, semantic_entropy_t], dim=-1
        )
        logit = (x @ self.W + self.b) / self.tau
        return torch.sigmoid(logit)

    def forward_single(
        self,
        graph_density_v: float,
        link_confidence_v: float,
        semantic_entropy_v: float,
    ) -> float:
        with torch.no_grad():
            gd = torch.tensor([graph_density_v], dtype=torch.float32)
            lc = torch.tensor([link_confidence_v], dtype=torch.float32)
            se = torch.tensor([semantic_entropy_v], dtype=torch.float32)
            return float(self.forward(gd, lc, se).item())

    def extra_repr(self) -> str:
        return (
            f"W={self.W.data.tolist()}, "
            f"b={self.b.data.item():.3f}, "
            f"tau={self.tau.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def compute_graph_density(triples) -> float:
    """Wrapper around :func:`kgproweight.kg.coverage.graph_density`."""
    return graph_density(triples)


def compute_link_confidence(
    step_entities: List[str],
    entity_linker,
    kg_embedding_model=None,
    context_vector: Optional[torch.Tensor] = None,
) -> float:
    """Mean cos(KG-embed(entity), LM-context-embed) over linked entities.

    If ``kg_embedding_model`` is None or any entity is missing, falls back to
    the fuzzy-match confidence from ``EntityLinker.link_confidence``.
    """
    if not step_entities:
        return 0.0

    if kg_embedding_model is not None and context_vector is not None:
        scores: List[float] = []
        for ent in step_entities:
            try:
                cos = kg_embedding_model.cosine(ent, context_vector)
                # cos in [-1, 1] → map to [0, 1] for the gate's BCE.
                scores.append(max(0.0, 0.5 * (cos + 1.0)))
            except Exception:
                scores.append(entity_linker.link_confidence(ent))
        return float(sum(scores) / len(scores))

    scores = [entity_linker.link_confidence(e) for e in step_entities]
    return float(sum(scores) / len(scores))


def entropy_from_logprobs(logprobs: Optional[Sequence[float]]) -> float:
    """Approximate token-level entropy via ``-mean(log p_token)``.

    This is the negentropy of an empirical token distribution; under a
    one-hot prior it coincides with predictive entropy.
    """
    if not logprobs:
        return 1.0
    return max(0.0, -sum(logprobs) / len(logprobs))


def compute_semantic_entropy(logprobs: Optional[Sequence[float]]) -> float:
    """Alias preserved for backward compatibility."""
    return entropy_from_logprobs(logprobs)


def compute_features(
    step_entities: List[str],
    kg_subgraph,
    logprobs: Optional[Sequence[float]],
    entity_linker,
    kg_embedding_model=None,
    context_vector: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """Single-step 3-tuple feature: ``(density, confidence, entropy)``."""
    f_density = compute_graph_density(kg_subgraph)
    f_confidence = compute_link_confidence(
        step_entities=step_entities,
        entity_linker=entity_linker,
        kg_embedding_model=kg_embedding_model,
        context_vector=context_vector,
    )
    f_entropy = entropy_from_logprobs(logprobs)
    return f_density, f_confidence, f_entropy


# ---------------------------------------------------------------------------
# Calibration loss
# ---------------------------------------------------------------------------

class AlphaCalibrationLoss(nn.Module):
    """``L = w · BCE(α, coverage_target)``."""

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight
        self.bce = nn.BCELoss()

    def forward(self, alpha: torch.Tensor, coverage_targets: torch.Tensor) -> torch.Tensor:
        return self.weight * self.bce(alpha, coverage_targets)
