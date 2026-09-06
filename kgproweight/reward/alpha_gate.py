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
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class AlphaGate(nn.Module):
    """Learnable gate ``α ∈ (0, 1)`` over the feature vector ``x_t``.

    FEATURE ORDER (positional, do not reorder -- checkpoints depend on it):

        0  graph_density     |E| / (|V| + eps) of the step's filtered subgraph
        1  link_confidence   mean fuzzy-link confidence of the step's entities
        2  semantic_entropy  -mean(log p_token) over the step's tokens
        3  cite_any          1.0 if the step cited ANY triple           (new)
        4  cite_match        fraction of cited triples found in the KG  (new)

    2026-08-23 -- why features 3 and 4 exist. The 3-feature gate could not fit
    its own calibration target. Fitting this exact functional form directly to
    ``kg_has_verdict`` (base rate 0.2983 over 33,011 accepted silver steps) gave
    a CEILING of R^2 = +0.038 over a constant predictor, and the shipped
    checkpoint scored WORSE than a constant (R^2 -0.674 at the measured
    f_entropy=0.603, -1.105 at 0.0). The cause is that neither live feature
    carries signal about the target: corr(f_density, target) = +0.172 and
    corr(f_confidence, target) = +0.130. Measured ceilings:

        constant (base rate)     BCE 0.6094  Brier 0.2093  R^2  0.000
        3 features (ceiling)     BCE 0.5854  Brier 0.2013  R^2 +0.038
        + cite_any               BCE 0.4317  Brier 0.1432  R^2 +0.316
        + cite_match             BCE 0.4344  Brier 0.1342  R^2 +0.359
        + both                   BCE 0.3762  Brier 0.1174  R^2 +0.439
        cite_any ALONE           BCE 0.4380  Brier 0.1466  R^2 +0.300

    One citation feature alone carries 8x the information of all three original
    features combined. Both together give 12x.

    CIRCULARITY, stated openly: ``PRMAnnotator.label`` returns NEUTRAL partly
    BECAUSE a step cited nothing (prm_annotator.py:184), so cite_any is not
    independent of the target. Measured conditional structure over the same
    33,011 steps:

        cite_any=0 (n=16,150)  ->  P(verdict) = 0.042    near-deterministic
        cite_any=1 (n=16,861)  ->  P(verdict) = 0.543    genuinely uncertain

    The definitional direction is the negative one; on the 51% of steps that DO
    cite, the feature narrows the target from 0.298 to 0.543 without determining
    it. That asymmetry is why this is a usable feature and not a label leak, but
    the gate must never be reported as "predicting" verdicts -- it is a WEIGHT on
    the KG channel, and cite_any is a legitimate input to that weight (a step
    citing nothing should not be graded on its KG grounding).

    Both new features are computable IDENTICALLY at training and inference:
    ``ParsedStep.cited_triples`` is filled from the silver ``cited_triples`` field
    during Phase 2 and by ``TRIPLE_RE`` over generated text at inference.
    """

    #: Number of features the current architecture expects.
    N_FEATURES = 5

    def __init__(
        self,
        init_weights: Sequence[float] = (1.0, 1.5, -0.8, 0.9, 1.0),
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
        cite_any_t: Optional[torch.Tensor] = None,
        cite_match_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute α. The citation features default to zeros when not supplied.

        Defaulting rather than requiring them keeps every existing 3-argument
        call site working. A zero default is also the RIGHT neutral value: it is
        what a step that cites nothing actually scores, so an old call site that
        never passes them behaves like "no citations observed" instead of
        silently reading uninitialised state.
        """
        return torch.sigmoid(
            self.forward_logits(
                graph_density_t,
                link_confidence_t,
                semantic_entropy_t,
                cite_any_t,
                cite_match_t,
            )
        )

    def forward_logits(
        self,
        graph_density_t: torch.Tensor,
        link_confidence_t: torch.Tensor,
        semantic_entropy_t: torch.Tensor,
        cite_any_t: Optional[torch.Tensor] = None,
        cite_match_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the pre-sigmoid gate logits for stable BCE training."""
        feats = [graph_density_t, link_confidence_t, semantic_entropy_t]
        if self.W.numel() > 3:
            zero = torch.zeros_like(graph_density_t)
            feats.append(cite_any_t if cite_any_t is not None else zero)
            if self.W.numel() > 4:
                feats.append(cite_match_t if cite_match_t is not None else zero)
        x = torch.stack(feats, dim=-1)
        return (x @ self.W + self.b) / self.tau

    def forward_single(
        self,
        graph_density_v: float,
        link_confidence_v: float,
        semantic_entropy_v: float,
        cite_any_v: float = 0.0,
        cite_match_v: float = 0.0,
    ) -> float:
        with torch.no_grad():
            t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
            return float(
                self.forward(
                    t(graph_density_v),
                    t(link_confidence_v),
                    t(semantic_entropy_v),
                    t(cite_any_v),
                    t(cite_match_v),
                ).item()
            )

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """Load a checkpoint, ZERO-PADDING ``W`` if it predates the new features.

        A 3-feature checkpoint would otherwise fail with a size mismatch at every
        load site (``phase3_ppo.py``, ``kg_proweight_pipeline.py``,
        ``phase3_grpo.py``). Padding with zeros reproduces the old gate EXACTLY --
        the new features get weight 0 -- so an old checkpoint keeps behaving as it
        did, and only a Phase 2 re-run gives the new features nonzero weight.

        This is deliberately permissive in ONE direction only. A checkpoint WIDER
        than the current architecture is an error, not something to truncate:
        silently dropping a trained weight would change α with no warning.
        """
        sd = dict(state_dict)
        w = sd.get("W")
        if w is not None and w.ndim == 1 and w.numel() < self.W.numel():
            pad = torch.zeros(self.W.numel() - w.numel(), dtype=w.dtype, device=w.device)
            sd["W"] = torch.cat([w, pad])
            logger.warning(
                "AlphaGate checkpoint has %d features, this build expects %d. "
                "Zero-padding the new weights, which reproduces the OLD gate "
                "exactly. Re-run Phase 2 to actually fit the citation features.",
                w.numel(), self.W.numel(),
            )
        elif w is not None and w.numel() > self.W.numel():
            raise ValueError(
                "AlphaGate checkpoint has %d features but this build expects %d. "
                "Refusing to truncate a trained weight." % (w.numel(), self.W.numel())
            )
        return super().load_state_dict(sd, strict=strict)

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
    """``L = w · BCEWithLogits(alpha_logit, coverage_target)``.

    This is mathematically equivalent to BCE(sigmoid(logit), target), while
    avoiding avoidable overflow/underflow at saturated gate values.  The target
    semantics are deliberately unchanged here.
    """

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, alpha_logits: torch.Tensor, coverage_targets: torch.Tensor) -> torch.Tensor:
        return self.weight * self.bce(alpha_logits, coverage_targets)
