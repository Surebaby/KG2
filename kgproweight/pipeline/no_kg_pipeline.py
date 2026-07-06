"""Bypass-KG inference pipeline (``no_kg`` ablation).

Shares the canonical SFT/inference schema with :class:`KGProWeightPipeline`
but skips Wikidata subgraph injection.
"""

from __future__ import annotations

from kgproweight.pipeline.kg_proweight_pipeline import KGProWeightPipeline


class NoKGReasoningPipeline(KGProWeightPipeline):
    """KG-ProWeight inference without KG context."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, inject_kg=False, record_alpha=False, **kwargs)
        self._kgpw_ablation = "no_kg"
