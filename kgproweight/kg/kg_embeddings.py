"""KG-embedding-based link confidence (TransE / RotatE on Wikidata5M).

This module provides an *optional* embedding-based confidence score:

  ``link_confidence = cos(embed_KG(entity), embed_LM(context_repr))``

If a PyKEEN checkpoint is not available, callers should fall back to the
fuzzy-matching score from :mod:`kgproweight.kg.entity_linker` (and log a
warning, as we do here).

The actual LM context embedding is supplied by the caller (e.g., the last
hidden state of the LLM for the relevant span). This module only handles
KG-side embedding lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


class KGEmbeddingModel:
    """Wraps a PyKEEN-style TransE/RotatE checkpoint to expose entity vectors."""

    def __init__(
        self,
        model_path: str,
        entity_to_id_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device

        try:
            import torch
            from pykeen.models import Model  # type: ignore # noqa: F401
        except ImportError as exc:  # pragma: no cover — optional dep
            raise ImportError(
                "pykeen + torch are required for KGEmbeddingModel. "
                "Install with `pip install -e .[kg-embeddings]`."
            ) from exc

        import torch as _torch  # local alias

        self._torch = _torch
        # PyKEEN saves the entire trained pipeline; we load and pull the entity tensor.
        pipeline = _torch.load(str(self.model_path), map_location=device, weights_only=False)
        if hasattr(pipeline, "model"):
            self.model = pipeline.model
        else:
            self.model = pipeline
        self.model.to(device)
        self.model.eval()

        # entity_to_id: dict[str, int] saved alongside the model
        self.entity_to_id: Dict[str, int] = {}
        if entity_to_id_path and Path(entity_to_id_path).exists():
            with open(entity_to_id_path, "r", encoding="utf-8") as fh:
                self.entity_to_id = json.load(fh)
        else:
            # PyKEEN convention: the triples-factory is usually attached
            tf = getattr(pipeline, "training", None) or getattr(self.model, "training_factory", None)
            if tf is not None and hasattr(tf, "entity_to_id"):
                self.entity_to_id = dict(tf.entity_to_id)

        if not self.entity_to_id:
            raise FileNotFoundError(
                "Could not resolve entity_to_id mapping; supply --entity_to_id_path."
            )

        try:
            self.entity_embeddings = self.model.entity_representations[0]().detach().to(device)
        except Exception:
            # Fallback for older PyKEEN versions.
            self.entity_embeddings = self.model.entity_embeddings.weight.detach().to(device)

        logger.info(
            "Loaded KG embeddings: %d entities, dim=%s",
            len(self.entity_to_id),
            tuple(self.entity_embeddings.shape),
        )

    def get_vector(self, entity_id_or_label: str) -> Optional["torch.Tensor"]:  # type: ignore[name-defined]
        """Return the entity vector or ``None`` if unknown."""
        idx = self.entity_to_id.get(entity_id_or_label)
        if idx is None:
            idx = self.entity_to_id.get(entity_id_or_label.lower())
        if idx is None:
            return None
        return self.entity_embeddings[idx]

    def cosine(self, entity: str, query_vector) -> float:  # type: ignore[no-untyped-def]
        """Return cos(entity_vec, query_vector). 0.0 when entity is OOV."""
        vec = self.get_vector(entity)
        if vec is None:
            return 0.0
        torch = self._torch
        v = vec.flatten().float()
        q = query_vector.flatten().float().to(v.device)
        denom = (v.norm() * q.norm()).clamp_min(1e-8)
        return float((v @ q / denom).item())


def load_kg_embeddings(
    model_path: Optional[str],
    entity_to_id_path: Optional[str] = None,
    device: str = "cpu",
) -> Optional[KGEmbeddingModel]:
    """Best-effort loader. Returns ``None`` on any failure with a warning."""
    if model_path is None:
        return None
    try:
        return KGEmbeddingModel(model_path, entity_to_id_path=entity_to_id_path, device=device)
    except Exception as exc:
        logger.warning(
            "Falling back to fuzzy link confidence: failed to load KG embeddings (%s).",
            exc,
        )
        return None
