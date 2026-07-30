"""Thin wrapper around FlashRAG's dataset and config utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

from kgproweight.utils.flashrag_bootstrap import setup_flashrag
from kgproweight.utils.logging import get_logger

logger = get_logger(__name__)


def get_dataset(config: Any, split: str = "dev"):
    """Load a FlashRAG ``Dataset`` from a FlashRAG ``Config`` (or dict).

    ``config`` can be either:
      • a real ``flashrag.config.Config`` instance, or
      • a dict (we wrap it in ``Config``).
    """
    setup_flashrag()
    from flashrag.config import Config
    from flashrag.utils import get_dataset as _get

    if not isinstance(config, Config):
        cfg = Config(config_dict=dict(config))
    else:
        cfg = config
    cfg["split"] = [split]
    dataset = _get(cfg)
    if isinstance(dataset, dict):
        selected = dataset.get(split)
        if selected is None:
            available = [k for k, v in dataset.items() if v is not None]
            try:
                dataset_path = cfg["dataset_path"]
            except Exception:
                dataset_path = "<unknown>"
            raise FileNotFoundError(
                f"FlashRAG dataset split '{split}' is missing under dataset_path='{dataset_path}'. "
                f"Available non-empty splits: {available}. "
                "Please pass a valid --split (e.g. test/dev) or prepare the missing split file."
            )
        return selected
    return dataset


def flashrag_config(config_dict: Dict[str, Any]) -> Any:
    """Build a real ``flashrag.config.Config`` from a plain dict."""
    setup_flashrag()
    from flashrag.config import Config

    return Config(config_dict=config_dict)


def load_corpus(corpus_path: Optional[str] = None):
    """Convenience wrapper around FlashRAG ``load_corpus``."""
    setup_flashrag()
    from flashrag.utils import load_corpus as _load

    return _load(corpus_path)
