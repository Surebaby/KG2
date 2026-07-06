"""Stage logging for long-running RAG pipelines (retrieval, refine, generation)."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("flashrag.progress")


def log_stage(message: str, *args) -> None:
    logger.info(message, *args)


class StageTimer:
    """Context manager: log start/end and elapsed seconds for a pipeline stage."""

    def __init__(self, name: str):
        self.name = name
        self._t0: float | None = None

    def __enter__(self) -> "StageTimer":
        self._t0 = time.time()
        log_stage("[Stage] %s — started", self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = time.time() - (self._t0 or time.time())
        if exc_type is None:
            log_stage("[Stage] %s — done (%.1fs)", self.name, elapsed)
        else:
            log_stage("[Stage] %s — failed after %.1fs: %s", self.name, elapsed, exc_val)
