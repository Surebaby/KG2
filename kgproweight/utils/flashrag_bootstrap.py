"""Single location for FlashRAG ``sys.path`` injection.

If FlashRAG is already importable (e.g. installed via
``pip install -e third_party/FlashRAG``), this module is a no-op.
Otherwise it tries to locate the FlashRAG source root via
``kgproweight.utils.paths.flashrag_root()`` and prepend it.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from kgproweight.utils.paths import flashrag_root


@lru_cache(maxsize=1)
def setup_flashrag(extra_root: Optional[str] = None) -> Optional[Path]:
    """Ensure ``import flashrag`` works.

    Returns the FlashRAG source root if path injection happened, ``None``
    if FlashRAG was already importable.
    """
    def _flashrag_usable() -> bool:
        try:
            importlib.import_module("flashrag.pipeline.reasoning_pipeline")
            return True
        except ImportError:
            return False

    if _flashrag_usable():
        return None

    candidates = []
    if extra_root:
        candidates.append(Path(extra_root))
    root = flashrag_root()
    if root is not None:
        candidates.append(root)

    for root_dir in candidates:
        root_dir = root_dir.resolve()
        if not root_dir.exists():
            continue
        # Insert the parent of the `flashrag` package, not the package dir itself.
        sys.path.insert(0, str(root_dir))
        try:
            importlib.import_module("flashrag")
            return root_dir
        except ImportError:
            sys.path.remove(str(root_dir))

    raise ImportError(
        "FlashRAG is not importable. Set KGPW_FLASHRAG_ROOT or "
        "`pip install -e third_party/FlashRAG`."
    )
