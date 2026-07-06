"""KG-ProWeight: Adaptive Process Supervision for Agentic RAG.

Public re-exports are intentionally minimal to keep `import kgproweight`
cheap (no torch import). Heavy submodules are imported lazily on access.
"""

from __future__ import annotations

from kgproweight.version import __version__

__all__ = ["__version__"]
