"""Single source of truth for seeding RNGs across libraries."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python, numpy, torch (CPU+GPU), and transformers.

    Notes
    -----
    Determinism is *not* forced (`torch.use_deterministic_algorithms`) because
    GAE/PPO depend on non-deterministic CUDA kernels. The paper recipe
    averages over multiple seeds instead.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass
