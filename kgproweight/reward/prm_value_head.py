"""Critic head used by the PPO trainer.

Lives in its own module so the PPO entrypoint can import it without pulling
in optional KG / fuzzy / reward dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PRMValueHead(nn.Module):
    """MLP value head: ``hidden_state → V(s)``."""

    def __init__(self, hidden_size: int = 4096, mid_size: int = 512) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.GELU(),
            nn.Linear(mid_size, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Hidden state may be ``(B, H)`` or ``(B, T, H)``. Returns ``(B,)`` or ``(B, T)``."""
        out = self.layers(hidden_state)
        return out.squeeze(-1)
