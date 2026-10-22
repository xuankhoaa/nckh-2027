"""Regression head around the unmodified MambaStock Mamba implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


MAMBASTOCK_DIR = Path(__file__).resolve().parents[1] / "external" / "MambaStock"
if not MAMBASTOCK_DIR.is_dir():
    raise ImportError(
        "MambaStock source is required at external/MambaStock. "
        "Clone the reference repository before training."
    )
if str(MAMBASTOCK_DIR) not in sys.path:
    sys.path.insert(0, str(MAMBASTOCK_DIR))

# Reuse the original selective state-space blocks and parallel scan without changes.
from mamba import Mamba, MambaConfig  # noqa: E402


class MambaRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16, layers: int = 2) -> None:
        super().__init__()
        config = MambaConfig(d_model=hidden_size, n_layers=layers)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.mamba = Mamba(config)
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.mamba(self.input_projection(x))
        # A window's target is next-day return, so only its final state feeds the head.
        return self.regression_head(encoded[:, -1, :]).squeeze(-1)
