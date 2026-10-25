"""LSTM đối chứng có cùng quy ước đầu vào và đầu ra như Selective SSM."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
        )
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(x)
        return self.regression_head(encoded[:, -1, :]).squeeze(-1)
