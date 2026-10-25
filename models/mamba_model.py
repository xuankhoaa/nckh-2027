"""Selective State Space Model dùng chung cho hồi quy return và volatility.

Triển khai này được viết riêng cho đề tài từ mô tả trong bài báo Mamba của Gu và
Dao (2023), không sao chép mã từ MambaStock. Phần quét trạng thái dùng phép
quét tiền tố kết hợp (associative prefix scan), tương đương chính xác với truy
hồi tuần tự nhưng tránh một vòng lặp Python cho từng phiên.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


def selective_scan(transitions: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Tính trạng thái của truy hồi ``h_t = a_t h_{t-1} + b_t`` với ``h_-1=0``.

    ``transitions`` và ``inputs`` có dạng ``[batch, length, ...]``. Mỗi cặp
    biến đổi affine ``(a, b)`` được ghép bởi
    ``(a_2, b_2) o (a_1, b_1) = (a_2*a_1, b_2 + a_2*b_1)``. Phép ghép này có
    tính kết hợp, vì vậy quét tiền tố theo cây cho cùng kết quả với quét lần
    lượt theo thời gian, trong khi chỉ cần ``ceil(log2(length))`` bước tensor.
    """
    if transitions.shape != inputs.shape:
        raise ValueError("transitions và inputs phải có cùng kích thước.")
    if transitions.ndim < 2:
        raise ValueError("Tensor selective scan cần ít nhất chiều batch và thời gian.")

    prefix_transition = transitions
    prefix_input = inputs
    length = transitions.shape[1]
    step = 1
    while step < length:
        # Tạo tensor mới để autograd không bị ảnh hưởng bởi ghi đè tại chỗ.
        right_transition = prefix_transition[:, step:]
        left_transition = prefix_transition[:, :-step]
        right_input = prefix_input[:, step:]
        left_input = prefix_input[:, :-step]
        prefix_transition = torch.cat(
            (prefix_transition[:, :step], right_transition * left_transition), dim=1
        )
        prefix_input = torch.cat(
            (prefix_input[:, :step], right_input + right_transition * left_input), dim=1
        )
        step *= 2
    return prefix_input


class RMSNorm(nn.Module):
    """Chuẩn hóa RMS tối giản cho tensor có chiều đặc trưng ở cuối."""

    def __init__(self, width: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return x * scale * self.weight


class SelectiveSSMBlock(nn.Module):
    """Một Mamba-style block gồm causal convolution, selective SSM và cổng SiLU."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_projection = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.causal_conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.parameter_projection = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.delta_projection = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.output_projection = nn.Linear(self.d_inner, d_model, bias=False)

        # A có trị riêng âm để trạng thái liên tục ổn định trước rời rạc hóa.
        initial = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.a_log = nn.Parameter(torch.log(initial))
        self.skip = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Nhận [batch, length, d_model] và trả về tensor cùng hình dạng."""
        batch, length, _ = sequence.shape
        content, gate = self.in_projection(sequence).chunk(2, dim=-1)
        content = self.causal_conv(content.transpose(1, 2))[..., :length].transpose(1, 2)
        content = functional.silu(content)

        parameters = self.parameter_projection(content)
        delta_raw, b_vector, c_vector = torch.split(
            parameters,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = functional.softplus(self.delta_projection(delta_raw))
        a_matrix = -torch.exp(self.a_log.float())
        # Δ, B, C thay đổi theo phiên; biểu thức dưới là hệ số a_t và b_t của
        # h_t = a_t h_(t-1) + b_t. selective_scan giữ đúng quan hệ nhân quả.
        delta_expanded = delta.unsqueeze(-1)
        transitions = torch.exp(delta_expanded * a_matrix)
        inputs = (
            delta_expanded
            * b_vector.unsqueeze(2)
            * content.unsqueeze(-1)
        )
        states = selective_scan(transitions, inputs)
        selective_output = (states * c_vector.unsqueeze(2)).sum(dim=-1)
        selective_output = selective_output + self.skip * content
        return self.output_projection(selective_output * functional.silu(gate))


class ResidualSelectiveSSM(nn.Module):
    """Khối chuẩn hóa trước và kết nối phần dư."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.block = SelectiveSSMBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.block(self.norm(sequence))


class MambaRegressor(nn.Module):
    """Hồi quy một target từ trạng thái cuối của chuỗi Selective SSM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 8,
        layers: int = 1,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList(
            [ResidualSelectiveSSM(hidden_size, d_state, d_conv, expand) for _ in range(layers)]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(sequence)
        for layer in self.layers:
            encoded = layer(encoded)
        encoded = self.final_norm(encoded)
        return self.regression_head(encoded[:, -1, :]).squeeze(-1)
