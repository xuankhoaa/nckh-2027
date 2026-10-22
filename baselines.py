"""Các baseline hồi quy đơn giản để so sánh công bằng với mạng nơ-ron."""

from __future__ import annotations

import numpy as np


def predict_baseline(name: str, y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Dự báo hằng số theo quy tắc chỉ được ước lượng từ tập huấn luyện."""
    if name == "naive_zero":
        return np.zeros(n_test, dtype=np.float32)
    if name == "train_mean":
        return np.full(n_test, float(np.mean(y_train)), dtype=np.float32)
    raise ValueError(f"Baseline không hỗ trợ: {name}")
