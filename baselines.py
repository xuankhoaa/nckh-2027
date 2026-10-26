"""Các baseline hồi quy đơn giản để so sánh công bằng với mạng nơ-ron."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def predict_baseline(name: str, y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Dự báo hằng số theo quy tắc chỉ được ước lượng từ tập huấn luyện."""
    if name == "naive_zero":
        return np.zeros(n_test, dtype=np.float32)
    if name == "train_mean":
        return np.full(n_test, float(np.mean(y_train)), dtype=np.float32)
    raise ValueError(f"Baseline không hỗ trợ: {name}")


def predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0),
    positive_target: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Chọn hệ số Ridge hoàn toàn trên validation rồi dự báo test.

    Chuỗi được trải phẳng vì mục tiêu của baseline này là kiểm tra liệu một mô
    hình tuyến tính có đủ sức khai thác cùng cửa sổ đầu vào hay không.
    """
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge alpha phải là các số dương.")
    train_2d = np.asarray(x_train, dtype=float).reshape(len(x_train), -1)
    valid_2d = np.asarray(x_valid, dtype=float).reshape(len(x_valid), -1)
    test_2d = np.asarray(x_test, dtype=float).reshape(len(x_test), -1)
    train_target = np.asarray(y_train, dtype=float)
    if positive_target:
        # Realized volatility có thể bằng 0 ở phiên không đổi giá. Một sàn nhỏ
        # giữ biến đổi log hữu hạn mà không thay đổi thang đo của quan sát thường.
        floor = np.finfo(np.float32).eps
        train_target = np.log(np.maximum(train_target, floor))
    candidates: list[tuple[float, float, Ridge]] = []
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(train_2d, train_target)
        validation_prediction = model.predict(valid_2d)
        if positive_target:
            validation_prediction = np.exp(validation_prediction)
        validation_mse = float(mean_squared_error(y_valid, validation_prediction))
        candidates.append((validation_mse, float(alpha), model))
    _, selected_alpha, selected_model = min(candidates, key=lambda item: (item[0], item[1]))
    parameter_count = int(selected_model.coef_.size + 1)
    prediction = selected_model.predict(test_2d)
    if positive_target:
        prediction = np.exp(prediction)
    return prediction.astype(np.float32), selected_alpha, parameter_count
