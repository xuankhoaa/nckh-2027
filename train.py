"""Huấn luyện hoặc đánh giá một cấu hình dự báo return có kiểm soát leakage."""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from baselines import predict_baseline
from evaluate import (
    plot_actual_vs_predicted,
    plot_learning_curve,
    regression_metrics,
    save_experiment_outputs,
)
from models.lstm import LSTMRegressor
from models.mamba_model import MambaRegressor
from preprocess import build_feature_frame, make_sequence_split, print_split_summary


NEURAL_MODELS = ("lstm", "mamba_baseline", "improved_mamba")
BASELINE_MODELS = ("naive_zero", "train_mean")
ALL_MODELS = (*BASELINE_MODELS, *NEURAL_MODELS)


@dataclass(frozen=True)
class Experiment:
    model: str
    ticker: str
    seed: int = 42
    window: int = 30
    epochs: int = 40
    learning_rate: float = 1e-3
    # LSTM (585 tham số với 8 đầu vào) và Selective SSM (545 tham số) được
    # ghép quy mô gần tương đương để so sánh công bằng trên bộ dữ liệu nhỏ.
    hidden_size: int = 8
    layers: int = 1
    batch_size: int = 128
    patience: int = 8

    @property
    def include_vnindex(self) -> bool:
        return self.model == "improved_mamba"

    @property
    def feature_set(self) -> str:
        if self.model in BASELINE_MODELS:
            return "khong_ap_dung"
        return "ohlcv_derived_plus_vnindex" if self.include_vnindex else "ohlcv_derived"


def set_seed(seed: int) -> None:
    """Cố định nguồn ngẫu nhiên ở mức phù hợp để chạy lại trên cùng môi trường."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def target_scale_from_train(y_train: np.ndarray) -> tuple[float, float]:
    """Ước lượng thang đo nhãn chỉ từ train để tránh làm lộ phân phối tương lai."""
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= np.finfo(np.float32).eps:
        raise ValueError("Nhãn train không đủ biến thiên để chuẩn hóa.")
    return mean, std


def _make_model(experiment: Experiment, input_size: int) -> nn.Module:
    arguments = {"input_size": input_size, "hidden_size": experiment.hidden_size, "layers": experiment.layers}
    if experiment.model == "lstm":
        return LSTMRegressor(**arguments)
    if experiment.model in ("mamba_baseline", "improved_mamba"):
        return MambaRegressor(**arguments)
    raise ValueError(f"Mô hình mạng không hỗ trợ: {experiment.model}")


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=False)


def _loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for features, target in loader:
            loss = criterion(model(features.to(device)), target.to(device))
            total += loss.item() * len(target)
            count += len(target)
    return total / count


def _evaluate_baseline(experiment: Experiment, dataset) -> tuple[np.ndarray, int, list[float], list[float]]:
    predicted = predict_baseline(experiment.model, dataset.y_train, len(dataset.y_test))
    return predicted, 0, [], []


def _train_neural_model(experiment: Experiment, dataset) -> tuple[np.ndarray, int, list[float], list[float]]:
    set_seed(experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Các return có biên độ rất nhỏ. Chuẩn hóa nhãn theo train giúp tối ưu ổn
    # định; dự báo được đảo chuẩn hóa về đơn vị log-return trước đánh giá.
    target_mean, target_std = target_scale_from_train(dataset.y_train)
    train_target = (dataset.y_train - target_mean) / target_std
    valid_target = (dataset.y_valid - target_mean) / target_std
    test_target = (dataset.y_test - target_mean) / target_std
    train_loader = _loader(dataset.x_train, train_target, experiment.batch_size)
    valid_loader = _loader(dataset.x_valid, valid_target, experiment.batch_size)
    test_loader = _loader(dataset.x_test, test_target, experiment.batch_size)
    model = _make_model(experiment, dataset.x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.learning_rate)
    criterion = nn.MSELoss()
    train_history: list[float] = []
    valid_history: list[float] = []
    best_loss, stale_epochs, best_state = float("inf"), 0, None

    for epoch in range(1, experiment.epochs + 1):
        model.train()
        train_total, train_count = 0.0, 0
        for features, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(target)
            train_count += len(target)
        # Loss train được tích lũy ngay khi tối ưu; tránh một lượt suy luận
        # lặp lại trên cùng tập huấn luyện ở mỗi epoch.
        train_history.append(train_total / train_count)
        valid_loss = _loss(model, valid_loader, criterion, device)
        valid_history.append(valid_loss)
        if valid_loss < best_loss:
            best_loss, stale_epochs, best_state = valid_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= experiment.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted_scaled = np.concatenate([model(features.to(device)).cpu().numpy() for features, _ in test_loader])
    predicted = predicted_scaled * target_std + target_mean
    return predicted, parameter_count(model), train_history, valid_history


def run_experiment(experiment: Experiment, show_alignment: bool = False) -> dict[str, float]:
    """Chạy đúng một cấu hình và lưu prediction/metric có định danh seed."""
    if experiment.model not in ALL_MODELS:
        raise ValueError(f"model phải thuộc {ALL_MODELS}")
    dataset = make_sequence_split(
        build_feature_frame(experiment.ticker),
        include_vnindex=experiment.include_vnindex,
        window=experiment.window,
    )
    if show_alignment:
        print_split_summary(dataset)
    if experiment.model in BASELINE_MODELS:
        predicted, count, train_history, valid_history = _evaluate_baseline(experiment, dataset)
    else:
        predicted, count, train_history, valid_history = _train_neural_model(experiment, dataset)
    metrics = regression_metrics(dataset.y_test, predicted)
    save_experiment_outputs(
        model=experiment.model,
        ticker=experiment.ticker,
        window=experiment.window,
        feature_set=experiment.feature_set,
        seed=experiment.seed,
        parameter_count=count,
        dates=dataset.dates_test,
        actual=dataset.y_test,
        predicted=predicted,
        metrics=metrics,
    )
    plot_learning_curve(experiment.model, experiment.ticker, experiment.seed, train_history, valid_history)
    plot_actual_vs_predicted(experiment.model, experiment.ticker, experiment.seed, dataset.dates_test, dataset.y_test, predicted)
    return metrics


def parse_args() -> Experiment:
    parser = argparse.ArgumentParser(description="Chạy một cấu hình dự báo return.")
    parser.add_argument("--model", choices=ALL_MODELS, required=True)
    parser.add_argument("--ticker", default="VCB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=12)
    return Experiment(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_experiment(parse_args(), show_alignment=True)
