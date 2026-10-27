"""Huấn luyện một cấu hình return/volatility với market/news ablation."""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from baselines import predict_baseline, predict_ridge
from config import (
    DEFAULT_NEWS_CUTOFF_HOUR,
    DEFAULT_RETURN_HORIZON,
    DEFAULT_VOLATILITY_HORIZON,
    DEFAULT_WINDOW,
    FEATURE_SETS,
    NEWS_FEATURES,
    TARGETS,
)
from console import configure_utf8_output
from evaluate import (
    plot_actual_vs_predicted,
    plot_learning_curve,
    regression_metrics,
    save_experiment_outputs,
)
from models.lstm import LSTMRegressor
from models.mamba_model import MambaRegressor
from preprocess import build_feature_frame, make_sequence_split, print_split_summary


NEURAL_MODELS = ("lstm", "mamba")
CONSTANT_BASELINES = ("naive_zero", "train_mean")
LEARNED_BASELINES = ("ridge",)
BASELINE_MODELS = (*CONSTANT_BASELINES, *LEARNED_BASELINES)
ALL_MODELS = (*BASELINE_MODELS, *NEURAL_MODELS)
LOSS_NAMES = ("mse", "huber")


@dataclass(frozen=True)
class Experiment:
    model: str
    ticker: str
    target: str = "return"
    feature_set: str = "market_only"
    seed: int = 42
    window: int = DEFAULT_WINDOW
    return_horizon: int = DEFAULT_RETURN_HORIZON
    volatility_horizon: int = DEFAULT_VOLATILITY_HORIZON
    news_path: str = "data/news_phobert.csv"
    news_cutoff_hour: int = DEFAULT_NEWS_CUTOFF_HOUR
    tail_weight: float = 1.0
    loss_name: str = "huber"
    weight_decay: float = 1e-4
    validation_shrinkage: bool = True
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
    epochs: int = 40
    learning_rate: float = 1e-3
    hidden_size: int = 8
    layers: int = 1
    batch_size: int = 128
    patience: int = 8

    @property
    def horizon(self) -> int:
        return self.return_horizon if self.target == "return" else self.volatility_horizon

    @property
    def include_news(self) -> bool:
        return any(feature in FEATURE_SETS[self.feature_set] for feature in NEWS_FEATURES)


def set_seed(seed: int) -> None:
    """Cố định các nguồn ngẫu nhiên để so sánh nhiều seed có thể tái lập."""
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
    """Ước lượng thang đo nhãn chỉ từ train để tránh rò rỉ tương lai."""
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= np.finfo(np.float32).eps:
        raise ValueError("Nhãn train không đủ biến thiên để chuẩn hóa.")
    return mean, std


class TailWeightedLoss(nn.Module):
    """MSE/Huber tùy chọn tăng trọng số cho 10% nhãn train cực trị."""

    def __init__(self, threshold: float, tail_weight: float, target: str, loss_name: str) -> None:
        super().__init__()
        if loss_name not in LOSS_NAMES:
            raise ValueError(f"loss_name phải thuộc {LOSS_NAMES}.")
        self.threshold = threshold
        self.tail_weight = tail_weight
        self.target = target
        self.loss_name = loss_name

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        score = actual.abs() if self.target == "return" else actual
        weights = torch.where(score >= self.threshold, self.tail_weight, 1.0)
        if self.loss_name == "huber":
            elementwise = functional.huber_loss(predicted, actual, reduction="none", delta=1.0)
        else:
            elementwise = (predicted - actual).pow(2)
        return torch.mean(weights * elementwise)


def validation_shrinkage_alpha(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: np.ndarray,
) -> float:
    """Ước lượng hệ số co [0, 1] chỉ từ validation, không nhìn test."""
    direction = np.asarray(predicted, dtype=float) - np.asarray(benchmark, dtype=float)
    residual = np.asarray(actual, dtype=float) - np.asarray(benchmark, dtype=float)
    denominator = float(np.dot(direction, direction))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.clip(np.dot(direction, residual) / denominator, 0.0, 1.0))


def _make_model(experiment: Experiment, input_size: int) -> nn.Module:
    arguments = {
        "input_size": input_size,
        "hidden_size": experiment.hidden_size,
        "layers": experiment.layers,
    }
    if experiment.model == "lstm":
        return LSTMRegressor(**arguments)
    if experiment.model == "mamba":
        return MambaRegressor(**arguments)
    raise ValueError(f"Mô hình mạng không hỗ trợ: {experiment.model}")


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for features, target in loader:
            loss = criterion(model(features.to(device)), target.to(device))
            total += loss.item() * len(target)
            count += len(target)
    return total / count


def _evaluate_baseline(
    experiment: Experiment, dataset
) -> tuple[np.ndarray, int, list[float], list[float], dict[str, float]]:
    if experiment.model == "ridge":
        predicted, alpha, count = predict_ridge(
            dataset.x_train,
            dataset.y_train,
            dataset.x_valid,
            dataset.y_valid,
            dataset.x_test,
            experiment.ridge_alphas,
            positive_target=(experiment.target == "volatility"),
        )
        return predicted, count, [], [], {"ridge_alpha": alpha, "shrinkage_alpha": float("nan")}
    predicted = predict_baseline(experiment.model, dataset.y_train, len(dataset.y_test))
    return predicted, 0, [], [], {"ridge_alpha": float("nan"), "shrinkage_alpha": float("nan")}


def _train_neural_model(
    experiment: Experiment, dataset
) -> tuple[np.ndarray, int, list[float], list[float], dict[str, float]]:
    set_seed(experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_mean, target_std = target_scale_from_train(dataset.y_train)
    train_target = (dataset.y_train - target_mean) / target_std
    valid_target = (dataset.y_valid - target_mean) / target_std
    train_loader = _loader(dataset.x_train, train_target, experiment.batch_size)
    valid_loader = _loader(dataset.x_valid, valid_target, experiment.batch_size)
    test_loader = _loader(dataset.x_test, dataset.y_test, experiment.batch_size)
    model = _make_model(experiment, dataset.x_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
    )
    tail_score = np.abs(train_target) if experiment.target == "return" else train_target
    criterion = TailWeightedLoss(
        threshold=float(np.quantile(tail_score, 0.9)),
        tail_weight=experiment.tail_weight,
        target=experiment.target,
        loss_name=experiment.loss_name,
    )
    train_history: list[float] = []
    valid_history: list[float] = []
    best_loss, stale_epochs, best_state = float("inf"), 0, None

    for _epoch in range(1, experiment.epochs + 1):
        model.train()
        train_total, train_count = 0.0, 0
        for features, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(target)
            train_count += len(target)
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
        valid_predicted_scaled = np.concatenate(
            [model(features.to(device)).cpu().numpy() for features, _ in valid_loader]
        )
        test_predicted_scaled = np.concatenate(
            [model(features.to(device)).cpu().numpy() for features, _ in test_loader]
        )
    valid_predicted = valid_predicted_scaled * target_std + target_mean
    predicted = test_predicted_scaled * target_std + target_mean
    shrinkage_alpha = 1.0
    if experiment.validation_shrinkage:
        valid_benchmark = np.full(len(dataset.y_valid), target_mean, dtype=float)
        shrinkage_alpha = validation_shrinkage_alpha(
            dataset.y_valid,
            valid_predicted,
            valid_benchmark,
        )
        predicted = target_mean + shrinkage_alpha * (predicted - target_mean)
    if experiment.target == "volatility":
        predicted = np.maximum(predicted, np.finfo(np.float32).eps)
    return (
        predicted,
        parameter_count(model),
        train_history,
        valid_history,
        {"ridge_alpha": float("nan"), "shrinkage_alpha": shrinkage_alpha},
    )


def run_experiment(experiment: Experiment, show_alignment: bool = False) -> dict[str, float]:
    """Chạy một cấu hình và lưu dự báo/metric với định danh đầy đủ."""
    if experiment.model not in ALL_MODELS:
        raise ValueError(f"model phải thuộc {ALL_MODELS}")
    if experiment.target not in TARGETS:
        raise ValueError(f"target phải thuộc {TARGETS}")
    if experiment.feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set phải thuộc {tuple(FEATURE_SETS)}")
    if experiment.tail_weight < 1:
        raise ValueError("tail_weight phải >= 1.")
    if experiment.loss_name not in LOSS_NAMES:
        raise ValueError(f"loss_name phải thuộc {LOSS_NAMES}.")
    if experiment.weight_decay < 0:
        raise ValueError("weight_decay phải không âm.")
    if experiment.model == "naive_zero" and experiment.target != "return":
        raise ValueError("naive_zero chỉ có ý nghĩa cho bài toán return.")

    frame = build_feature_frame(
        experiment.ticker,
        include_news=experiment.include_news,
        news_path=experiment.news_path,
        news_cutoff_hour=experiment.news_cutoff_hour,
        return_horizon=experiment.return_horizon,
        volatility_horizon=experiment.volatility_horizon,
    )
    dataset = make_sequence_split(
        frame,
        window=experiment.window,
        feature_set=experiment.feature_set,
        target=experiment.target,
    )
    if show_alignment:
        print_split_summary(dataset)
    if experiment.model in BASELINE_MODELS:
        predicted, count, train_history, valid_history, diagnostics = _evaluate_baseline(experiment, dataset)
    else:
        predicted, count, train_history, valid_history, diagnostics = _train_neural_model(experiment, dataset)
    benchmark = predict_baseline("train_mean", dataset.y_train, len(dataset.y_test))
    metrics = regression_metrics(dataset.y_test, predicted, experiment.target, benchmark=benchmark)
    saved_feature_set = (
        "not_applicable" if experiment.model in CONSTANT_BASELINES else experiment.feature_set
    )
    save_experiment_outputs(
        model=experiment.model,
        ticker=experiment.ticker,
        target=experiment.target,
        horizon=experiment.horizon,
        window=experiment.window,
        feature_set=saved_feature_set,
        news_cutoff_hour=experiment.news_cutoff_hour,
        tail_weight=experiment.tail_weight,
        loss_name=(experiment.loss_name if experiment.model in NEURAL_MODELS else "not_applicable"),
        weight_decay=(experiment.weight_decay if experiment.model in NEURAL_MODELS else 0.0),
        validation_shrinkage=(experiment.validation_shrinkage if experiment.model in NEURAL_MODELS else False),
        seed=experiment.seed,
        parameter_count=count,
        dates=dataset.dates_test,
        actual=dataset.y_test,
        predicted=predicted,
        metrics={**metrics, **diagnostics},
    )
    plot_learning_curve(
        experiment.model,
        experiment.ticker,
        experiment.target,
        experiment.horizon,
        experiment.window,
        experiment.feature_set,
        experiment.news_cutoff_hour,
        experiment.tail_weight,
        experiment.seed,
        train_history,
        valid_history,
    )
    plot_actual_vs_predicted(
        experiment.model,
        experiment.ticker,
        experiment.target,
        experiment.horizon,
        experiment.window,
        experiment.feature_set,
        experiment.news_cutoff_hour,
        experiment.tail_weight,
        experiment.seed,
        dataset.dates_test,
        dataset.y_test,
        predicted,
    )
    return metrics


def parse_args() -> Experiment:
    parser = argparse.ArgumentParser(description="Chạy một cấu hình return hoặc volatility.")
    parser.add_argument("--model", choices=ALL_MODELS, required=True)
    parser.add_argument("--ticker", default="VCB")
    parser.add_argument("--target", choices=TARGETS, default="return")
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="market_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--return-horizon", type=int, default=DEFAULT_RETURN_HORIZON)
    parser.add_argument("--volatility-horizon", type=int, default=DEFAULT_VOLATILITY_HORIZON)
    parser.add_argument("--news-path", default="data/news_phobert.csv")
    parser.add_argument("--news-cutoff-hour", type=int, default=DEFAULT_NEWS_CUTOFF_HOUR)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--loss-name", choices=LOSS_NAMES, default="huber")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-validation-shrinkage", dest="validation_shrinkage", action="store_false")
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    arguments = vars(parser.parse_args())
    arguments["ridge_alphas"] = tuple(arguments["ridge_alphas"])
    return Experiment(**arguments)


if __name__ == "__main__":
    configure_utf8_output()
    run_experiment(parse_args(), show_alignment=True)
