"""Train one leakage-safe LSTM or Mamba next-day-return experiment."""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import (
    plot_actual_vs_predicted,
    plot_learning_curve,
    regression_metrics,
    save_experiment_outputs,
)
from models.lstm import LSTMRegressor
from models.mamba_model import MambaRegressor
from preprocess import build_feature_frame, make_sequence_split, print_split_summary


@dataclass(frozen=True)
class Experiment:
    model: str
    ticker: str = "VCB"
    include_vnindex: bool = False
    window: int = 30
    epochs: int = 75
    learning_rate: float = 1e-3
    hidden_size: int = 16
    layers: int = 2
    batch_size: int = 64
    patience: int = 12
    seed: int = 42

    @property
    def name(self) -> str:
        if self.model == "lstm":
            return "lstm"
        return "improved_mamba" if self.include_vnindex else "mamba_baseline"

    @property
    def feature_set(self) -> str:
        return "ohlcv_derived_plus_vnindex" if self.include_vnindex else "ohlcv_derived"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_model(experiment: Experiment, input_size: int) -> nn.Module:
    arguments = {"input_size": input_size, "hidden_size": experiment.hidden_size, "layers": experiment.layers}
    if experiment.model == "lstm":
        return LSTMRegressor(**arguments)
    if experiment.model == "mamba":
        return MambaRegressor(**arguments)
    raise ValueError(f"Unsupported model: {experiment.model}")


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    # Sequence order within each sample is never shuffled; training batches also stay chronological.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for features, target in loader:
            prediction = model(features.to(device))
            loss = criterion(prediction, target.to(device))
            total += loss.item() * len(target)
            count += len(target)
    return total / count


def run_experiment(experiment: Experiment) -> dict[str, float]:
    """Fit on train, choose epoch by validation loss, and evaluate only once on test."""
    set_seed(experiment.seed)
    frame = build_feature_frame(experiment.ticker)
    dataset = make_sequence_split(
        frame,
        include_vnindex=experiment.include_vnindex,
        window=experiment.window,
    )
    print_split_summary(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = _loader(dataset.x_train, dataset.y_train, experiment.batch_size)
    valid_loader = _loader(dataset.x_valid, dataset.y_valid, experiment.batch_size)
    test_loader = _loader(dataset.x_test, dataset.y_test, experiment.batch_size)
    model = _make_model(experiment, dataset.x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.learning_rate)
    criterion = nn.MSELoss()

    train_history: list[float] = []
    valid_history: list[float] = []
    best_loss, stale_epochs = float("inf"), 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, experiment.epochs + 1):
        model.train()
        for features, target in train_loader:
            optimizer.zero_grad()
            prediction = model(features.to(device))
            loss = criterion(prediction, target.to(device))
            loss.backward()
            optimizer.step()

        train_loss = _loss(model, train_loader, criterion, device)
        valid_loss = _loss(model, valid_loader, criterion, device)
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= experiment.patience:
                print(f"Early stopping at epoch {epoch}; best validation loss={best_loss:.8f}")
                break
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.8f}, valid_loss={valid_loss:.8f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in test_loader:
            predictions.append(model(features.to(device)).cpu().numpy())
    predicted = np.concatenate(predictions)
    metrics = regression_metrics(dataset.y_test, predicted)

    print("Test metrics:", ", ".join(f"{name}={value:.6f}" for name, value in metrics.items()))
    save_experiment_outputs(
        model=experiment.name,
        ticker=experiment.ticker,
        window=experiment.window,
        feature_set=experiment.feature_set,
        dates=dataset.dates_test,
        actual=dataset.y_test,
        predicted=predicted,
        metrics=metrics,
    )
    plot_learning_curve(experiment.name, experiment.ticker, train_history, valid_history)
    plot_actual_vs_predicted(experiment.name, experiment.ticker, dataset.dates_test, dataset.y_test, predicted)
    return metrics


def parse_args() -> Experiment:
    parser = argparse.ArgumentParser(description="Train one stock-return forecasting model.")
    parser.add_argument("--model", choices=("lstm", "mamba"), required=True)
    parser.add_argument("--ticker", default="VCB")
    parser.add_argument("--include-vnindex", action="store_true")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return Experiment(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_experiment(parse_args())
