"""Metrics, reproducible result tables, and report-ready plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RESULTS_DIR = Path("results")
PER_TICKER_PLOTS_DIR = RESULTS_DIR / "bieu_do" / "tung_ma"
GROUP_PLOTS_DIR = RESULTS_DIR / "bieu_do" / "tong_hop"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "directional_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
    }


def _upsert_csv(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        old_rows = pd.read_csv(path)
        key_values = new_rows.loc[:, keys].drop_duplicates()
        marker = old_rows.merge(key_values, on=keys, how="left", indicator=True)["_merge"]
        old_rows = old_rows.loc[marker.eq("left_only")]
        new_rows = pd.concat([old_rows, new_rows], ignore_index=True)
    new_rows.to_csv(path, index=False)


def save_experiment_outputs(
    model: str,
    ticker: str,
    window: int,
    feature_set: str,
    dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: dict[str, float],
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    identity = {"model": model, "ticker": ticker, "window": window, "feature_set": feature_set}
    metric_row = pd.DataFrame([{**identity, **metrics}])
    _upsert_csv(RESULTS_DIR / "metrics.csv", metric_row, list(identity))

    predictions = pd.DataFrame(
        {
            **identity,
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "actual_return": actual,
            "predicted_return": predicted,
        }
    )
    _upsert_csv(
        RESULTS_DIR / "predictions.csv",
        predictions,
        [*identity, "date"],
    )


def plot_learning_curve(model: str, ticker: str, train_loss: list[float], valid_loss: list[float]) -> None:
    PER_TICKER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(train_loss, label="train")
    axis.plot(valid_loss, label="validation")
    axis.set(title=f"Training loss — {model}, {ticker}", xlabel="Epoch", ylabel="MSE loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PER_TICKER_PLOTS_DIR / f"loss_{ticker}_{model}.png", dpi=160)
    plt.close(figure)


def plot_actual_vs_predicted(model: str, ticker: str, dates: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> None:
    PER_TICKER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(dates, actual, label="Actual return", linewidth=1)
    axis.plot(dates, predicted, label="Predicted return", linewidth=1)
    axis.set(title=f"Actual vs predicted next-day return — {model}, {ticker}", xlabel="Target date", ylabel="Log return")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(PER_TICKER_PLOTS_DIR / f"actual_vs_predicted_{ticker}_{model}.png", dpi=160)
    plt.close(figure)


def plot_metric_comparison(ticker: str, window: int) -> None:
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        return
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[(metrics["ticker"] == ticker) & (metrics["window"] == window)]
    if metrics.empty:
        return
    PER_TICKER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = metrics["model"]
    for metric, title in (
        ("rmse", "RMSE comparison"),
        ("mae", "MAE comparison"),
        ("directional_accuracy", "Directional Accuracy comparison"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(labels, metrics[metric])
        axis.set(title=f"{title} — {ticker}", xlabel="Model", ylabel=metric)
        figure.tight_layout()
        figure.savefig(PER_TICKER_PLOTS_DIR / f"{metric}_comparison_{ticker}.png", dpi=160)
        plt.close(figure)


def save_group_summary(window: int) -> pd.DataFrame:
    """Lưu trung bình và độ lệch chuẩn theo mô hình cho toàn bộ nhóm ngân hàng."""
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Chưa có results/metrics.csv để tổng hợp.")

    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["window"] == window].copy()
    columns = ["rmse", "mae", "r2", "directional_accuracy"]
    grouped = metrics.groupby(["model", "window", "feature_set"], as_index=False)
    summary = grouped.agg(
        tickers=("ticker", "nunique"),
        **{f"{column}_mean": (column, "mean") for column in columns},
        **{f"{column}_std": (column, "std") for column in columns},
    )
    summary.to_csv(RESULTS_DIR / "group_summary.csv", index=False)
    return summary


def plot_group_metric_comparison(window: int) -> None:
    """Vẽ biểu đồ trung bình nhóm ngân hàng từ bảng tổng hợp đã lưu."""
    summary = save_group_summary(window)
    GROUP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric, title in (
        ("rmse", "RMSE trung bình nhóm ngân hàng"),
        ("mae", "MAE trung bình nhóm ngân hàng"),
        ("directional_accuracy", "Directional Accuracy trung bình nhóm ngân hàng"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(summary["model"], summary[f"{metric}_mean"])
        axis.set(title=title, xlabel="Mô hình", ylabel=metric)
        figure.tight_layout()
        figure.savefig(GROUP_PLOTS_DIR / f"group_{metric}_comparison_window_{window}.png", dpi=160)
        plt.close(figure)
