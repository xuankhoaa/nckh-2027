"""Đánh giá, lưu kết quả và trực quan hóa thí nghiệm có thể tái lập."""

from __future__ import annotations

import shutil
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
METRIC_NAMES = ("rmse", "mae", "r2", "directional_accuracy")


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Tính metric hồi quy sau khi kiểm tra kích thước và giá trị hữu hạn."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("actual và predicted phải là hai vector một chiều cùng kích thước.")
    if len(actual) < 2 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Dự báo cần ít nhất hai giá trị hữu hạn.")
    return {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "directional_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
    }


def reset_generated_results() -> None:
    """Xóa đúng các đầu ra có thể tái tạo trước một lần chạy toàn bộ mới."""
    for filename in (
        "metrics.csv",
        "predictions.csv",
        "group_summary.csv",
        "per_ticker_summary.csv",
        "statistical_tests.csv",
        "BAO_CAO_KET_QUA.md",
        "BANG_KET_QUA_CHI_TIET_VI.csv",
        "BANG_TONG_HOP_NHOM_VI.csv",
        "KIEM_DINH_HAC_VI.csv",
        "DU_BAO_THEO_NGAY_VI.csv",
        "EXPERIMENT_MANIFEST.json",
    ):
        path = RESULTS_DIR / filename
        if path.exists():
            path.unlink()
    for directory in (PER_TICKER_PLOTS_DIR, GROUP_PLOTS_DIR):
        if directory.exists():
            shutil.rmtree(directory)


def _upsert_csv(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    """Thay đúng cùng định danh run, không cộng dồn trùng lặp khi chạy lại."""
    if path.exists():
        old_rows = pd.read_csv(path)
        if not set(keys).issubset(old_rows.columns):
            raise ValueError(f"{path} không cùng schema; hãy chạy lại với --reset-results.")
        markers = old_rows.merge(new_rows.loc[:, keys].drop_duplicates(), on=keys, how="left", indicator=True)["_merge"]
        new_rows = pd.concat([old_rows.loc[markers.eq("left_only")], new_rows], ignore_index=True)
    new_rows.to_csv(path, index=False)


def save_experiment_outputs(
    *,
    model: str,
    ticker: str,
    window: int,
    feature_set: str,
    seed: int,
    parameter_count: int,
    dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Lưu metric và dự báo với định danh đầy đủ của từng lần chạy."""
    RESULTS_DIR.mkdir(exist_ok=True)
    identity = {
        "model": model,
        "ticker": ticker,
        "window": window,
        "feature_set": feature_set,
        "seed": seed,
        "parameter_count": parameter_count,
    }
    _upsert_csv(RESULTS_DIR / "metrics.csv", pd.DataFrame([{**identity, **metrics}]), list(identity))
    predictions = pd.DataFrame(
        {
            **identity,
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "actual_return": actual,
            "predicted_return": predicted,
        }
    )
    _upsert_csv(RESULTS_DIR / "predictions.csv", predictions, [*identity, "date"])


def plot_learning_curve(model: str, ticker: str, seed: int, train_loss: list[float], valid_loss: list[float]) -> None:
    if not train_loss:
        return
    PER_TICKER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(train_loss, label="Huấn luyện")
    axis.plot(valid_loss, label="Xác thực")
    axis.set(title=f"MSE theo epoch - {model}, {ticker}, seed {seed}", xlabel="Epoch", ylabel="MSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(PER_TICKER_PLOTS_DIR / f"loss_{ticker}_{model}_seed_{seed}.png", dpi=160)
    plt.close(figure)


def plot_actual_vs_predicted(
    model: str, ticker: str, seed: int, dates: np.ndarray, actual: np.ndarray, predicted: np.ndarray
) -> None:
    PER_TICKER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(dates, actual, label="Log return thực tế", linewidth=1)
    axis.plot(dates, predicted, label="Log return dự báo", linewidth=1)
    axis.set(title=f"Dự báo và thực tế - {model}, {ticker}, seed {seed}", xlabel="Ngày mục tiêu", ylabel="Log return")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(PER_TICKER_PLOTS_DIR / f"actual_vs_predicted_{ticker}_{model}_seed_{seed}.png", dpi=160)
    plt.close(figure)


def per_ticker_summary(window: int) -> pd.DataFrame:
    """Gộp các seed trước khi so sánh mô hình theo từng mã cổ phiếu."""
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    metrics = metrics.loc[metrics["window"] == window].copy()
    grouped = metrics.groupby(["model", "ticker", "window", "feature_set"], as_index=False)
    result = grouped.agg(
        seed_count=("seed", "nunique"),
        parameter_count=("parameter_count", "max"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRIC_NAMES},
        **{f"{metric}_std_seed": (metric, "std") for metric in METRIC_NAMES},
    )
    # Hai baseline là dự báo xác định, chỉ chạy seed 0. Độ biến thiên giữa seed
    # của chúng bằng 0 theo thiết kế, thay vì để NaN làm khó đọc bảng báo cáo.
    for metric in METRIC_NAMES:
        result[f"{metric}_std_seed"] = result[f"{metric}_std_seed"].fillna(0.0)
    return result


def save_group_summary(window: int) -> pd.DataFrame:
    """Tổng hợp trung bình theo mã sau khi đã trung bình qua các seed."""
    per_ticker = per_ticker_summary(window)
    per_ticker.to_csv(RESULTS_DIR / "per_ticker_summary.csv", index=False)
    grouped = per_ticker.groupby(["model", "window", "feature_set"], as_index=False)
    summary = grouped.agg(
        tickers=("ticker", "nunique"),
        seed_count=("seed_count", "max"),
        parameter_count=("parameter_count", "max"),
        **{f"{metric}_mean": (f"{metric}_mean", "mean") for metric in METRIC_NAMES},
        **{f"{metric}_std_ticker": (f"{metric}_mean", "std") for metric in METRIC_NAMES},
        **{f"{metric}_std_seed_mean": (f"{metric}_std_seed", "mean") for metric in METRIC_NAMES},
    )
    summary.to_csv(RESULTS_DIR / "group_summary.csv", index=False)
    return summary


def plot_group_metric_comparison(window: int) -> None:
    summary = save_group_summary(window)
    GROUP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric, title in (("rmse", "RMSE trung bình"), ("mae", "MAE trung bình"), ("directional_accuracy", "Độ chính xác hướng trung bình")):
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar(summary["model"], summary[f"{metric}_mean"], yerr=summary[f"{metric}_std_ticker"], capsize=4)
        axis.set(title=f"{title} theo mô hình", xlabel="Mô hình", ylabel=metric)
        figure.tight_layout()
        figure.savefig(GROUP_PLOTS_DIR / f"group_{metric}_window_{window}.png", dpi=160)
        plt.close(figure)
