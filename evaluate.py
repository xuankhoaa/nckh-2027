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
METRIC_NAMES = ("rmse", "mae", "r2_test", "r2_oos", "directional_accuracy", "qlike")


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target: str = "return",
    benchmark: np.ndarray | None = None,
) -> dict[str, float]:
    """Tính metric hồi quy sau khi kiểm tra kích thước và giá trị hữu hạn."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("actual và predicted phải là hai vector một chiều cùng kích thước.")
    if len(actual) < 2 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Dự báo cần ít nhất hai giá trị hữu hạn.")
    result: dict[str, float] = {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2_test": float(r2_score(actual, predicted)),
    }
    if benchmark is None:
        result["r2_oos"] = float("nan")
    else:
        benchmark = np.asarray(benchmark, dtype=float)
        if benchmark.shape != actual.shape or not np.isfinite(benchmark).all():
            raise ValueError("benchmark phải là vector hữu hạn cùng kích thước actual.")
        denominator = float(np.sum((actual - benchmark) ** 2))
        result["r2_oos"] = (
            float(1.0 - np.sum((actual - predicted) ** 2) / denominator)
            if denominator > 0
            else float("nan")
        )
    if target == "return":
        result["directional_accuracy"] = float((np.sign(actual) == np.sign(predicted)).mean())
        result["qlike"] = float("nan")
    elif target == "volatility":
        epsilon = np.finfo(float).eps
        actual_variance = np.maximum(actual, epsilon) ** 2
        predicted_variance = np.maximum(predicted, epsilon) ** 2
        ratio = actual_variance / predicted_variance
        result["directional_accuracy"] = float("nan")
        result["qlike"] = float(np.mean(ratio - np.log(ratio) - 1.0))
    else:
        raise ValueError("target phải là 'return' hoặc 'volatility'.")
    return result


def reset_generated_results() -> None:
    """Xóa đúng các đầu ra có thể tái tạo trước một lần chạy toàn bộ mới."""
    for filename in (
        "metrics.csv",
        "predictions.csv",
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
    for pattern in ("group_summary_*.csv", "per_ticker_summary_*.csv"):
        for path in RESULTS_DIR.glob(pattern):
            path.unlink()
    for directory in (PER_TICKER_PLOTS_DIR, GROUP_PLOTS_DIR):
        if directory.exists():
            shutil.rmtree(directory)


def _upsert_csv(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    """Thay đúng cùng định danh run, không cộng dồn trùng lặp khi chạy lại."""
    if path.exists():
        old_rows = pd.read_csv(path)
        if not set(keys).issubset(old_rows.columns):
            raise ValueError(f"{path} không cùng schema; hãy chạy lại không dùng --keep-results.")
        markers = old_rows.merge(new_rows.loc[:, keys].drop_duplicates(), on=keys, how="left", indicator=True)["_merge"]
        new_rows = pd.concat([old_rows.loc[markers.eq("left_only")], new_rows], ignore_index=True)
    new_rows.to_csv(path, index=False)


def save_experiment_outputs(
    *,
    model: str,
    ticker: str,
    target: str,
    horizon: int,
    window: int,
    feature_set: str,
    news_cutoff_hour: int,
    tail_weight: float,
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
        "target": target,
        "horizon": horizon,
        "window": window,
        "feature_set": feature_set,
        "news_cutoff_hour": news_cutoff_hour,
        "tail_weight": tail_weight,
        "seed": seed,
        "parameter_count": parameter_count,
    }
    _upsert_csv(RESULTS_DIR / "metrics.csv", pd.DataFrame([{**identity, **metrics}]), list(identity))
    predictions = pd.DataFrame(
        {
            **identity,
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "actual_value": actual,
            "predicted_value": predicted,
        }
    )
    _upsert_csv(RESULTS_DIR / "predictions.csv", predictions, [*identity, "date"])


def plot_learning_curve(
    model: str,
    ticker: str,
    target: str,
    horizon: int,
    window: int,
    feature_set: str,
    news_cutoff_hour: int,
    tail_weight: float,
    seed: int,
    train_loss: list[float],
    valid_loss: list[float],
) -> None:
    if not train_loss:
        return
    PER_TICKER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(train_loss, label="Huấn luyện")
    axis.plot(valid_loss, label="Xác thực")
    axis.set(title=f"Loss - {target}, {model}, {feature_set}, {ticker}, seed {seed}", xlabel="Epoch", ylabel="MSE có trọng số")
    axis.legend()
    figure.tight_layout()
    suffix = f"{target}_h{horizon}_w{window}_{ticker}_{model}_{feature_set}_c{news_cutoff_hour}_tw{tail_weight:g}_seed_{seed}"
    figure.savefig(PER_TICKER_PLOTS_DIR / f"loss_{suffix}.png", dpi=160)
    plt.close(figure)


def plot_actual_vs_predicted(
    model: str,
    ticker: str,
    target: str,
    horizon: int,
    window: int,
    feature_set: str,
    news_cutoff_hour: int,
    tail_weight: float,
    seed: int,
    dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    PER_TICKER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 4))
    label = "Log-return" if target == "return" else "Độ biến động thực hiện"
    axis.plot(dates, actual, label=f"{label} thực tế", linewidth=1)
    axis.plot(dates, predicted, label=f"{label} dự báo", linewidth=1)
    axis.set(title=f"{target} - {model}, {feature_set}, {ticker}, seed {seed}", xlabel="Ngày cuối kỳ mục tiêu", ylabel=label)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    suffix = f"{target}_h{horizon}_w{window}_{ticker}_{model}_{feature_set}_c{news_cutoff_hour}_tw{tail_weight:g}_seed_{seed}"
    figure.savefig(PER_TICKER_PLOTS_DIR / f"actual_vs_predicted_{suffix}.png", dpi=160)
    plt.close(figure)


def per_ticker_summary(window: int, target: str, horizon: int) -> pd.DataFrame:
    """Gộp các seed trước khi so sánh mô hình theo từng mã cổ phiếu."""
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    metrics = metrics.loc[
        metrics["window"].eq(window)
        & metrics["target"].eq(target)
        & metrics["horizon"].eq(horizon)
    ].copy()
    grouped = metrics.groupby(
        ["model", "ticker", "target", "horizon", "window", "feature_set", "news_cutoff_hour", "tail_weight"],
        as_index=False,
    )
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


def save_group_summary(window: int, target: str, horizon: int) -> pd.DataFrame:
    """Tổng hợp trung bình theo mã sau khi đã trung bình qua các seed."""
    per_ticker = per_ticker_summary(window, target, horizon)
    per_ticker_path = RESULTS_DIR / f"per_ticker_summary_{target}_h{horizon}_w{window}.csv"
    per_ticker.to_csv(per_ticker_path, index=False)
    grouped = per_ticker.groupby(
        ["model", "target", "horizon", "window", "feature_set", "news_cutoff_hour", "tail_weight"],
        as_index=False,
    )
    summary = grouped.agg(
        tickers=("ticker", "nunique"),
        seed_count=("seed_count", "max"),
        parameter_count=("parameter_count", "max"),
        **{f"{metric}_mean": (f"{metric}_mean", "mean") for metric in METRIC_NAMES},
        **{f"{metric}_std_ticker": (f"{metric}_mean", "std") for metric in METRIC_NAMES},
        **{f"{metric}_std_seed_mean": (f"{metric}_std_seed", "mean") for metric in METRIC_NAMES},
    )
    for metric in METRIC_NAMES:
        summary[f"{metric}_std_ticker"] = summary[f"{metric}_std_ticker"].fillna(0.0)
    summary.to_csv(RESULTS_DIR / f"group_summary_{target}_h{horizon}_w{window}.csv", index=False)
    return summary


def plot_group_metric_comparison(window: int, target: str, horizon: int) -> None:
    summary = save_group_summary(window, target, horizon)
    GROUP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    target_metrics = (("rmse", "RMSE trung bình"), ("mae", "MAE trung bình"))
    target_metrics += (("directional_accuracy", "Độ chính xác hướng trung bình"),) if target == "return" else (("qlike", "QLIKE trung bình"),)
    for metric, title in target_metrics:
        figure, axis = plt.subplots(figsize=(9, 4))
        labels = summary["model"] + "\n" + summary["feature_set"]
        axis.bar(labels, summary[f"{metric}_mean"], yerr=summary[f"{metric}_std_ticker"], capsize=4)
        axis.set(title=f"{title} - {target}", xlabel="Mô hình / tập đặc trưng", ylabel=metric)
        axis.tick_params(axis="x", labelrotation=20)
        figure.tight_layout()
        figure.savefig(GROUP_PLOTS_DIR / f"group_{target}_{metric}_h{horizon}_w{window}.png", dpi=160)
        plt.close(figure)
