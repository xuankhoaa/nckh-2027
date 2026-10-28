"""Kiểm tra mã, dữ liệu và tính đầy đủ của ma trận thực nghiệm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import TICKERS
from console import configure_utf8_output
from data_quality import data_inventory, sha256_file, sha256_text_file
from evaluate import RESULTS_DIR


IDENTITY = [
    "model", "ticker", "target", "horizon", "window", "feature_set",
    "news_cutoff_hour", "tail_weight", "loss_name", "weight_decay",
    "validation_shrinkage", "seed", "parameter_count",
]

# Manifest gốc ghi lại toàn bộ mã lúc huấn luyện. Công cụ điều phối và kiểm tra
# có thể được sửa sau đó mà không làm thay đổi các dự báo đã công bố.
PREDICTION_SOURCE_FILES = {
    "config.py", "data_quality.py", "phobert_sentiment.py", "preprocess.py",
    "baselines.py", "models/lstm.py", "models/mamba_model.py",
    "train.py", "evaluate.py",
}


def _boolean_series(values: pd.Series) -> pd.Series:
    """Đọc cột bool ổn định dù pandas trả về bool hay chuỗi."""
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("Cột validation_shrinkage phải chỉ chứa true/false.")
    return normalized.eq("true")


def assert_source_layout() -> None:
    expected = (
        "config.py", "phobert_sentiment.py", "evaluate_sentiment.py",
        "compare_sentiment_methods.py",
        "data_quality.py", "preprocess.py",
        "baselines.py", "train.py", "run_experiments.py", "evaluate.py",
        "generate_report.py", "provenance.py", "docs/NEWS_DATA_SCHEMA.md",
        "models/lstm.py", "models/mamba_model.py",
    )
    missing = [name for name in expected if not Path(name).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu tệp cốt lõi: {missing}")


def assert_data_cache() -> None:
    inventory = data_inventory(Path("data"), [*TICKERS, "VNINDEX"], source="KBS")
    print(f"Dữ liệu thị trường: {len(inventory['files'])} tệp OHLCV hợp lệ.")
    for warning in inventory["quality_warnings"]:
        print(f"Cảnh báo dữ liệu: {warning}")


def _expected_runs(manifest: dict[str, object]) -> int:
    experiment = manifest["experiment"]
    total = 0
    for target in experiment["targets"]:
        horizons = experiment["return_horizons"] if target == "return" else [experiment["volatility_horizon"]]
        constant_count = 2 if target == "return" else 1
        ridge_count = len(experiment["feature_sets"]) if experiment.get("include_ridge", False) else 0
        neural_count = 0
        for feature_set in experiment["feature_sets"]:
            cutoff_count = len(experiment["news_cutoff_hours"]) if "news" in feature_set else 1
            neural_count += 2 * len(experiment["seeds"]) * len(experiment["tail_weights"]) * cutoff_count
        total += (
            len(horizons)
            * len(experiment["tickers"])
            * len(experiment["windows"])
            * (constant_count + ridge_count + neural_count)
        )
    return total


def assert_metric_reproduction(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Tính lại metric từ dự báo test, đối chiếu từng run với bảng công bố."""
    baseline = predictions.loc[predictions["model"].eq("train_mean")]
    baseline_keys = ["ticker", "target", "horizon", "window"]
    benchmark_by_group: dict[tuple[object, ...], pd.DataFrame] = {}
    for key, frame in baseline.groupby(baseline_keys, sort=False):
        if frame["date"].duplicated().any():
            raise ValueError(f"Train Mean có ngày test trùng lặp: {key}")
        benchmark_by_group[key] = frame.set_index("date")

    indexed_metrics = metrics.set_index(IDENTITY)
    for identity, frame in predictions.groupby(IDENTITY, sort=False):
        if frame["date"].duplicated().any():
            raise ValueError(f"Một cấu hình có ngày dự báo trùng lặp: {identity}")
        row = indexed_metrics.loc[identity]
        actual = frame["actual_value"].to_numpy(dtype=float)
        forecast = frame["predicted_value"].to_numpy(dtype=float)
        squared_error = (actual - forecast) ** 2
        calculated = {
            "rmse": float(np.sqrt(squared_error.mean())),
            "mae": float(np.abs(actual - forecast).mean()),
            "r2_test": float(1 - squared_error.sum() / ((actual - actual.mean()) ** 2).sum()),
        }
        group_key = (identity[1], identity[2], identity[3], identity[4])
        if group_key not in benchmark_by_group:
            raise ValueError(f"Thiếu Train Mean cho {group_key}")
        reference = benchmark_by_group[group_key].reindex(frame["date"])
        if reference["predicted_value"].isna().any() or not np.allclose(
            reference["actual_value"].to_numpy(dtype=float), actual, rtol=0, atol=1e-8
        ):
            raise ValueError(f"Ngày/nhãn test không khớp Train Mean: {identity}")
        benchmark = reference["predicted_value"].to_numpy(dtype=float)
        calculated["r2_oos"] = float(1 - squared_error.sum() / ((actual - benchmark) ** 2).sum())
        if identity[2] == "return":
            calculated["directional_accuracy"] = float((np.sign(actual) == np.sign(forecast)).mean())
        else:
            epsilon = np.finfo(float).eps
            ratio = np.maximum(actual, epsilon) ** 2 / np.maximum(forecast, epsilon) ** 2
            calculated["qlike"] = float(np.mean(ratio - np.log(ratio) - 1))
        for name, value in calculated.items():
            if not np.isclose(float(row[name]), value, rtol=1e-6, atol=1e-8):
                raise ValueError(f"Metric {name} không khớp dự báo test của {identity}")


def assert_experiment_outputs() -> None:
    metrics_path = RESULTS_DIR / "metrics.csv"
    predictions_path = RESULTS_DIR / "predictions.csv"
    manifest_path = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    for path in (metrics_path, predictions_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu đầu ra thí nghiệm: {path}")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_sources = [
        name
        for name, expected_hash in manifest.get("source_sha256", {}).items()
        if name in PREDICTION_SOURCE_FILES
        if not Path(name).is_file() or sha256_text_file(Path(name)) != expected_hash
    ]
    if changed_sources:
        raise ValueError(f"Mã nguồn đã khác manifest của lần chạy: {changed_sources}")
    changed_data = [
        record["symbol"]
        for record in manifest.get("data_manifest", {}).get("files", [])
        if not (Path("data") / f"{record['symbol']}.csv").is_file()
        or sha256_file(Path("data") / f"{record['symbol']}.csv") != record["sha256"]
    ]
    if changed_data:
        raise ValueError(f"Dữ liệu OHLCV đã khác manifest của lần chạy: {changed_data}")
    news_info = manifest.get("news_data", {})
    news_path = Path(str(news_info.get("path", "")).replace("\\", "/"))
    if not news_path.is_file() or sha256_file(news_path) != news_info.get("sha256"):
        raise ValueError("Dữ liệu sentiment tin tức đã khác manifest của lần chạy.")
    collection_manifest = Path(
        str(news_info.get("collection_manifest_path", "")).replace("\\", "/")
    )
    if (
        not collection_manifest.is_file()
        or sha256_file(collection_manifest) != news_info.get("collection_manifest_sha256")
    ):
        raise ValueError("Manifest thu thập tin tức đã khác lần chạy thực nghiệm.")
    required_metrics = {
        *IDENTITY, "rmse", "mae", "r2_test", "r2_oos",
        "directional_accuracy", "qlike", "ridge_alpha", "shrinkage_alpha",
    }
    required_predictions = {*IDENTITY, "date", "actual_value", "predicted_value"}
    if not required_metrics.issubset(metrics.columns) or not required_predictions.issubset(predictions.columns):
        raise ValueError("Output thuộc schema cũ; hãy chạy lại không dùng --keep-results.")
    expected_runs = _expected_runs(manifest)
    if len(metrics) != expected_runs or metrics.duplicated(IDENTITY).any():
        raise ValueError(f"metrics.csv phải có đúng {expected_runs} lần chạy duy nhất; hiện có {len(metrics)}.")
    if set(metrics["ticker"]) != set(manifest["experiment"]["tickers"]):
        raise ValueError("Danh sách mã trong kết quả không khớp manifest.")
    if not np.isfinite(metrics[["rmse", "mae", "r2_test", "r2_oos"]].to_numpy(dtype=float)).all():
        raise ValueError("RMSE/MAE/R² có giá trị không hữu hạn.")
    return_rows = metrics["target"].eq("return")
    volatility_rows = metrics["target"].eq("volatility")
    if return_rows.any() and not np.isfinite(metrics.loc[return_rows, "directional_accuracy"]).all():
        raise ValueError("Bài toán return thiếu DA hữu hạn.")
    if volatility_rows.any() and not np.isfinite(metrics.loc[volatility_rows, "qlike"]).all():
        raise ValueError("Bài toán volatility thiếu QLIKE hữu hạn.")
    ridge_rows = metrics["model"].eq("ridge")
    if ridge_rows.any() and not np.isfinite(metrics.loc[ridge_rows, "ridge_alpha"]).all():
        raise ValueError("Ridge thiếu alpha được chọn trên validation.")
    shrunk_rows = metrics["model"].isin(["lstm", "mamba"]) & _boolean_series(
        metrics["validation_shrinkage"]
    )
    if shrunk_rows.any() and not metrics.loc[shrunk_rows, "shrinkage_alpha"].between(0, 1).all():
        raise ValueError("Hệ số validation shrinkage phải nằm trong [0, 1].")
    prediction_ids = predictions.loc[:, IDENTITY].drop_duplicates()
    metric_ids = metrics.loc[:, IDENTITY].drop_duplicates()
    paired = metric_ids.merge(prediction_ids, on=IDENTITY, how="outer", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Có metric không có dự báo tương ứng hoặc ngược lại.")
    if not np.isfinite(predictions[["actual_value", "predicted_value"]].to_numpy(dtype=float)).all():
        raise ValueError("Dự báo có giá trị không hữu hạn.")
    volatility = predictions["target"].eq("volatility")
    if volatility.any() and not predictions.loc[volatility, "predicted_value"].gt(0).all():
        raise ValueError("Dự báo volatility phải dương.")
    assert_metric_reproduction(metrics, predictions)
    print(f"Kết quả: {len(metrics)} lần chạy và {len(predictions)} dự báo hợp lệ.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Xác minh dự án NCKH.")
    parser.add_argument("--require-results", action="store_true")
    args = parser.parse_args()
    assert_source_layout()
    assert_data_cache()
    metrics_path = RESULTS_DIR / "metrics.csv"
    if metrics_path.exists() and "target" not in pd.read_csv(metrics_path, nrows=1).columns:
        if args.require_results:
            raise ValueError("results/ là kết quả pilot schema cũ; cần chạy lại ma trận hai bài toán.")
        print("Kết quả hiện có là pilot schema cũ; bỏ qua cho đến khi chạy lại ma trận mới.")
    elif metrics_path.exists() or args.require_results:
        assert_experiment_outputs()
    else:
        print("Chưa có output schema mới; bỏ qua kiểm tra kết quả.")
    print("Xác minh dự án hoàn tất.")


if __name__ == "__main__":
    configure_utf8_output()
    main()
