"""Kiểm tra nhanh tính toàn vẹn của mã, dữ liệu cache và kết quả thực nghiệm."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from console import configure_utf8_output
from config import RESEARCH_SEEDS, TICKERS
from data_quality import data_inventory
from evaluate import RESULTS_DIR


def assert_source_layout() -> None:
    """Bảo đảm các thành phần cốt lõi không bị thiếu sau khi dọn dẹp kho mã."""
    expected = (
        "config.py",
        "get_data.py",
        "data_quality.py",
        "preprocess.py",
        "baselines.py",
        "train.py",
        "run_experiments.py",
        "evaluate.py",
        "generate_report.py",
        "console.py",
        "models/lstm.py",
        "models/mamba_model.py",
    )
    missing = [name for name in expected if not Path(name).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu tệp mã cốt lõi: {missing}")


def assert_data_cache() -> None:
    """Kiểm tra OHLCV gốc, gồm tính đầy đủ và quy tắc chất lượng dữ liệu."""
    inventory = data_inventory(Path("data"), [*TICKERS, "VNINDEX"], source="KBS")
    print(f"Dữ liệu: {len(inventory['files'])} tệp OHLCV hợp lệ.")
    for warning in inventory["quality_warnings"]:
        print(f"Cảnh báo dữ liệu: {warning}")


def assert_experiment_outputs() -> None:
    """Kiểm tra đúng thiết kế 136 lần chạy và tính nhất quán prediction/metric."""
    metrics_path = RESULTS_DIR / "metrics.csv"
    predictions_path = RESULTS_DIR / "predictions.csv"
    manifest_path = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    for path in (metrics_path, predictions_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu đầu ra thí nghiệm: {path}")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    identity = ["model", "ticker", "window", "feature_set", "seed", "parameter_count"]
    expected_runs = len(TICKERS) * (2 + 3 * len(RESEARCH_SEEDS))
    if len(metrics) != expected_runs or metrics.duplicated(identity).any():
        raise ValueError(f"metrics.csv phải có đúng {expected_runs} lần chạy duy nhất.")
    if set(metrics["ticker"]) != set(TICKERS):
        raise ValueError("metrics.csv không chứa đủ tám mã cổ phiếu quy định.")
    metric_columns = ["rmse", "mae", "r2", "directional_accuracy"]
    if not np.isfinite(metrics.loc[:, metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("metrics.csv có metric không hữu hạn.")

    prediction_ids = predictions.loc[:, identity].drop_duplicates()
    metric_ids = metrics.loc[:, identity].drop_duplicates()
    paired = metric_ids.merge(prediction_ids, on=identity, how="outer", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Có lần chạy có metric nhưng không có dự báo, hoặc ngược lại.")
    if not np.isfinite(predictions[["actual_return", "predicted_return"]].to_numpy(dtype=float)).all():
        raise ValueError("predictions.csv có return không hữu hạn.")
    print(f"Kết quả: {len(metrics)} lần chạy và {len(predictions)} dự báo hợp lệ.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Xác minh thành phần chính của đề tài NCKH.")
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Báo lỗi nếu chưa có đủ kết quả 136 lần chạy.",
    )
    args = parser.parse_args()
    assert_source_layout()
    assert_data_cache()
    if (RESULTS_DIR / "metrics.csv").exists() or args.require_results:
        assert_experiment_outputs()
    else:
        print("Kết quả chưa được tạo; bỏ qua kiểm tra output. Hãy chạy run_experiments.py.")
    print("Xác minh dự án hoàn tất.")


if __name__ == "__main__":
    configure_utf8_output()
    main()
