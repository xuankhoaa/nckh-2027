"""Kiểm thử các lỗi có thể làm báo cáo kết quả sai khi chạy lại."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_experiments import _completed_keys
from verify_project import IDENTITY, assert_metric_reproduction


class ReproducibilityTests(unittest.TestCase):
    def test_resume_requires_both_metrics_and_predictions(self) -> None:
        identity = {
            "model": "mamba", "ticker": "VCB", "target": "return",
            "horizon": 1, "window": 30, "feature_set": "market_stationary",
            "news_cutoff_hour": 15, "tail_weight": 1.0, "loss_name": "huber",
            "weight_decay": 0.0001, "validation_shrinkage": True, "seed": 42,
        }
        with tempfile.TemporaryDirectory() as raw:
            original = Path.cwd()
            os.chdir(raw)
            try:
                Path("results").mkdir()
                pd.DataFrame([identity]).to_csv("results/metrics.csv", index=False)
                self.assertEqual(_completed_keys(), set())
                pd.DataFrame([{**identity, "date": "2025-01-02"}]).to_csv(
                    "results/predictions.csv", index=False
                )
                self.assertEqual(len(_completed_keys()), 1)
            finally:
                os.chdir(original)

    def test_metric_audit_rejects_a_modified_rmse(self) -> None:
        common = {
            "ticker": "VCB", "target": "return", "horizon": 1, "window": 30,
            "news_cutoff_hour": 15, "tail_weight": 1.0,
            "loss_name": "not_applicable", "weight_decay": 0.0,
            "validation_shrinkage": False, "seed": 0, "parameter_count": 0,
        }
        actual = np.array([1.0, 2.0, 3.0])
        dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
        records = []
        metric_rows = []
        for model, forecast in (("train_mean", np.full(3, 2.0)), ("naive_zero", np.zeros(3))):
            identity = {**common, "model": model, "feature_set": "not_applicable"}
            metric_rows.append({
                **identity,
                "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
                "mae": float(np.mean(np.abs(actual - forecast))),
                "r2_test": float(1 - np.sum((actual - forecast) ** 2) / 2),
                "r2_oos": float(1 - np.sum((actual - forecast) ** 2) / 2),
                "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(forecast))),
            })
            records.extend(
                {**identity, "date": date, "actual_value": y, "predicted_value": yhat}
                for date, y, yhat in zip(dates, actual, forecast)
            )
        metrics = pd.DataFrame(metric_rows).loc[:, [*IDENTITY, "rmse", "mae", "r2_test", "r2_oos", "directional_accuracy"]]
        predictions = pd.DataFrame(records)
        assert_metric_reproduction(metrics, predictions)
        metrics.loc[metrics["model"].eq("naive_zero"), "rmse"] += 0.01
        with self.assertRaisesRegex(ValueError, "Metric rmse"):
            assert_metric_reproduction(metrics, predictions)


if __name__ == "__main__":
    unittest.main()
