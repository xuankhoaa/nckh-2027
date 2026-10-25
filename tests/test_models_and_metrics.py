from __future__ import annotations

import unittest

import numpy as np
import torch

from baselines import predict_baseline
from evaluate import regression_metrics
from models.lstm import LSTMRegressor
from models.mamba_model import MambaRegressor, selective_scan
from train import target_scale_from_train


class ModelAndMetricTests(unittest.TestCase):
    def test_models_return_one_value_per_sample(self) -> None:
        inputs = torch.randn(4, 30, 9)
        for model in (LSTMRegressor(9), MambaRegressor(9)):
            output = model(inputs)
            self.assertEqual(tuple(output.shape), (4,))
            self.assertTrue(torch.isfinite(output).all())

    def test_perfect_prediction_has_zero_error(self) -> None:
        actual = np.array([-0.02, 0.01, 0.03])
        metrics = regression_metrics(actual, actual.copy())
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["directional_accuracy"], 1.0)

    def test_volatility_metrics_include_qlike(self) -> None:
        actual = np.array([0.01, 0.02, 0.03])
        metrics = regression_metrics(actual, actual.copy(), target="volatility")
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["qlike"], 0.0)
        self.assertTrue(np.isnan(metrics["directional_accuracy"]))

    def test_oos_r2_uses_train_mean_benchmark(self) -> None:
        actual = np.array([-0.02, 0.01, 0.03])
        benchmark = np.full(3, 0.005)
        baseline_metrics = regression_metrics(actual, benchmark, benchmark=benchmark)
        perfect_metrics = regression_metrics(actual, actual, benchmark=benchmark)
        self.assertAlmostEqual(baseline_metrics["r2_oos"], 0.0)
        self.assertAlmostEqual(perfect_metrics["r2_oos"], 1.0)

    def test_parallel_scan_matches_sequential_recurrence(self) -> None:
        """Quét tiền tố phải giữ nguyên kết quả của h_t = a_t h_(t-1) + b_t."""
        torch.manual_seed(7)
        transitions = torch.rand(2, 7, 3, 4)
        inputs = torch.randn(2, 7, 3, 4)
        state = torch.zeros(2, 3, 4)
        expected = []
        for index in range(transitions.shape[1]):
            state = transitions[:, index] * state + inputs[:, index]
            expected.append(state)
        sequential = torch.stack(expected, dim=1)
        self.assertTrue(torch.allclose(selective_scan(transitions, inputs), sequential, atol=1e-6))

    def test_target_scaling_is_fit_from_training_values(self) -> None:
        mean, std = target_scale_from_train(np.array([-0.02, 0.00, 0.02], dtype=np.float32))
        self.assertAlmostEqual(mean, 0.0, places=7)
        self.assertAlmostEqual(std, float(np.std(np.array([-0.02, 0.00, 0.02], dtype=np.float32))), places=7)

    def test_baselines_use_only_expected_training_rule(self) -> None:
        y_train = np.array([-0.01, 0.03], dtype=np.float32)
        self.assertTrue(np.array_equal(predict_baseline("naive_zero", y_train, 3), np.zeros(3, dtype=np.float32)))
        self.assertTrue(np.allclose(predict_baseline("train_mean", y_train, 3), 0.01))
