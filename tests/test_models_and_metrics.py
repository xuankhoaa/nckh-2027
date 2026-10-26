from __future__ import annotations

import unittest

import numpy as np
import torch

from baselines import predict_baseline, predict_ridge
from evaluate import regression_metrics
from models.lstm import LSTMRegressor
from models.mamba_model import MambaRegressor, selective_scan
from train import TailWeightedLoss, target_scale_from_train, validation_shrinkage_alpha


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

    def test_ridge_selects_alpha_on_validation_and_returns_finite_values(self) -> None:
        x_train = np.arange(48, dtype=np.float32).reshape(8, 3, 2) / 10
        y_train = x_train[:, -1, 0] * 0.5
        x_valid = np.arange(48, 72, dtype=np.float32).reshape(4, 3, 2) / 10
        y_valid = x_valid[:, -1, 0] * 0.5
        x_test = np.arange(72, 96, dtype=np.float32).reshape(4, 3, 2) / 10
        predicted, alpha, count = predict_ridge(
            x_train, y_train, x_valid, y_valid, x_test, (0.1, 1.0, 10.0)
        )
        self.assertIn(alpha, (0.1, 1.0, 10.0))
        self.assertEqual(len(predicted), len(x_test))
        self.assertEqual(count, x_train.shape[1] * x_train.shape[2] + 1)
        self.assertTrue(np.isfinite(predicted).all())

    def test_ridge_log_target_produces_positive_predictions(self) -> None:
        x_train = np.arange(48, dtype=np.float32).reshape(8, 3, 2) / 10
        y_train = np.linspace(0.01, 0.03, 8, dtype=np.float32)
        x_valid = np.arange(48, 72, dtype=np.float32).reshape(4, 3, 2) / 10
        y_valid = np.linspace(0.02, 0.03, 4, dtype=np.float32)
        x_test = -np.ones((4, 3, 2), dtype=np.float32) * 100
        predicted, _, _ = predict_ridge(
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            (1.0, 10.0),
            positive_target=True,
        )
        self.assertTrue((predicted > 0).all())

    def test_ridge_log_target_accepts_zero_realized_volatility(self) -> None:
        x_train = np.arange(36, dtype=np.float32).reshape(6, 3, 2)
        y_train = np.array([0.0, 0.01, 0.02, 0.0, 0.01, 0.03], dtype=np.float32)
        predicted, _, _ = predict_ridge(
            x_train,
            y_train,
            x_train[:2],
            y_train[:2],
            x_train[:2],
            (1.0,),
            positive_target=True,
        )
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted > 0).all())

    def test_validation_shrinkage_uses_clipped_least_squares_coefficient(self) -> None:
        actual = np.array([0.0, 2.0])
        predicted = np.array([0.0, 4.0])
        benchmark = np.zeros(2)
        self.assertAlmostEqual(validation_shrinkage_alpha(actual, predicted, benchmark), 0.5)
        self.assertEqual(
            validation_shrinkage_alpha(-actual, predicted, benchmark),
            0.0,
        )

    def test_huber_loss_is_less_sensitive_to_one_large_error_than_mse(self) -> None:
        actual = torch.zeros(2)
        predicted = torch.tensor([0.0, 10.0])
        huber = TailWeightedLoss(100.0, 1.0, "return", "huber")(predicted, actual)
        mse = TailWeightedLoss(100.0, 1.0, "return", "mse")(predicted, actual)
        self.assertLess(huber.item(), mse.item())
