from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_SETS, STATIONARY_MARKET_FEATURES, TimeSplit
from preprocess import aggregate_news_features, build_feature_frame, make_sequence_split


def write_ohlcv(path: Path, scale: float = 1.0, late_spike: bool = False) -> None:
    dates = pd.bdate_range("2023-01-02", periods=90)
    close = (100 + np.arange(len(dates), dtype=float)) * scale
    if late_spike:
        close[-1] = 1_000_000.0
    frame = pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + np.arange(len(dates)),
        }
    )
    frame.to_csv(path, index=False)


class PreprocessTests(unittest.TestCase):
    def test_news_ablations_remove_exactly_one_component(self) -> None:
        """Ablation giữ cùng market features và chỉ bỏ đúng một biến tin."""
        market = set(STATIONARY_MARKET_FEATURES)
        self.assertEqual(
            set(FEATURE_SETS["market_news_no_sentiment"]),
            market | {"news_count"},
        )
        self.assertEqual(
            set(FEATURE_SETS["market_news_no_count"]),
            market | {"sentiment_score"},
        )

    def make_data(self, directory: Path, spike: bool = False) -> None:
        write_ohlcv(directory / "VCB.csv", late_spike=spike)
        write_ohlcv(directory / "VNINDEX.csv")

    def test_target_is_after_input_and_splits_are_chronological(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.make_data(directory)
            frame = build_feature_frame("VCB", directory)
            split = TimeSplit(train_end="2023-03-01", validation_end="2023-03-31")
            dataset = make_sequence_split(frame, include_vnindex=True, window=10, split=split)
            self.assertLess(dataset.alignment["input_end_date"], dataset.alignment["target_date"])
            self.assertTrue((dataset.dates_train <= np.datetime64("2023-03-01")).all())
            self.assertTrue((dataset.dates_valid > np.datetime64("2023-03-01")).all())
            self.assertTrue((dataset.dates_valid <= np.datetime64("2023-03-31")).all())
            self.assertTrue((dataset.dates_test > np.datetime64("2023-03-31")).all())

    def test_future_value_does_not_change_training_features(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_dir, second_dir = Path(first), Path(second)
            self.make_data(first_dir, spike=False)
            self.make_data(second_dir, spike=True)
            split = TimeSplit(train_end="2023-03-01", validation_end="2023-03-31")
            first_set = make_sequence_split(build_feature_frame("VCB", first_dir), False, 10, split)
            second_set = make_sequence_split(build_feature_frame("VCB", second_dir), False, 10, split)
            np.testing.assert_allclose(first_set.x_train, second_set.x_train)

    def test_volatility_target_uses_only_next_five_returns(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.make_data(directory)
            frame = build_feature_frame("VCB", directory, volatility_horizon=5)
            row = frame.iloc[10]
            full = pd.read_csv(directory / "VCB.csv")
            returns = np.log(full["close"] / full["close"].shift(1))
            source_index = full.index[full["time"].eq(row["time"].strftime("%Y-%m-%d"))][0]
            expected = np.sqrt(np.mean(np.square(returns.iloc[source_index + 1 : source_index + 6])))
            self.assertAlmostEqual(row["target_volatility"], expected)
            self.assertEqual(
                row["target_volatility_date"],
                pd.Timestamp(full.iloc[source_index + 5]["time"]),
            )

    def test_five_day_return_is_sum_of_next_five_log_returns(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.make_data(directory)
            frame = build_feature_frame("VCB", directory, return_horizon=5)
            row = frame.iloc[10]
            full = pd.read_csv(directory / "VCB.csv")
            returns = np.log(full["close"] / full["close"].shift(1))
            source_index = full.index[full["time"].eq(row["time"].strftime("%Y-%m-%d"))][0]
            expected = returns.iloc[source_index + 1 : source_index + 6].sum()
            self.assertAlmostEqual(row["target_return"], expected)
            self.assertEqual(
                row["target_return_date"],
                pd.Timestamp(full.iloc[source_index + 5]["time"]),
            )

    def test_stationary_feature_set_has_no_missing_values_in_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.make_data(directory)
            frame = build_feature_frame("VCB", directory)
            split = TimeSplit(train_end="2023-03-15", validation_end="2023-04-10")
            dataset = make_sequence_split(
                frame,
                window=10,
                split=split,
                feature_set="market_stationary",
            )
            self.assertTrue(np.isfinite(dataset.x_train).all())
            self.assertIn("relative_return", dataset.feature_names)
            self.assertNotIn("close", dataset.feature_names)

    def test_news_after_cutoff_and_weekend_move_to_next_trading_day(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            news_path = directory / "news.csv"
            pd.DataFrame(
                {
                    "published_at": [
                        "2023-01-06T14:00:00+07:00",
                        "2023-01-06T16:00:00+07:00",
                        "2023-01-07T10:00:00+07:00",
                    ],
                    "ticker": ["VCB", "VCB", "VCB"],
                    "sentiment_score": [0.5, -0.5, 1.0],
                }
            ).to_csv(news_path, index=False)
            trading_dates = pd.Series(pd.bdate_range("2023-01-02", periods=10))
            daily = aggregate_news_features(news_path, "VCB", trading_dates, cutoff_hour=15)
            friday = daily.loc[daily["time"].eq(pd.Timestamp("2023-01-06"))].iloc[0]
            monday = daily.loc[daily["time"].eq(pd.Timestamp("2023-01-09"))].iloc[0]
            self.assertEqual(friday["news_count"], 1)
            self.assertAlmostEqual(friday["sentiment_score"], 0.5)
            self.assertEqual(monday["news_count"], 2)
            self.assertAlmostEqual(monday["sentiment_score"], 0.25)
