from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import TimeSplit
from preprocess import build_feature_frame, make_sequence_split


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
