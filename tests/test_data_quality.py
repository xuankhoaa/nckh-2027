from __future__ import annotations

import unittest

import pandas as pd

from data_quality import validate_ohlcv


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 11.0],
            "high": [12.0, 12.0],
            "low": [9.0, 10.0],
            "close": [11.0, 11.0],
            "volume": [100.0, 120.0],
        }
    )


class DataQualityTests(unittest.TestCase):
    def test_valid_stock_has_no_warning(self) -> None:
        self.assertEqual(validate_ohlcv(valid_frame(), "VCB"), [])

    def test_stock_close_outside_range_is_rejected(self) -> None:
        frame = valid_frame()
        frame.loc[0, "close"] = 13.0
        with self.assertRaises(ValueError):
            validate_ohlcv(frame, "VCB")

    def test_vnindex_close_outside_range_is_recorded(self) -> None:
        frame = valid_frame()
        frame.loc[0, "close"] = 13.0
        warnings = validate_ohlcv(frame, "VNINDEX")
        self.assertEqual(len(warnings), 1)
