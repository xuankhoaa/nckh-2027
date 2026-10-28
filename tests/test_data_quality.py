from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_quality import validate_news_coverage, validate_ohlcv


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
    def test_text_checksum_is_independent_of_line_endings(self) -> None:
        from data_quality import sha256_text_file

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            lf_path = directory / "lf.txt"
            crlf_path = directory / "crlf.txt"
            lf_path.write_bytes(b"dong 1\ndong 2\n")
            crlf_path.write_bytes(b"dong 1\r\ndong 2\r\n")
            self.assertEqual(sha256_text_file(lf_path), sha256_text_file(crlf_path))

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

    def test_news_coverage_requires_each_time_split(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["VCB", "VCB", "VCB"],
                "published_at": ["2023-01-02", "2024-01-02", "2025-01-02"],
            }
        )
        coverage = validate_news_coverage(frame, ["VCB"], minimum_per_split=1)
        self.assertEqual(coverage.loc["VCB"].to_dict(), {"train": 1, "validation": 1, "test": 1})
