"""Kiểm tra chất lượng và lập manifest cho dữ liệu OHLCV.

Không tự ý thay đổi giá từ nguồn dữ liệu. Các bất thường được ghi nhận rõ ràng để
người dùng quyết định có thay nguồn hoặc xử lý riêng hay không.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import TimeSplit


REQUIRED_OHLCV = ("time", "open", "high", "low", "close", "volume")


def validate_news_coverage(
    frame: pd.DataFrame,
    tickers: Iterable[str],
    minimum_per_split: int = 20,
) -> pd.DataFrame:
    """Bắt buộc mỗi mã có đủ tin ở train, validation và test."""
    split = TimeSplit()
    news = frame.copy()
    news["published_at"] = pd.to_datetime(news["published_at"], errors="raise", utc=True)
    local_date = (
        news["published_at"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None).dt.normalize()
    )
    news["split"] = pd.cut(
        local_date,
        bins=[
            pd.Timestamp.min,
            pd.Timestamp(split.train_end),
            pd.Timestamp(split.validation_end),
            pd.Timestamp.max,
        ],
        labels=["train", "validation", "test"],
    )
    ticker_list = list(tickers)
    coverage = (
        news.groupby(["ticker", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=ticker_list, columns=["train", "validation", "test"], fill_value=0)
        .astype(int)
    )
    insufficient = coverage.lt(minimum_per_split)
    if insufficient.any().any():
        details = [
            f"{coverage.index[row]}/{coverage.columns[column]}={coverage.iloc[row, column]}"
            for row, column in zip(*insufficient.to_numpy().nonzero())
        ]
        raise ValueError(
            f"Dữ liệu tin chưa đạt tối thiểu {minimum_per_split} bài cho mỗi mã/mỗi tập: "
            + ", ".join(details)
        )
    return coverage


def validate_ohlcv(frame: pd.DataFrame, symbol: str) -> list[str]:
    """Kiểm tra điều kiện cần của OHLCV và trả về cảnh báo không chặn xử lý.

    Giá đóng cửa VN-Index nằm ngoài [low, high] ở vài dòng nguồn KBS. Vì pipeline
    chỉ dùng close của VN-Index để tính return, các dòng này được ghi nhận là cảnh
    báo thay vì tự sửa hay loại bỏ. Với cổ phiếu, vi phạm tương tự là lỗi chặn.
    """
    missing = set(REQUIRED_OHLCV).difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: thiếu cột bắt buộc {sorted(missing)}.")
    if frame["time"].duplicated().any():
        raise ValueError(f"{symbol}: có ngày giao dịch trùng lặp.")

    numeric = frame.loc[:, REQUIRED_OHLCV[1:]]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol}: có giá trị thiếu hoặc không hữu hạn.")
    if (numeric.loc[:, ["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: giá OHLC phải dương.")
    if (numeric["volume"] < 0).any():
        raise ValueError(f"{symbol}: khối lượng giao dịch không được âm.")
    if (numeric["high"] < numeric["low"]).any():
        raise ValueError(f"{symbol}: phát hiện high < low.")

    out_of_range = (numeric["close"] < numeric["low"]) | (numeric["close"] > numeric["high"])
    count = int(out_of_range.sum())
    if count and symbol.upper() != "VNINDEX":
        raise ValueError(f"{symbol}: có {count} dòng close nằm ngoài [low, high].")
    if count:
        return [
            f"{symbol}: {count} dòng close nằm ngoài [low, high] từ nguồn dữ liệu; "
            "pipeline chỉ dùng close để tính vnindex_return và không tự sửa dữ liệu."
        ]
    return []


def sha256_file(path: Path) -> str:
    """Tính SHA-256 theo từng khối để có thể kiểm chứng đúng bản dữ liệu."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Checksum văn bản độc lập LF/CRLF để kiểm chứng trên nhiều hệ điều hành."""
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def data_inventory(data_dir: Path, symbols: Iterable[str], source: str) -> dict[str, object]:
    """Trả về thông tin đủ để tái lập đúng snapshot dữ liệu đang sử dụng."""
    records: list[dict[str, object]] = []
    warnings: list[str] = []
    for symbol in symbols:
        path = data_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy {path}.")
        frame = pd.read_csv(path)
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        for column in REQUIRED_OHLCV[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        warnings.extend(validate_ohlcv(frame, symbol))
        records.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "start": frame["time"].min().date().isoformat(),
                "end": frame["time"].max().date().isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "files": records,
        "quality_warnings": warnings,
    }


def write_data_manifest(data_dir: Path, symbols: Iterable[str], source: str) -> Path:
    """Ghi manifest JSON cạnh dữ liệu; file này không chứa API key hay dữ liệu thô."""
    output = data_dir / "DATA_MANIFEST.json"
    output.write_text(
        json.dumps(data_inventory(data_dir, symbols, source), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
