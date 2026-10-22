"""Tải dữ liệu OHLCV và tạo manifest cho lần thực nghiệm tái lập."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

import pandas as pd
from dotenv import load_dotenv

try:
    # Pipeline này dùng API Quote của dòng Vnstock 4.x.
    from vnstock import Quote
except ImportError:  # Cho phép kiểm tra/cache dữ liệu sẵn có mà không gọi mạng.
    Quote = None

from config import DATA_END, DATA_SOURCE, DATA_START
from console import configure_utf8_output
from data_quality import REQUIRED_OHLCV, validate_ohlcv, write_data_manifest

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
REQUIRED_COLUMNS = list(REQUIRED_OHLCV)
DEFAULT_SOURCE = DATA_SOURCE


def normalise_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate Vnstock's response and write dates in a stable CSV-friendly form."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: Vnstock response lacks columns: {sorted(missing)}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="raise").dt.strftime("%Y-%m-%d")
    frame = frame.dropna().drop_duplicates(subset="time").sort_values("time")
    if frame.empty:
        raise ValueError(f"{symbol}: Vnstock returned no usable daily records.")
    for issue in validate_ohlcv(frame, symbol):
        warnings.warn(issue, stacklevel=2)
    return frame.reset_index(drop=True)


def fetch_history(
    symbol: str,
    start: str,
    end: str,
    source: str = DEFAULT_SOURCE,
    force: bool = False,
) -> pd.DataFrame:
    """Return cached data when present; otherwise retrieve and persist a clean CSV."""
    destination = DATA_DIR / f"{symbol}.csv"
    if destination.exists() and not force:
        return pd.read_csv(destination)

    if Quote is None:
        raise RuntimeError(
            "Không tìm thấy Vnstock 4.x. Hãy cài requirements.txt trong môi trường riêng "
            "trước khi tải hoặc làm mới dữ liệu."
        )
    if not os.getenv("VNSTOCK_API_KEY"):
        raise RuntimeError(
            "Thiếu VNSTOCK_API_KEY. Tạo .env từ .env.example trước khi tải hoặc làm mới dữ liệu."
        )

    raw = Quote(source=source, symbol=symbol).history(
        start=start,
        end=end,
        interval="1D",
    )
    clean = normalise_ohlcv(raw, symbol)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(destination, index=False)
    return clean


def print_data_summary(symbol: str, frame: pd.DataFrame) -> None:
    print(f"\n[{symbol}] rows={len(frame)}; range={frame['time'].iloc[0]} to {frame['time'].iloc[-1]}")
    print("columns:", ", ".join(frame.columns))
    print(frame.head(5).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tải OHLCV và đóng băng mốc dữ liệu cho thí nghiệm.")
    parser.add_argument("--start", default=DATA_START)
    parser.add_argument("--end", default=DATA_END)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["VCB"],
        help="Các mã cổ phiếu cần lưu cache; VNINDEX luôn được bổ sung.",
    )
    parser.add_argument("--force", action="store_true", help="Tải lại và ghi đè các CSV đã có.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    args = parse_args()
    symbols = [*dict.fromkeys([*args.symbols, "VNINDEX"])]
    for symbol in symbols:
        frame = fetch_history(
            symbol=symbol,
            start=args.start,
            end=args.end,
            source=args.source,
            force=args.force,
        )
        print_data_summary(symbol, frame)
    manifest = write_data_manifest(DATA_DIR, symbols, args.source)
    print(f"Manifest dữ liệu: {manifest}")


if __name__ == "__main__":
    configure_utf8_output()
    main()
