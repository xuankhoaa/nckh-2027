"""Download and cache daily OHLCV data needed for the first VCB experiment."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from vnstock import Quote


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
DEFAULT_SOURCE = "KBS"


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
    parser = argparse.ArgumentParser(description="Fetch VCB and VN-Index daily OHLCV data.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["VCB"],
        help="Stock tickers to cache; VNINDEX is always included.",
    )
    parser.add_argument("--force", action="store_true", help="Refresh existing CSV files.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("VNSTOCK_API_KEY"):
        raise RuntimeError(
            "VNSTOCK_API_KEY is missing. Add it to .env; see .env.example for the format."
        )

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


if __name__ == "__main__":
    main()
