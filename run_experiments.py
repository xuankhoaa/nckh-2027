"""Run the three agreed first-pass models and create comparison charts."""

from __future__ import annotations

import argparse

from config import TICKERS
from evaluate import plot_group_metric_comparison, plot_metric_comparison
from train import Experiment, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM, Mamba baseline, and improved Mamba.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(TICKERS),
        help="Các mã cần chạy; mặc định là toàn bộ 8 ngân hàng trong config.py.",
    )
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ticker in args.tickers:
        experiments = (
            Experiment(
                model="lstm", ticker=ticker, window=args.window, epochs=args.epochs, batch_size=args.batch_size
            ),
            Experiment(
                model="mamba", ticker=ticker, window=args.window, epochs=args.epochs, batch_size=args.batch_size
            ),
            Experiment(
                model="mamba",
                ticker=ticker,
                include_vnindex=True,
                window=args.window,
                epochs=args.epochs,
                batch_size=args.batch_size,
            ),
        )
        for experiment in experiments:
            print(f"\n=== {experiment.name}: {experiment.ticker} ===")
            run_experiment(experiment)
        plot_metric_comparison(ticker, args.window)
    plot_group_metric_comparison(args.window)


if __name__ == "__main__":
    main()
