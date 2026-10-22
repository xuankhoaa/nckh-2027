"""Chạy bộ thực nghiệm đầy đủ: 2 baseline và 3 mạng qua nhiều seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from config import DEFAULT_WINDOW, RESEARCH_SEEDS, TICKERS
from evaluate import plot_group_metric_comparison, reset_generated_results
from provenance import write_experiment_manifest
from train import Experiment, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm tái lập cho nhóm cổ phiếu ngân hàng.")
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(RESEARCH_SEEDS))
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--keep-results", action="store_true", help="Không xóa các đầu ra cũ trước khi chạy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.keep_results:
        reset_generated_results()
    started_at = datetime.now(timezone.utc).isoformat()
    total = len(args.tickers) * (2 + 3 * len(args.seeds))
    completed = 0

    for ticker in args.tickers:
        baseline_runs = (
            Experiment(model="naive_zero", ticker=ticker, seed=0, window=args.window),
            Experiment(model="train_mean", ticker=ticker, seed=0, window=args.window),
        )
        neural_runs = tuple(
            Experiment(model=model, ticker=ticker, seed=seed, window=args.window, epochs=args.epochs, batch_size=args.batch_size)
            for seed in args.seeds
            for model in ("lstm", "mamba_baseline", "improved_mamba")
        )
        for experiment in (*baseline_runs, *neural_runs):
            completed += 1
            print(f"[{completed}/{total}] {ticker} | {experiment.model} | seed={experiment.seed}")
            metrics = run_experiment(experiment, show_alignment=(completed == 1))
            print(", ".join(f"{name}={value:.6f}" for name, value in metrics.items()))

    plot_group_metric_comparison(args.window)
    write_experiment_manifest(
        window=args.window,
        seeds=args.seeds,
        tickers=args.tickers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        started_at=started_at,
    )


if __name__ == "__main__":
    main()
