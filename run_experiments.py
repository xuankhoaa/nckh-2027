"""Điều phối ma trận mô hình × nguồn dữ liệu × hai bài toán."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    ABLATION_FEATURE_SETS,
    DEFAULT_NEWS_CUTOFF_HOUR,
    DEFAULT_VOLATILITY_HORIZON,
    DEFAULT_WINDOW,
    FEATURE_SETS,
    MAIN_FEATURE_SETS,
    RESEARCH_SEEDS,
    TARGETS,
    TICKERS,
)
from console import configure_utf8_output
from collect_news import validate_news_coverage
from evaluate import plot_group_metric_comparison, reset_generated_results
from provenance import write_experiment_manifest
from preprocess import aggregate_news_features
from train import Experiment, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy ma trận thực nghiệm NCKH đầy đủ.")
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--feature-sets", nargs="+", choices=tuple(FEATURE_SETS), default=list(MAIN_FEATURE_SETS))
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(RESEARCH_SEEDS))
    parser.add_argument("--windows", nargs="+", type=int, default=[DEFAULT_WINDOW])
    parser.add_argument("--tail-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--news-cutoff-hours", nargs="+", type=int, default=[DEFAULT_NEWS_CUTOFF_HOUR])
    parser.add_argument("--volatility-horizon", type=int, default=DEFAULT_VOLATILITY_HORIZON)
    parser.add_argument("--news-path", default="data/news.csv")
    parser.add_argument("--minimum-news-per-split", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--keep-results", action="store_true", help="Giữ output cùng schema để chạy tiếp.")
    return parser.parse_args()


def _neural_runs(args: argparse.Namespace, ticker: str, target: str, window: int, feature_sets: tuple[str, ...]):
    for feature_set in feature_sets:
        cutoff_hours = args.news_cutoff_hours if feature_set != "market_only" else [DEFAULT_NEWS_CUTOFF_HOUR]
        for cutoff_hour in cutoff_hours:
            for tail_weight in args.tail_weights:
                for seed in args.seeds:
                    for model in ("lstm", "mamba"):
                        yield Experiment(
                            model=model,
                            ticker=ticker,
                            target=target,
                            feature_set=feature_set,
                            seed=seed,
                            window=window,
                            volatility_horizon=args.volatility_horizon,
                            news_path=args.news_path,
                            news_cutoff_hour=cutoff_hour,
                            tail_weight=tail_weight,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                        )


def _validate_news_before_reset(args: argparse.Namespace) -> None:
    """Kiểm tra toàn bộ mã/cutoff trước khi xóa output của lần chạy trước."""
    news_path = Path(args.news_path)
    if not news_path.is_file():
        raise FileNotFoundError(
            f"Thiếu {args.news_path}. Hãy chuẩn bị dữ liệu theo NEWS_DATA_SCHEMA.md, "
            "hoặc chạy mốc kỹ thuật bằng --feature-sets market_only."
        )
    news = pd.read_csv(news_path)
    validate_news_coverage(news, list(args.tickers), args.minimum_news_per_split)
    for ticker in args.tickers:
        market_path = Path("data") / f"{ticker}.csv"
        trading_dates = pd.read_csv(market_path, usecols=["time"])["time"]
        for cutoff_hour in args.news_cutoff_hours:
            aggregate_news_features(news_path, ticker, trading_dates, cutoff_hour=cutoff_hour)


def main() -> None:
    args = parse_args()
    feature_sets = tuple(dict.fromkeys([
        *args.feature_sets,
        *(ABLATION_FEATURE_SETS if args.include_ablations else ()),
    ]))
    if any(name != "market_only" for name in feature_sets):
        _validate_news_before_reset(args)
    if not args.keep_results:
        reset_generated_results()
    started_at = datetime.now(timezone.utc).isoformat()
    runs: list[Experiment] = []
    for target in args.targets:
        for ticker in args.tickers:
            for window in args.windows:
                baseline_names = ("naive_zero", "train_mean") if target == "return" else ("train_mean",)
                runs.extend(
                    Experiment(
                        model=model,
                        ticker=ticker,
                        target=target,
                        feature_set="market_only",
                        seed=0,
                        window=window,
                        volatility_horizon=args.volatility_horizon,
                    )
                    for model in baseline_names
                )
                runs.extend(_neural_runs(args, ticker, target, window, feature_sets))

    for completed, experiment in enumerate(runs, start=1):
        print(
            f"[{completed}/{len(runs)}] {experiment.target} | {experiment.ticker} | "
            f"{experiment.model} | {experiment.feature_set} | seed={experiment.seed}"
        )
        metrics = run_experiment(experiment, show_alignment=(completed == 1))
        visible = {name: value for name, value in metrics.items() if value == value}
        print(", ".join(f"{name}={value:.6f}" for name, value in visible.items()))

    for target in args.targets:
        horizon = 1 if target == "return" else args.volatility_horizon
        for window in args.windows:
            plot_group_metric_comparison(window, target, horizon)
    write_experiment_manifest(
        windows=args.windows,
        seeds=args.seeds,
        tickers=args.tickers,
        targets=args.targets,
        feature_sets=feature_sets,
        volatility_horizon=args.volatility_horizon,
        news_path=args.news_path,
        news_cutoff_hours=args.news_cutoff_hours,
        tail_weights=args.tail_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        started_at=started_at,
    )


if __name__ == "__main__":
    configure_utf8_output()
    main()
