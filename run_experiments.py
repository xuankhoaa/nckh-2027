"""Điều phối ma trận mô hình × nguồn dữ liệu × hai bài toán."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    ABLATION_FEATURE_SETS,
    DEFAULT_NEWS_CUTOFF_HOUR,
    DEFAULT_RETURN_HORIZON,
    DEFAULT_VOLATILITY_HORIZON,
    DEFAULT_WINDOW,
    FEATURE_SETS,
    MAIN_FEATURE_SETS,
    NEWS_FEATURES,
    RESEARCH_SEEDS,
    TARGETS,
    TICKERS,
)
from console import configure_utf8_output
from data_quality import sha256_file, validate_news_coverage
from evaluate import plot_group_metric_comparison, reset_generated_results
from provenance import write_experiment_manifest
from preprocess import aggregate_news_features
from train import Experiment, run_experiment


def _uses_news(feature_set: str) -> bool:
    return any(feature in FEATURE_SETS[feature_set] for feature in NEWS_FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy ma trận thực nghiệm NCKH đầy đủ.")
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--feature-sets", nargs="+", choices=tuple(FEATURE_SETS), default=list(MAIN_FEATURE_SETS))
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(RESEARCH_SEEDS))
    parser.add_argument("--windows", nargs="+", type=int, default=[DEFAULT_WINDOW])
    parser.add_argument("--return-horizons", nargs="+", type=int, default=[DEFAULT_RETURN_HORIZON, 5])
    parser.add_argument("--tail-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--news-cutoff-hours", nargs="+", type=int, default=[DEFAULT_NEWS_CUTOFF_HOUR])
    parser.add_argument("--volatility-horizon", type=int, default=DEFAULT_VOLATILITY_HORIZON)
    parser.add_argument("--return-loss", choices=("mse", "huber"), default="huber")
    parser.add_argument("--volatility-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-validation-shrinkage", dest="validation_shrinkage", action="store_false")
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0],
    )
    parser.add_argument("--no-ridge", dest="include_ridge", action="store_false")
    parser.add_argument("--news-path", default="data/news_phobert.csv")
    parser.add_argument("--minimum-news-per-split", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--keep-results", action="store_true", help="Giữ output cùng schema để chạy tiếp.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Giữ output và bỏ qua các định danh thí nghiệm đã hoàn tất.",
    )
    return parser.parse_args()


def _experiment_key(experiment: Experiment) -> tuple[object, ...]:
    feature_set = (
        "not_applicable"
        if experiment.model in ("naive_zero", "train_mean")
        else experiment.feature_set
    )
    neural = experiment.model in ("lstm", "mamba")
    return (
        experiment.model,
        experiment.ticker,
        experiment.target,
        experiment.horizon,
        experiment.window,
        feature_set,
        experiment.news_cutoff_hour,
        float(experiment.tail_weight),
        experiment.loss_name if neural else "not_applicable",
        float(experiment.weight_decay) if neural else 0.0,
        bool(experiment.validation_shrinkage) if neural else False,
        experiment.seed,
    )


def _completed_keys() -> set[tuple[object, ...]]:
    path = Path("results/metrics.csv")
    prediction_path = Path("results/predictions.csv")
    if not path.is_file() or not prediction_path.is_file():
        return set()
    frame = pd.read_csv(path)
    prediction_rows = pd.read_csv(prediction_path)
    required = [
        "model", "ticker", "target", "horizon", "window", "feature_set",
        "news_cutoff_hour", "tail_weight", "loss_name", "weight_decay",
        "validation_shrinkage", "seed",
    ]
    if not set(required).issubset(frame.columns) or not set(required).issubset(prediction_rows.columns):
        raise ValueError("Không thể resume từ results/ khác schema.")
    bool_values = frame["validation_shrinkage"]
    if not pd.api.types.is_bool_dtype(bool_values):
        bool_values = bool_values.astype(str).str.lower().eq("true")
    metric_keys = {
        (
            row.model, row.ticker, row.target, int(row.horizon), int(row.window),
            row.feature_set, int(row.news_cutoff_hour), float(row.tail_weight),
            row.loss_name, float(row.weight_decay), bool(shrinkage), int(row.seed),
        )
        for row, shrinkage in zip(frame[required].itertuples(index=False), bool_values)
    }
    prediction_keys = {
        (
            row.model, row.ticker, row.target, int(row.horizon), int(row.window),
            row.feature_set, int(row.news_cutoff_hour), float(row.tail_weight),
            row.loss_name, float(row.weight_decay), str(row.validation_shrinkage).lower() == "true", int(row.seed),
        )
        for row in prediction_rows[required].itertuples(index=False)
    }
    return metric_keys & prediction_keys


def _assert_resume_compatible(args: argparse.Namespace, feature_sets: tuple[str, ...]) -> None:
    """Không cho trộn kết quả khi cấu hình ma trận đã thay đổi."""
    manifest_path = Path("results/EXPERIMENT_MANIFEST.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("Không thể --resume khi thiếu EXPERIMENT_MANIFEST.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = manifest["experiment"]
    requested = {
        "tickers": list(args.tickers),
        "seeds": list(args.seeds),
        "windows": list(args.windows),
        "targets": list(args.targets),
        "feature_sets": list(feature_sets),
        "return_horizons": list(args.return_horizons),
        "volatility_horizon": args.volatility_horizon,
        "news_cutoff_hours": list(args.news_cutoff_hours),
        "tail_weights": list(args.tail_weights),
        "return_loss": args.return_loss,
        "volatility_loss": args.volatility_loss,
        "weight_decay": args.weight_decay,
        "validation_shrinkage": args.validation_shrinkage,
        "ridge_alphas": list(args.ridge_alphas),
        "include_ridge": args.include_ridge,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    changed = [name for name, value in requested.items() if previous.get(name) != value]
    if changed:
        raise ValueError(
            "Không thể --resume vì cấu hình khác lần chạy trước ở: "
            + ", ".join(changed)
            + ". Hãy chạy ma trận mới không dùng --resume."
        )
    if any(_uses_news(name) for name in feature_sets):
        news_info = manifest.get("news_data", {})
        news_path = Path(args.news_path)
        if (
            news_info.get("path", "").replace("\\", "/") != news_path.as_posix()
            or not news_path.is_file()
            or news_info.get("sha256") != sha256_file(news_path)
        ):
            raise ValueError("Không thể --resume vì file tin tức khác lần chạy trước.")
    for record in manifest.get("data_manifest", {}).get("files", []):
        market_path = Path("data") / f"{record['symbol']}.csv"
        if not market_path.is_file() or sha256_file(market_path) != record["sha256"]:
            raise ValueError(f"Không thể chạy tiếp vì dữ liệu {record['symbol']} đã thay đổi.")


def _neural_runs(
    args: argparse.Namespace,
    ticker: str,
    target: str,
    horizon: int,
    window: int,
    feature_sets: tuple[str, ...],
):
    for feature_set in feature_sets:
        cutoff_hours = args.news_cutoff_hours if _uses_news(feature_set) else [DEFAULT_NEWS_CUTOFF_HOUR]
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
                            return_horizon=horizon if target == "return" else DEFAULT_RETURN_HORIZON,
                            volatility_horizon=args.volatility_horizon,
                            news_path=args.news_path,
                            news_cutoff_hour=cutoff_hour,
                            tail_weight=tail_weight,
                            loss_name=(args.return_loss if target == "return" else args.volatility_loss),
                            weight_decay=args.weight_decay,
                            validation_shrinkage=args.validation_shrinkage,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                        )


def _validate_news_before_reset(args: argparse.Namespace) -> None:
    """Kiểm tra toàn bộ mã/cutoff trước khi xóa output của lần chạy trước."""
    news_path = Path(args.news_path)
    if not news_path.is_file():
        raise FileNotFoundError(
            f"Thiếu {args.news_path}. Hãy chuẩn bị dữ liệu theo docs/NEWS_DATA_SCHEMA.md, "
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
    if any(_uses_news(name) for name in feature_sets):
        _validate_news_before_reset(args)
    if args.resume or args.keep_results:
        _assert_resume_compatible(args, feature_sets)
    if not args.keep_results and not args.resume:
        reset_generated_results()
    started_at = datetime.now(timezone.utc).isoformat()
    if not args.resume and not args.keep_results:
        write_experiment_manifest(
            windows=args.windows, seeds=args.seeds, tickers=args.tickers,
            targets=args.targets, feature_sets=feature_sets,
            return_horizons=args.return_horizons,
            volatility_horizon=args.volatility_horizon, news_path=args.news_path,
            news_cutoff_hours=args.news_cutoff_hours,
            tail_weights=args.tail_weights, return_loss=args.return_loss,
            volatility_loss=args.volatility_loss, weight_decay=args.weight_decay,
            validation_shrinkage=args.validation_shrinkage,
            ridge_alphas=args.ridge_alphas, include_ridge=args.include_ridge,
            epochs=args.epochs, batch_size=args.batch_size, started_at=started_at,
        )
    runs: list[Experiment] = []
    for target in args.targets:
        horizons = args.return_horizons if target == "return" else [args.volatility_horizon]
        for horizon in horizons:
            for ticker in args.tickers:
                for window in args.windows:
                    constant_names = ("naive_zero", "train_mean") if target == "return" else ("train_mean",)
                    runs.extend(
                        Experiment(
                            model=model,
                            ticker=ticker,
                            target=target,
                            feature_set=feature_sets[0],
                            seed=0,
                            window=window,
                            return_horizon=horizon if target == "return" else DEFAULT_RETURN_HORIZON,
                            volatility_horizon=args.volatility_horizon,
                        )
                        for model in constant_names
                    )
                    if args.include_ridge:
                        runs.extend(
                            Experiment(
                                model="ridge",
                                ticker=ticker,
                                target=target,
                                feature_set=feature_set,
                                seed=0,
                                window=window,
                                return_horizon=horizon if target == "return" else DEFAULT_RETURN_HORIZON,
                                volatility_horizon=args.volatility_horizon,
                                news_path=args.news_path,
                                news_cutoff_hour=(
                                    args.news_cutoff_hours[0]
                                    if _uses_news(feature_set)
                                    else DEFAULT_NEWS_CUTOFF_HOUR
                                ),
                                ridge_alphas=tuple(args.ridge_alphas),
                            )
                            for feature_set in feature_sets
                        )
                    runs.extend(_neural_runs(args, ticker, target, horizon, window, feature_sets))

    completed_keys = _completed_keys() if args.resume else set()
    pending_runs = [experiment for experiment in runs if _experiment_key(experiment) not in completed_keys]
    if args.resume:
        print(f"Tiếp tục: bỏ qua {len(runs) - len(pending_runs)} lượt đã hoàn tất, còn {len(pending_runs)} lượt.")
    for completed, experiment in enumerate(pending_runs, start=1):
        print(
            f"[{completed}/{len(pending_runs)}] {experiment.target} | {experiment.ticker} | "
            f"{experiment.model} | {experiment.feature_set} | seed={experiment.seed}"
        )
        metrics = run_experiment(experiment, show_alignment=(completed == 1 and not completed_keys))
        visible = {name: value for name, value in metrics.items() if value == value}
        print(", ".join(f"{name}={value:.6f}" for name, value in visible.items()))

    for target in args.targets:
        horizons = args.return_horizons if target == "return" else [args.volatility_horizon]
        for horizon in horizons:
            for window in args.windows:
                plot_group_metric_comparison(window, target, horizon)


if __name__ == "__main__":
    configure_utf8_output()
    main()
