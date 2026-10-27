"""So sánh kết quả dự báo dùng PhoBERT, từ điển và Market-only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from console import configure_utf8_output


PAIR_KEYS = [
    "model", "ticker", "target", "horizon", "window", "news_cutoff_hour",
    "tail_weight", "loss_name", "weight_decay", "validation_shrinkage", "seed",
]


def _summarize(
    frame: pd.DataFrame,
    *,
    comparison: str,
    first_column: str,
    second_column: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (target, horizon, model), group in frame.groupby(["target", "horizon", "model"]):
        delta = group[first_column] - group[second_column]
        rows.append(
            {
                "comparison": comparison,
                "target": target,
                "horizon": int(horizon),
                "model": model,
                "n_pairs": int(len(group)),
                "first_wins": int((delta < 0).sum()),
                "ties": int((delta.abs() < 1e-15).sum()),
                "delta_rmse_mean": float(delta.mean()),
                "first_rmse_mean": float(group[first_column].mean()),
                "second_rmse_mean": float(group[second_column].mean()),
            }
        )
    return rows


def compare(current_path: Path, lexicon_path: Path) -> pd.DataFrame:
    current = pd.read_csv(current_path)
    lexicon = pd.read_csv(lexicon_path)
    required = {*PAIR_KEYS, "feature_set", "rmse"}
    for path, frame in ((current_path, current), (lexicon_path, lexicon)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} thiếu cột {sorted(missing)}.")

    news = current.loc[current["feature_set"].eq("market_stationary_news")]
    market = current.loc[current["feature_set"].eq("market_stationary")]
    phobert_vs_market = news[PAIR_KEYS + ["rmse"]].rename(columns={"rmse": "phobert"}).merge(
        market[PAIR_KEYS + ["rmse"]].rename(columns={"rmse": "market"}),
        on=PAIR_KEYS,
        validate="one_to_one",
    )
    rows = _summarize(
        phobert_vs_market,
        comparison="phobert_news_vs_market_only",
        first_column="phobert",
        second_column="market",
    )

    old_news = lexicon.loc[lexicon["feature_set"].eq("market_stationary_news")]
    phobert_vs_lexicon = news[PAIR_KEYS + ["rmse"]].rename(columns={"rmse": "phobert"}).merge(
        old_news[PAIR_KEYS + ["rmse"]].rename(columns={"rmse": "lexicon"}),
        on=PAIR_KEYS,
        validate="one_to_one",
    )
    rows.extend(
        _summarize(
            phobert_vs_lexicon,
            comparison="phobert_news_vs_lexicon_news",
            first_column="phobert",
            second_column="lexicon",
        )
    )
    return pd.DataFrame(rows).sort_values(["comparison", "target", "horizon", "model"])


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh PhoBERT với từ điển và Market-only.")
    parser.add_argument("--current", default="results/metrics.csv")
    parser.add_argument("--lexicon", default="results/lexicon/metrics.csv")
    parser.add_argument("--output", default="results/sentiment_method_comparison.csv")
    args = parser.parse_args()
    summary = compare(Path(args.current), Path(args.lexicon))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"Đã lưu {output}.")


if __name__ == "__main__":
    configure_utf8_output()
    main()
