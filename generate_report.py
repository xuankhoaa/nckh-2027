"""Sinh bảng và báo cáo tiếng Việt cho cả return và volatility."""

from __future__ import annotations

import argparse
import json
from statistics import NormalDist

import numpy as np
import pandas as pd

from console import configure_utf8_output
from evaluate import RESULTS_DIR, per_ticker_summary, save_group_summary


MODEL_NAMES = {
    "naive_zero": "Baseline return bằng 0",
    "train_mean": "Baseline trung bình train",
    "ridge": "Ridge (alpha chọn bằng validation)",
    "lstm": "LSTM",
    "mamba": "Selective SSM (Mamba-style)",
}
FEATURE_NAMES = {
    "not_applicable": "Không áp dụng",
    "market_only": "Market-only",
    "market_news": "Market + News",
    "market_news_no_sentiment": "Market + số lượng tin",
    "market_news_no_count": "Market + điểm cảm xúc",
    "market_stationary": "Market stationary",
    "market_stationary_news": "Market stationary + News",
}
TARGET_NAMES = {
    "return": "Tỷ suất sinh lợi tương lai",
    "volatility": "Độ biến động thực hiện tương lai",
}


def target_display_name(target: str, horizon: int) -> str:
    if target == "return":
        return (
            "Tỷ suất sinh lợi phiên kế tiếp"
            if horizon == 1
            else f"Tỷ suất sinh lợi tích lũy {horizon} phiên"
        )
    return f"Độ biến động thực hiện {horizon} phiên tương lai"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Không có dữ liệu._"
    headers = [str(value) for value in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def hac_squared_loss_test(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """Kiểm định loss differential với Newey-West, có xét horizon chồng lấn."""
    differential = (baseline - actual) ** 2 - (predicted - actual) ** 2
    count = len(differential)
    centered = differential - differential.mean()
    max_lag = min(max(horizon - 1, int(count ** (1 / 3))), count - 1)
    long_run_variance = float(np.mean(centered * centered))
    for lag in range(1, max_lag + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run_variance += 2 * (1 - lag / (max_lag + 1)) * covariance
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / count))
    statistic = float(differential.mean() / standard_error) if standard_error > 0 else float("nan")
    p_value = float(2 * (1 - NormalDist().cdf(abs(statistic)))) if np.isfinite(statistic) else float("nan")
    return {
        "mean_squared_error_improvement": float(differential.mean()),
        "hac_statistic": statistic,
        "p_value_two_sided": p_value,
    }


def statistical_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    """So sánh từng mô hình học được với Train Mean trên đúng các ngày test."""
    baselines = predictions.loc[
        predictions["model"].eq("train_mean"),
        ["ticker", "target", "horizon", "window", "date", "actual_value", "predicted_value"],
    ]
    records: list[dict[str, object]] = []
    learned = predictions.loc[predictions["model"].isin(["ridge", "lstm", "mamba"])]
    group_columns = [
        "model", "ticker", "target", "horizon", "window", "feature_set",
        "news_cutoff_hour", "tail_weight", "loss_name", "weight_decay",
        "validation_shrinkage", "seed",
    ]
    for identity, frame in learned.groupby(group_columns):
        (
            model, ticker, target, horizon, window, feature_set, cutoff,
            tail_weight, loss_name, weight_decay, shrinkage, seed,
        ) = identity
        matching_baseline = baselines.loc[
            baselines["ticker"].eq(ticker)
            & baselines["target"].eq(target)
            & baselines["horizon"].eq(horizon)
            & baselines["window"].eq(window)
        ]
        paired = frame.merge(
            matching_baseline,
            on=["ticker", "target", "horizon", "window", "date", "actual_value"],
            suffixes=("_model", "_baseline"),
            validate="one_to_one",
        )
        result = hac_squared_loss_test(
            paired["actual_value"].to_numpy(),
            paired["predicted_value_model"].to_numpy(),
            paired["predicted_value_baseline"].to_numpy(),
            int(horizon),
        )
        records.append({
            "model": model,
            "ticker": ticker,
            "target": target,
            "horizon": horizon,
            "window": window,
            "feature_set": feature_set,
            "news_cutoff_hour": cutoff,
            "tail_weight": tail_weight,
            "loss_name": loss_name,
            "weight_decay": weight_decay,
            "validation_shrinkage": shrinkage,
            "seed": seed,
            "n_test": len(paired),
            **result,
        })
    return pd.DataFrame(records)


def news_value_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    """So sánh ghép cặp Market+News với Market-only cùng mô hình và seed."""
    records: list[dict[str, object]] = []
    group_columns = [
        "model", "ticker", "target", "horizon", "window",
        "news_cutoff_hour", "tail_weight", "loss_name", "weight_decay",
        "validation_shrinkage", "seed",
    ]
    pairs = {
        "market_news": "market_only",
        "market_stationary_news": "market_stationary",
    }
    eligible = predictions.loc[predictions["model"].isin(["ridge", "lstm", "mamba"])]
    for news_feature_set, reference_feature_set in pairs.items():
        news = eligible.loc[eligible["feature_set"].eq(news_feature_set)]
        market = eligible.loc[eligible["feature_set"].eq(reference_feature_set)]
        for identity, frame in news.groupby(group_columns):
            (
                model, ticker, target, horizon, window, cutoff, tail_weight,
                loss_name, weight_decay, shrinkage, seed,
            ) = identity
            reference = market.loc[
                market["model"].eq(model)
                & market["ticker"].eq(ticker)
                & market["target"].eq(target)
                & market["horizon"].eq(horizon)
                & market["window"].eq(window)
                & market["tail_weight"].eq(tail_weight)
                & market["loss_name"].eq(loss_name)
                & market["weight_decay"].eq(weight_decay)
                & market["validation_shrinkage"].eq(shrinkage)
                & market["seed"].eq(seed)
            ]
            paired = frame.merge(
                reference,
                on=[
                    "model", "ticker", "target", "horizon", "window",
                    "tail_weight", "loss_name", "weight_decay",
                    "validation_shrinkage", "seed", "date", "actual_value",
                ],
                suffixes=("_news", "_market"),
                validate="one_to_one",
            )
            result = hac_squared_loss_test(
                paired["actual_value"].to_numpy(),
                paired["predicted_value_news"].to_numpy(),
                paired["predicted_value_market"].to_numpy(),
                int(horizon),
            )
            records.append({
                "model": model,
                "ticker": ticker,
                "target": target,
                "horizon": horizon,
                "window": window,
                "feature_set": news_feature_set,
                "reference_feature_set": reference_feature_set,
                "news_cutoff_hour": cutoff,
                "tail_weight": tail_weight,
                "loss_name": loss_name,
                "weight_decay": weight_decay,
                "validation_shrinkage": shrinkage,
                "seed": seed,
                "n_test": len(paired),
                **result,
            })
    return pd.DataFrame(records)


def _format_summary(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    columns = ["model", "feature_set", "tickers", "seed_count", "rmse_mean", "mae_mean", "r2_oos_mean"]
    columns.append("directional_accuracy_mean" if target == "return" else "qlike_mean")
    result = frame.loc[:, columns].copy()
    result["model"] = result["model"].map(MODEL_NAMES)
    result["feature_set"] = result["feature_set"].map(FEATURE_NAMES)
    for column in result.columns[4:]:
        result[column] = result[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    return result.rename(columns={
        "model": "Mô hình",
        "feature_set": "Tập đặc trưng",
        "tickers": "Số mã",
        "seed_count": "Số seed",
        "rmse_mean": "RMSE",
        "mae_mean": "MAE",
        "r2_oos_mean": "R² OOS",
        "directional_accuracy_mean": "DA",
        "qlike_mean": "QLIKE",
    })


def _vietnamese_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "model" in result:
        result["model"] = result["model"].map(MODEL_NAMES).fillna(result["model"])
    if "feature_set" in result:
        result["feature_set"] = result["feature_set"].map(FEATURE_NAMES).fillna(result["feature_set"])
    if "reference_feature_set" in result:
        result["reference_feature_set"] = (
            result["reference_feature_set"].map(FEATURE_NAMES).fillna(result["reference_feature_set"])
        )
    if "target" in result:
        if "horizon" in result:
            result["target"] = [
                target_display_name(str(target), int(horizon))
                for target, horizon in zip(result["target"], result["horizon"])
            ]
        else:
            result["target"] = result["target"].map(TARGET_NAMES).fillna(result["target"])
    return result.rename(columns={
        "model": "Mô hình", "ticker": "Mã", "target": "Bài toán",
        "horizon": "Horizon", "window": "Cửa sổ", "feature_set": "Tập đặc trưng",
        "news_cutoff_hour": "Giờ cutoff tin", "tail_weight": "Trọng số tail",
        "loss_name": "Hàm mất mát", "weight_decay": "Weight decay",
        "validation_shrinkage": "Co dự báo theo validation",
        "shrinkage_alpha": "Hệ số co", "ridge_alpha": "Ridge alpha",
        "reference_feature_set": "Tập đặc trưng đối chứng",
        "seed": "Seed", "parameter_count": "Số tham số", "rmse": "RMSE",
        "mae": "MAE", "r2_test": "R² test", "r2_oos": "R² OOS so với Train Mean", "directional_accuracy": "DA",
        "qlike": "QLIKE", "date": "Ngày cuối target", "actual_value": "Thực tế",
        "predicted_value": "Dự báo", "tickers": "Số mã", "seed_count": "Số seed",
        "n_test": "Số quan sát test", "mean_squared_error_improvement": "Cải thiện MSE so với Train Mean",
        "hac_statistic": "Thống kê HAC", "p_value_two_sided": "p hai phía",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh báo cáo tổng hợp từ kết quả đã chạy.")
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()
    metrics_path = RESULTS_DIR / "metrics.csv"
    predictions_path = RESULTS_DIR / "predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Chưa có kết quả schema mới; hãy chạy run_experiments.py trước.")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    required = {
        "model", "ticker", "target", "horizon", "window", "feature_set", "seed",
        "news_cutoff_hour", "tail_weight", "rmse", "mae", "r2_test", "r2_oos", "directional_accuracy", "qlike",
    }
    if not required.issubset(metrics.columns):
        raise ValueError("Kết quả thuộc schema cũ. Hãy chạy lại run_experiments.py không dùng --keep-results.")
    selected = metrics.loc[metrics["window"].eq(args.window)].copy()
    if selected.empty:
        raise ValueError(f"Không có kết quả cho window={args.window}.")

    summaries: list[pd.DataFrame] = []
    ticker_summaries: list[pd.DataFrame] = []
    report_sections: list[str] = []
    for target, horizon in selected[["target", "horizon"]].drop_duplicates().itertuples(index=False, name=None):
        summary = save_group_summary(args.window, target, int(horizon))
        summaries.append(summary)
        ticker_summaries.append(per_ticker_summary(args.window, target, int(horizon)))
        display = _format_summary(summary, target)
        best = summary.loc[summary["rmse_mean"].idxmin()]
        baseline = summary.loc[summary["model"].eq("train_mean")].iloc[0]
        neural = summary.loc[summary["model"].isin(["lstm", "mamba"])]
        best_neural = neural.loc[neural["rmse_mean"].idxmin()]
        verdict = "vượt" if best_neural["rmse_mean"] < baseline["rmse_mean"] else "chưa vượt"
        report_sections.append(
            f"## {target_display_name(target, int(horizon))}\n\n"
            f"{markdown_table(display)}\n\n"
            f"Cấu hình RMSE thấp nhất: **{MODEL_NAMES[best['model']]} / "
            f"{FEATURE_NAMES[best['feature_set']]}** ({best['rmse_mean']:.6f}). "
            f"Mạng tốt nhất {verdict} Train Mean ({baseline['rmse_mean']:.6f})."
        )

    group_summary = pd.concat(summaries, ignore_index=True)
    ticker_summary = pd.concat(ticker_summaries, ignore_index=True)
    group_summary.to_csv(RESULTS_DIR / "group_summary.csv", index=False)
    ticker_summary.to_csv(RESULTS_DIR / "per_ticker_summary.csv", index=False)
    tests = statistical_tests(predictions.loc[predictions["window"].eq(args.window)].copy())
    news_tests = news_value_tests(predictions.loc[predictions["window"].eq(args.window)].copy())
    _vietnamese_values(selected).to_csv(
        RESULTS_DIR / "BANG_KET_QUA_CHI_TIET_VI.csv", index=False, encoding="utf-8-sig"
    )
    _vietnamese_values(group_summary).to_csv(
        RESULTS_DIR / "BANG_TONG_HOP_NHOM_VI.csv", index=False, encoding="utf-8-sig"
    )
    _vietnamese_values(tests).to_csv(
        RESULTS_DIR / "KIEM_DINH_HAC_VI.csv", index=False, encoding="utf-8-sig"
    )
    _vietnamese_values(news_tests).to_csv(
        RESULTS_DIR / "KIEM_DINH_GIA_TRI_TIN_TUC_VI.csv", index=False, encoding="utf-8-sig"
    )
    _vietnamese_values(predictions.loc[predictions["window"].eq(args.window)]).to_csv(
        RESULTS_DIR / "DU_BAO_THEO_NGAY_VI.csv", index=False, encoding="utf-8-sig"
    )
    tests.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)
    news_tests.to_csv(RESULTS_DIR / "news_value_tests.csv", index=False)

    news_counts = (
        news_tests.assign(
            improved=lambda frame: frame["mean_squared_error_improvement"].gt(0),
            significant=lambda frame: frame["p_value_two_sided"].lt(0.05),
        )
        .groupby(["target", "horizon", "model"], as_index=False)
        .agg(
            paired_runs=("seed", "size"),
            improved_runs=("improved", "sum"),
            significant_improved_runs=(
                "mean_squared_error_improvement",
                lambda values: int(((values > 0) & (news_tests.loc[values.index, "p_value_two_sided"] < 0.05)).sum()),
            ),
            mean_mse_improvement=("mean_squared_error_improvement", "mean"),
        )
    )
    news_counts_display = _vietnamese_values(news_counts).rename(columns={
        "paired_runs": "Số cặp mã-seed",
        "improved_runs": "Số cặp news tốt hơn",
        "significant_improved_runs": "Số cặp tốt hơn với p<0,05",
        "mean_mse_improvement": "Cải thiện MSE trung bình",
    })
    news_counts_display["Cải thiện MSE trung bình"] = news_counts_display[
        "Cải thiện MSE trung bình"
    ].map(lambda value: f"{value:.8f}")

    manifest_path = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    manifest_note = ""
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_note = f"\n\nManifest lần chạy: Git `{manifest.get('git_commit')}`, Python {manifest.get('python')}."
    report = (
        "# Báo cáo kết quả hai bài toán\n\n"
        "Mỗi mô hình được so sánh trên cùng tập đặc trưng. Train Mean là ngưỡng tối thiểu; "
        "Ridge là baseline tuyến tính chọn mức phạt bằng validation; Market-only và "
        "Market+News là thí nghiệm loại bỏ đặc trưng, không phải tên mô hình.\n\n"
        + "\n\n".join(report_sections)
        + "\n\n## Cách đọc kết quả\n\n"
        "RMSE/MAE càng thấp càng tốt. DA chỉ dùng cho return; QLIKE chỉ dùng cho volatility. "
        "Các kiểm định HAC trong `KIEM_DINH_HAC_VI.csv` so sánh từng lần chạy Ridge/mạng với Train Mean; "
        "giá trị cải thiện MSE dương mới có lợi cho mô hình. Hệ số co dự báo được ước lượng chỉ "
        "trên validation, không dùng test. Kết luận phải đồng thời xem từng mã, "
        "trung bình nhóm và độ lệch chuẩn giữa seed.\n\n"
        "## Giá trị tăng thêm của tin tức\n\n"
        + markdown_table(news_counts_display)
        + "\n\nBảng này ghép đúng cùng mô hình, mã và seed. Cải thiện MSE dương nghĩa là "
        "Market+News tốt hơn Market-only; p-value HAC là bằng chứng bổ sung, chưa điều chỉnh nhiều lần."
        + manifest_note
        + "\n"
    )
    (RESULTS_DIR / "BAO_CAO_KET_QUA.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    configure_utf8_output()
    main()
