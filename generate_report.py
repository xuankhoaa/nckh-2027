"""Sinh báo cáo tiếng Việt từ kết quả nhiều seed và manifest tái lập."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from evaluate import RESULTS_DIR, per_ticker_summary, save_group_summary


MODEL_ORDER = {"naive_zero": 0, "train_mean": 1, "lstm": 2, "mamba_baseline": 3, "improved_mamba": 4}
MODEL_NAMES = {
    "naive_zero": "Baseline zero return",
    "train_mean": "Baseline trung bình train",
    "lstm": "LSTM",
    "mamba_baseline": "Selective SSM cơ sở",
    "improved_mamba": "Selective SSM có VN-Index",
}
FEATURE_NAMES = {
    "khong_ap_dung": "Không áp dụng",
    "ohlcv_derived": "OHLCV và đặc trưng dẫn xuất",
    "ohlcv_derived_plus_vnindex": "OHLCV, đặc trưng dẫn xuất và return VN-Index",
}


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(value) for value in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def format_metrics(frame: pd.DataFrame, numeric_columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in numeric_columns:
        if column in result:
            result[column] = result[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    return result


def hac_squared_loss_test(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    """Kiểm định chênh lệch squared loss với ước lượng HAC cho horizon một phiên."""
    differential = (baseline - actual) ** 2 - (predicted - actual) ** 2
    count = len(differential)
    centered = differential - differential.mean()
    max_lag = min(int(count ** (1 / 3)), count - 1)
    long_run_variance = float(np.mean(centered * centered))
    for lag in range(1, max_lag + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run_variance += 2 * (1 - lag / (max_lag + 1)) * covariance
    standard_error = float(np.sqrt(max(long_run_variance, 0.0) / count))
    statistic = float(differential.mean() / standard_error) if standard_error > 0 else float("nan")
    p_value = float(2 * (1 - NormalDist().cdf(abs(statistic)))) if np.isfinite(statistic) else float("nan")
    return {"mean_squared_error_improvement": float(differential.mean()), "hac_statistic": statistic, "p_value_two_sided": p_value}


def statistical_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    """So sánh từng seed mạng với baseline zero trên cùng ticker và ngày test."""
    baseline = predictions.loc[predictions["model"].eq("naive_zero"), ["ticker", "date", "actual_return", "predicted_return"]]
    records: list[dict[str, object]] = []
    for (model, ticker, seed), frame in predictions.loc[predictions["model"].isin(["lstm", "mamba_baseline", "improved_mamba"])].groupby(["model", "ticker", "seed"]):
        paired = frame.merge(baseline, on=["ticker", "date", "actual_return"], suffixes=("_model", "_baseline"), validate="one_to_one")
        result = hac_squared_loss_test(
            paired["actual_return"].to_numpy(),
            paired["predicted_return_model"].to_numpy(),
            paired["predicted_return_baseline"].to_numpy(),
        )
        records.append({"model": model, "ticker": ticker, "seed": seed, "n_test": len(paired), **result})
    return pd.DataFrame(records)


def vietnamese_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["model"] = result["model"].map(MODEL_NAMES)
    result["feature_set"] = result["feature_set"].map(FEATURE_NAMES)
    return result.rename(
        columns={
            "model": "Mô hình",
            "window": "Cửa sổ phiên",
            "feature_set": "Tập đặc trưng",
            "tickers": "Số mã",
            "seed_count": "Số seed",
            "parameter_count": "Số tham số",
            "rmse_mean": "RMSE trung bình",
            "mae_mean": "MAE trung bình",
            "r2_mean": "R² trung bình",
            "directional_accuracy_mean": "DA trung bình",
            "rmse_std_ticker": "ĐLC RMSE giữa mã",
            "mae_std_ticker": "ĐLC MAE giữa mã",
            "r2_std_ticker": "ĐLC R² giữa mã",
            "directional_accuracy_std_ticker": "ĐLC DA giữa mã",
            "rmse_std_seed_mean": "ĐLC RMSE giữa seed",
            "mae_std_seed_mean": "ĐLC MAE giữa seed",
            "r2_std_seed_mean": "ĐLC R² giữa seed",
            "directional_accuracy_std_seed_mean": "ĐLC DA giữa seed",
        }
    )


def main() -> None:
    # PowerShell cũ có thể mặc định cp1252; bảo đảm thông báo tiếng Việt không
    # làm hỏng một lượt sinh báo cáo vốn đã hoàn tất ghi tệp.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Sinh báo cáo NCKH Markdown và CSV tiếng Việt.")
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()
    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    required = {"model", "ticker", "seed", "parameter_count", "window", "feature_set", "rmse", "mae", "r2", "directional_accuracy"}
    if not required.issubset(metrics.columns):
        raise ValueError("Kết quả thuộc schema cũ. Hãy chạy lại run_experiments.py.")
    metrics = metrics.loc[metrics["window"].eq(args.window)].copy()
    if metrics["ticker"].nunique() != 8 or len(metrics) != 136:
        raise ValueError("Lần chạy đầy đủ cần 136 dòng metric: 8 mã × (2 baseline + 3 mạng × 5 seed).")
    metrics["_order"] = metrics["model"].map(MODEL_ORDER)
    metrics = metrics.sort_values(["ticker", "_order", "seed"]).drop(columns="_order")
    per_ticker = per_ticker_summary(args.window)
    summary = save_group_summary(args.window)
    summary["_order"] = summary["model"].map(MODEL_ORDER)
    summary = summary.sort_values("_order").drop(columns="_order")
    predictions = pd.read_csv(RESULTS_DIR / "predictions.csv")
    predictions = predictions.loc[predictions["window"].eq(args.window)].copy()
    tests = statistical_tests(predictions)

    detail_vi = metrics.copy()
    detail_vi["model"] = detail_vi["model"].map(MODEL_NAMES)
    detail_vi["feature_set"] = detail_vi["feature_set"].map(FEATURE_NAMES)
    detail_vi = detail_vi.rename(columns={"model": "Mô hình", "ticker": "Mã", "seed": "Seed", "window": "Cửa sổ phiên", "feature_set": "Tập đặc trưng", "parameter_count": "Số tham số", "rmse": "RMSE", "mae": "MAE", "r2": "R²", "directional_accuracy": "DA"})
    summary_vi = vietnamese_summary(summary)
    tests_vi = tests.copy()
    tests_vi["model"] = tests_vi["model"].map(MODEL_NAMES)
    tests_vi = tests_vi.rename(columns={"model": "Mô hình", "ticker": "Mã", "seed": "Seed", "n_test": "Số phiên test", "mean_squared_error_improvement": "Cải thiện MSE so với zero", "hac_statistic": "Thống kê HAC", "p_value_two_sided": "p hai phía"})
    detail_vi.to_csv(RESULTS_DIR / "BANG_KET_QUA_CHI_TIET_VI.csv", index=False, encoding="utf-8-sig")
    summary_vi.to_csv(RESULTS_DIR / "BANG_TONG_HOP_NHOM_VI.csv", index=False, encoding="utf-8-sig")
    tests_vi.to_csv(RESULTS_DIR / "KIEM_DINH_HAC_VI.csv", index=False, encoding="utf-8-sig")
    predictions_vi = predictions.rename(
        columns={
            "model": "Mô hình",
            "ticker": "Mã",
            "window": "Cửa sổ phiên",
            "feature_set": "Tập đặc trưng",
            "seed": "Seed",
            "parameter_count": "Số tham số",
            "date": "Ngày mục tiêu",
            "actual_return": "Log-return thực tế",
            "predicted_return": "Log-return dự báo",
        }
    )
    predictions_vi["Mô hình"] = predictions_vi["Mô hình"].map(MODEL_NAMES)
    predictions_vi["Tập đặc trưng"] = predictions_vi["Tập đặc trưng"].map(FEATURE_NAMES)
    predictions_vi.to_csv(RESULTS_DIR / "DU_BAO_THEO_NGAY_VI.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    formatted_summary = format_metrics(summary_vi, tuple(column for column in summary_vi.columns if "trung bình" in column.lower() or "ĐLC" in column or column == "DA"))
    manifest = json.loads((RESULTS_DIR / "EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    warnings = manifest["data_manifest"].get("quality_warnings", [])
    winner = MODEL_NAMES[summary.loc[summary["rmse_mean"].idxmin(), "model"]]
    baseline_row = summary.loc[summary["model"].eq("train_mean")].iloc[0]
    neural_rows = summary.loc[summary["model"].isin(["lstm", "mamba_baseline", "improved_mamba"])]
    best_neural_row = neural_rows.loc[neural_rows["rmse_mean"].idxmin()]
    ssm_row = summary.loc[summary["model"].eq("mamba_baseline")].iloc[0]
    ssm_vnindex_row = summary.loc[summary["model"].eq("improved_mamba")].iloc[0]
    neural_gap = float(best_neural_row["rmse_mean"] - baseline_row["rmse_mean"])
    vnindex_gap = float(ssm_vnindex_row["rmse_mean"] - ssm_row["rmse_mean"])
    warning_text = "; ".join(warnings).rstrip(".") if warnings else "không có"
    git_dirty = "có" if manifest.get("git_worktree_dirty") else "không"
    report = f"""# Báo cáo kết quả thí nghiệm dự báo tỷ suất sinh lợi cổ phiếu ngân hàng

## 1. Phạm vi và thiết kế thực nghiệm

- Mã cổ phiếu: {", ".join(manifest["experiment"]["tickers"])}.
- Dữ liệu: OHLCV hằng ngày và VN-Index, snapshot kết thúc ngày {manifest["data_manifest"]["files"][0]["end"]}.
- Nhiệm vụ: dùng cửa sổ {args.window} phiên kết thúc tại t để dự báo log return ở t+1.
- Chia thời gian: train đến 31/12/2023; validation trong năm 2024; test từ 01/01/2025.
- Mô hình: baseline zero return, baseline trung bình train, LSTM, Selective SSM cơ sở và Selective SSM có thêm return VN-Index.
- Seed mạng: {", ".join(map(str, manifest["experiment"]["seeds"]))}. Baseline là xác định nên chỉ có seed 0.
- Git HEAD: `{manifest["git_commit"]}`; worktree có thay đổi chưa commit tại lúc ghi manifest: {git_dirty}. SHA-256 các tệp mã chính nằm trong `EXPERIMENT_MANIFEST.json`. Python {manifest["python"]}; PyTorch {manifest["packages"]["torch"]}; chạy CUDA: {manifest["cuda_available"]}.

## 2. Kết quả trung bình nhóm

{markdown_table(formatted_summary)}

## 3. Diễn giải có kiểm soát

Mô hình có RMSE trung bình thấp nhất trong lần chạy này là **{winner}**. Trong các mạng nơ-ron, **{MODEL_NAMES[best_neural_row["model"]]}** có RMSE trung bình {best_neural_row["rmse_mean"]:.6f}, cao hơn baseline trung bình train ({baseline_row["rmse_mean"]:.6f}) một lượng {neural_gap:.6f}. Vì vậy, thiết lập hiện tại **chưa cho thấy mạng nơ-ron giảm sai số trung bình nhóm so với baseline train mean**. Selective SSM thêm VN-Index có RMSE chênh {vnindex_gap:+.6f} so với Selective SSM cơ sở; dấu dương nghĩa là không có cải thiện RMSE trong lần chạy này. Kết luận chỉ áp dụng cho snapshot dữ liệu, tập đặc trưng, khoảng thời gian và siêu tham số đã nêu. RMSE/MAE đo độ lớn sai số; DA đo đúng chiều biến động; vì vậy không được chọn mô hình chỉ theo DA. Nếu R² âm, mô hình còn kém hơn dự báo hằng bằng trung bình tập test theo thước đo R².

## 4. Kiểm định chênh lệch squared loss

Tệp `KIEM_DINH_HAC_VI.csv` so sánh từng seed mạng với baseline zero return trên cùng ngày kiểm tra. Cột “Cải thiện MSE so với zero” dương nghĩa là mạng giảm squared loss trung bình. p-value dùng xấp xỉ chuẩn với phương sai HAC; đây là bằng chứng bổ sung, không thay thế việc đọc RMSE/MAE và phân tích từng mã.

## 5. Kiểm soát tái lập và chất lượng dữ liệu

- `EXPERIMENT_MANIFEST.json` lưu commit, phiên bản môi trường, thiết bị, cấu hình, SHA-256 và phạm vi từng file dữ liệu.
- `DATA_MANIFEST.json` trong thư mục data lưu SHA-256 để xác nhận đúng snapshot đầu vào.
- Dữ liệu không được chia ngẫu nhiên; scaler chỉ fit trên train; target được dịch sang phiên tương lai.
- Cảnh báo dữ liệu nguồn: {warning_text}.

## 6. Tệp đầu ra

- `BANG_KET_QUA_CHI_TIET_VI.csv`: metric của mọi mã, mô hình và seed.
- `BANG_TONG_HOP_NHOM_VI.csv`: bảng dùng trong báo cáo, tách độ lệch chuẩn giữa mã và giữa seed.
- `KIEM_DINH_HAC_VI.csv`: kiểm định chênh lệch squared loss so với baseline zero.
- `DU_BAO_THEO_NGAY_VI.csv`: log return thực tế và dự báo ở từng ngày test, với tiêu đề tiếng Việt.
- `per_ticker_summary.csv`, `group_summary.csv`, `statistical_tests.csv`: bảng kỹ thuật để tái lập.
"""
    (RESULTS_DIR / "BAO_CAO_KET_QUA.md").write_text(report, encoding="utf-8")
    report_dir = Path("bao_cao")
    report_dir.mkdir(exist_ok=True)
    full_report = f"""# Báo cáo NCKH dự báo tỷ suất sinh lợi cổ phiếu ngân hàng Việt Nam bằng Selective State Space Model

## Tóm tắt

Nghiên cứu xây dựng quy trình dự báo log return một phiên kế tiếp cho tám cổ phiếu ngân hàng Việt Nam. Dữ liệu OHLCV và VN-Index trong giai đoạn 2020-2026 được chia theo thời gian thành tập huấn luyện, xác thực và kiểm tra. Nghiên cứu đối chiếu hai baseline hằng số, LSTM, Selective SSM cơ sở và Selective SSM có thêm return VN-Index. Mọi cấu hình mạng được lặp trên năm seed; kết quả được đánh giá bằng RMSE, MAE, R², độ chính xác hướng và kiểm định chênh lệch squared loss. Kết quả chi tiết được sinh tự động từ mã nguồn và cần được diễn giải trong phạm vi snapshot dữ liệu đã công bố.

**Từ khóa:** dự báo tỷ suất sinh lợi; cổ phiếu ngân hàng; Selective State Space Model; Mamba; LSTM; thị trường chứng khoán Việt Nam.

## 1. Mở đầu

Dự báo return ngắn hạn là bài toán hồi quy có độ nhiễu cao. Mục tiêu của nghiên cứu không phải đưa ra khuyến nghị đầu tư mà là đánh giá công bằng khả năng dự báo của các kiến trúc chuỗi thời gian trên cùng dữ liệu và cùng quy trình không rò rỉ dữ liệu. Câu hỏi nghiên cứu là liệu Selective SSM có giảm sai số dự báo so với baseline và LSTM, và việc thêm return VN-Index có hỗ trợ mô hình hay không.

## 2. Dữ liệu và phương pháp

Đầu vào gồm open, high, low, close, volume; các đặc trưng dẫn xuất là log return, thay đổi khối lượng, biên độ high-low chuẩn hóa và log return VN-Index. Nhãn là `log(close_(t+1) / close_t)`. Mỗi mẫu dùng 30 phiên kết thúc tại t để dự báo t+1. StandardScaler chỉ fit trên các quan sát thuộc train. Train kéo dài đến 31/12/2023, validation là năm 2024, test bắt đầu từ 01/01/2025.

Các mạng dùng hidden size 8, một lớp, Adam với learning rate 0,001, tối đa 40 epoch và early stopping sau 8 epoch không cải thiện trên validation. Nhãn được chuẩn hóa bằng trung bình và độ lệch chuẩn của train, rồi đảo chuẩn hóa trước đánh giá. LSTM có 585 tham số; Selective SSM cơ sở có 545 tham số và bản thêm VN-Index có 553 tham số. Selective SSM được tự triển khai từ mô tả thuật toán Mamba; mã nguồn MambaStock không được đưa vào kho.

## 3. Kết quả

{markdown_table(formatted_summary)}

Kết quả theo từng mã, từng seed, dự báo theo ngày và kiểm định HAC nằm trong các CSV đi kèm. Mô hình có RMSE trung bình thấp nhất là **{winner}**. Mạng tốt nhất theo RMSE là **{MODEL_NAMES[best_neural_row["model"]]}** ({best_neural_row["rmse_mean"]:.6f}), nhưng vẫn cao hơn baseline trung bình train ({baseline_row["rmse_mean"]:.6f}) {neural_gap:.6f}. Vì vậy, câu trả lời trung thực cho câu hỏi nghiên cứu trong cấu hình hiện tại là: chưa có bằng chứng rằng mạng Selective SSM hoặc LSTM cải thiện sai số trung bình nhóm so với baseline hằng. Thêm return VN-Index cũng không giảm RMSE so với Selective SSM cơ sở (chênh {vnindex_gap:+.6f}). Kết luận này không khẳng định ưu thế tổng quát ngoài khoảng dữ liệu và thiết lập đã kiểm tra.

## 4. Thảo luận và giới hạn

Return một phiên chịu ảnh hưởng của thông tin mới và nhiễu thị trường, nên R² âm cần được báo cáo minh bạch. Độ chính xác hướng không thay thế sai số hồi quy. Nghiên cứu chưa dùng dữ liệu tin tức, sentiment, yếu tố vĩ mô, báo cáo tài chính hoặc nhãn biến động/rủi ro. Dữ liệu VN-Index có cảnh báo nguồn được ghi trong manifest; pipeline chỉ dùng close để tính return và không tự thay đổi số liệu gốc.

## 5. Kết luận

Mã nguồn cung cấp quy trình tái lập từ dữ liệu, kiểm tra chất lượng, tạo đặc trưng, baseline, huấn luyện nhiều seed, lưu manifest và sinh báo cáo. Kết quả hiện tại là một kết quả âm có giá trị: các feature và cấu hình đang dùng chưa vượt baseline train mean ở trung bình nhóm. Hướng mở rộng là walk-forward validation, thêm đặc trưng ngoài giá-khối lượng, dự báo độ biến động tương lai và phân loại rủi ro.

## Tài liệu tham khảo

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
2. Shi, Z. (2024). MambaStock: Selective State Space Model for Stock Prediction. arXiv:2402.18959.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
"""
    (report_dir / "BAO_CAO_NCKH.md").write_text(full_report, encoding="utf-8")
    print("Đã tạo báo cáo tại results/ và bao_cao/BAO_CAO_NCKH.md")


if __name__ == "__main__":
    main()
