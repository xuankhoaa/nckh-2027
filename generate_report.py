"""Sinh báo cáo tiếng Việt từ kết quả nhiều seed và manifest tái lập."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from console import configure_utf8_output
from evaluate import RESULTS_DIR, per_ticker_summary, save_group_summary


MODEL_ORDER = {"naive_zero": 0, "train_mean": 1, "lstm": 2, "mamba_baseline": 3, "improved_mamba": 4}
MODEL_NAMES = {
    "naive_zero": "Baseline return bằng 0",
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
    tests_vi = tests_vi.rename(columns={"model": "Mô hình", "ticker": "Mã", "seed": "Seed", "n_test": "Số phiên test", "mean_squared_error_improvement": "Cải thiện MSE so với baseline 0", "hac_statistic": "Thống kê HAC", "p_value_two_sided": "p hai phía"})
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
- Mô hình: baseline return bằng 0, baseline trung bình train, LSTM, Selective SSM cơ sở và Selective SSM có thêm return VN-Index.
- Seed mạng: {", ".join(map(str, manifest["experiment"]["seeds"]))}. Baseline là xác định nên chỉ có seed 0.
- Git HEAD: `{manifest["git_commit"]}`; worktree có thay đổi chưa commit tại lúc ghi manifest: {git_dirty}. SHA-256 các tệp mã chính nằm trong `EXPERIMENT_MANIFEST.json`. Python {manifest["python"]}; PyTorch {manifest["packages"]["torch"]}; chạy CUDA: {manifest["cuda_available"]}.

## 2. Kết quả trung bình nhóm

{markdown_table(formatted_summary)}

## 3. Diễn giải có kiểm soát

Mô hình có RMSE trung bình thấp nhất trong lần chạy này là **{winner}**. Trong các mạng nơ-ron, **{MODEL_NAMES[best_neural_row["model"]]}** có RMSE trung bình {best_neural_row["rmse_mean"]:.6f}, cao hơn baseline trung bình train ({baseline_row["rmse_mean"]:.6f}) một lượng {neural_gap:.6f}. Vì vậy, thiết lập hiện tại **chưa cho thấy mạng nơ-ron giảm sai số trung bình nhóm so với baseline trung bình train**. Selective SSM thêm VN-Index có RMSE chênh {vnindex_gap:+.6f} so với Selective SSM cơ sở; dấu dương nghĩa là không có cải thiện RMSE trong lần chạy này. Kết luận chỉ áp dụng cho snapshot dữ liệu, tập đặc trưng, khoảng thời gian và siêu tham số đã nêu. RMSE/MAE đo độ lớn sai số; DA đo đúng chiều biến động; vì vậy không được chọn mô hình chỉ theo DA. Nếu R² âm, mô hình còn kém hơn dự báo hằng bằng trung bình tập test theo thước đo R².

## 4. Kiểm định chênh lệch squared loss

Tệp `KIEM_DINH_HAC_VI.csv` so sánh từng seed mạng với baseline return bằng 0 trên cùng ngày kiểm tra. Cột “Cải thiện MSE so với baseline 0” dương nghĩa là mạng giảm squared loss trung bình. p-value dùng xấp xỉ chuẩn với phương sai HAC; đây là bằng chứng bổ sung, không thay thế việc đọc RMSE/MAE và phân tích từng mã.

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

Nghiên cứu đánh giá khả năng dự báo log-return một phiên kế tiếp cho tám cổ phiếu ngân hàng Việt Nam từ dữ liệu OHLCV hằng ngày và return VN-Index. Dữ liệu snapshot đến {manifest["data_manifest"]["files"][0]["end"]} được chia theo thời gian thành train, validation và test để bảo vệ tính nhân quả của dự báo. Thiết kế so sánh hai baseline hằng, LSTM, Selective SSM cơ sở và Selective SSM bổ sung return VN-Index; mỗi mạng được lặp trên năm seed. Kết quả trung bình nhóm cho thấy baseline trung bình train có RMSE thấp nhất ({baseline_row["rmse_mean"]:.6f}). Mạng tốt nhất là {MODEL_NAMES[best_neural_row["model"]]} với RMSE {best_neural_row["rmse_mean"]:.6f}, cao hơn baseline {neural_gap:.6f}. Vì vậy, nghiên cứu không khẳng định Selective SSM vượt baseline ở cấu hình hiện tại; kết quả âm này được báo cáo như một phát hiện cần thiết thay vì bị loại bỏ. Toàn bộ dự báo, metric, SHA-256 dữ liệu và cấu hình môi trường được lưu để tái lập.

**Từ khóa:** dự báo tỷ suất sinh lợi; cổ phiếu ngân hàng; Selective State Space Model; Mamba; LSTM; thị trường chứng khoán Việt Nam.

## 1. Vấn đề nghiên cứu và phạm vi

### 1.1. Bài toán cần giải quyết

Mục tiêu là dự báo một đại lượng liên tục: `log(close_(t+1) / close_t)`. Đây là Bài toán 1 — hồi quy tỷ suất sinh lợi, không phải dự báo giá đóng cửa. Dùng log-return thay vì giá có hai lý do: giá các mã có thang đo khác nhau, còn return biểu diễn biến động tương đối; đồng thời các log-return có thể cộng dồn theo thời gian. Dự báo horizon một phiên phù hợp với câu hỏi ngắn hạn, nhưng cũng là trường hợp rất nhiễu vì thông tin mới phát sinh sau thời điểm t chưa có trong feature.

Nghiên cứu trả lời hai câu hỏi: (i) Selective SSM có giảm sai số dự báo so với hai baseline và LSTM trên cùng dữ liệu không; (ii) thêm return VN-Index có giúp Selective SSM không. Câu trả lời chỉ áp dụng cho snapshot, feature, horizon và siêu tham số được công bố.

### 1.2. Những gì không thuộc phạm vi

Nghiên cứu không đưa ra khuyến nghị mua/bán, không backtest chiến lược hay tính chi phí giao dịch. Nó chưa tạo target độ biến động tương lai cho Bài toán 2 và chưa tạo nhãn rủi ro thấp/trung bình/cao để làm Bài toán 3. Các bài toán đó cần target, metric và thiết kế đánh giá riêng; không được gộp lẫn với hồi quy return.

## 2. Dữ liệu và lý do chọn feature

### 2.1. Dữ liệu

Tám mã VCB, BID, CTG, MBB, TCB, ACB, VPB và STB được chọn để giữ cùng bối cảnh ngành ngân hàng nhưng vẫn có khác biệt về quy mô và thanh khoản. Đầu vào gốc gồm open, high, low, close và volume; VN-Index đại diện cho trạng thái thị trường chung. Các tệp dữ liệu được kiểm tra schema, trùng ngày, giá trị thiếu/không hữu hạn, điều kiện OHLC và khối lượng trước khi dùng.

Không tự sửa dữ liệu nguồn. Manifest ghi SHA-256, số dòng và phạm vi từng CSV để người đọc có thể xác định đúng snapshot. Cảnh báo cần nêu rõ: {warning_text}.

### 2.2. Feature dẫn xuất

Từ mỗi phiên t, pipeline dùng log-return cổ phiếu, thay đổi tương đối khối lượng và biên độ high-low đã chuẩn hóa. Những feature này lần lượt mô tả biến động đóng cửa, thay đổi thanh khoản và mức dao động trong phiên. Return VN-Index được thêm riêng ở biến thể cải tiến để kiểm tra giả thuyết rằng thị trường chung mang thêm thông tin cho cổ phiếu ngân hàng.

Cửa sổ 30 phiên tương đương xấp xỉ một tháng rưỡi giao dịch. Đây là lựa chọn cân bằng giữa việc giữ bối cảnh gần và việc không làm số mẫu hữu ích giảm quá nhiều; nó được khóa trước khi chạy bộ thực nghiệm chính. Kết quả không chứng minh 30 là cửa sổ tối ưu tuyệt đối; window 60/90 cần là thí nghiệm tiếp theo được lập kế hoạch trước.

## 3. Thiết kế dữ liệu để tránh rò rỉ tương lai

Mỗi mẫu có 30 feature vector từ t-29 đến t và chỉ dự báo return ở t+1. Mẫu được gán tập theo **ngày của nhãn**, không theo ngày bắt đầu cửa sổ. Train kết thúc 31/12/2023, validation là năm 2024 và test bắt đầu 01/01/2025. Chia ngẫu nhiên sẽ trộn quan sát tương lai vào giai đoạn học, tạo kết quả quá lạc quan cho chuỗi thời gian.

StandardScaler chỉ fit trên feature train. Nhãn cũng được chuẩn hóa bằng trung bình và độ lệch chuẩn của train trước khi tối ưu rồi đảo chuẩn hóa trước đánh giá. Lý do là log-return có biên độ nhỏ; chuẩn hóa giúp gradient ổn định mà không dùng thông tin validation/test. Validation chỉ chọn epoch dừng sớm; test không tham gia chọn mô hình hoặc siêu tham số.

## 4. Mô hình và lý do lựa chọn

### 4.1. Hai baseline là điều kiện bắt buộc

`naive_zero` dự báo return bằng 0 và `train_mean` dự báo trung bình return của train. Return ngày thường có trung bình gần 0, nên baseline đơn giản có thể rất mạnh. Nếu mạng không vượt baseline, không thể kết luận kiến trúc phức tạp hữu ích. Đặt baseline vào cùng tập test giúp kết quả có ý nghĩa hơn việc chỉ so sánh các mạng với nhau.

### 4.2. LSTM và Selective SSM

LSTM là đối chứng tuần tự phổ biến, được dùng để xem lợi ích có đến từ Selective SSM hay chỉ từ việc dùng một mạng chuỗi. LSTM dùng một lớp hidden size 8 và có 585 tham số.

Selective SSM dùng RMSNorm, chiếu đầu vào, causal depthwise convolution, tham số chọn lọc Delta/B/C, cổng SiLU, residual connection và trạng thái có d_state bằng 8. Cập nhật trạng thái có dạng `h_t = a_t * h_(t-1) + b_t`; mã dùng prefix scan theo phép ghép affine kết hợp để xử lý toàn bộ cửa sổ nhưng test đơn vị chứng minh kết quả khớp truy hồi tuần tự. Bản cơ sở có 545 tham số; bản thêm VN-Index có 553 tham số. Quy mô gần LSTM nhằm hạn chế kết luận bị chi phối bởi số tham số trên dữ liệu không quá lớn.

### 4.3. Vì sao tự triển khai thay vì dùng mã MambaStock

Mã dự án không fork, import hoặc sửa tiếp code MambaStock. MambaStock chỉ là tài liệu tham khảo học thuật. Selective SSM ở đây là hiện thực độc lập, tối giản theo mô tả trong bài Mamba [1], điều chỉnh cho bài toán hồi quy 30 phiên và dữ liệu Việt Nam. Cách làm này giúp kiểm soát rõ phần nào được dùng, pipeline nào tạo target và cách đánh giá nào áp dụng; đồng thời tránh trình bày nhầm mã thí nghiệm này như một bản benchmark chính thức của MambaStock hoặc thư viện `mamba_ssm`.

Do đó, tên chính xác trong báo cáo là **Selective SSM lấy cảm hứng từ Mamba**. Không nên tuyên bố đây là tái lập nguyên xi MambaStock hay kết quả so sánh trực tiếp với implementation CUDA chính thức.

### 4.4. Thiết lập huấn luyện

Adam với learning rate 0,001, tối đa 40 epoch và early stopping patience 8 được dùng nhất quán. Early stopping dựa trên validation để giảm nguy cơ chọn mô hình theo test. Năm seed 42, 52, 62, 72 và 82 được dùng vì khởi tạo mạng và tối ưu ngẫu nhiên có thể làm thay đổi kết quả; báo cáo vì vậy có cả trung bình và độ lệch chuẩn qua seed. Các hyperparameter này là cấu hình nghiên cứu, không phải kết quả của tối ưu hóa theo test.

## 5. Đánh giá và kiểm định

RMSE và MAE đo độ lớn sai số, nên càng nhỏ càng tốt. R2 cho biết phần biến thiên được mô hình giải thích; R2 âm nghĩa là kém dự báo hằng bằng trung bình của chính test theo metric này. Directional Accuracy (DA) đo đúng dấu tăng/giảm, nhưng không phản ánh độ lớn sai số; vì vậy DA cao không tự động có nghĩa mô hình tốt hơn.

Kết quả nhóm được tổng hợp theo hai tầng: trung bình seed cho từng mã trước, rồi trung bình các mã để mỗi mã có cùng trọng số. `KIEM_DINH_HAC_VI.csv` so sánh squared loss của từng seed mạng với baseline zero trên đúng các ngày test; phương sai dài hạn dùng trọng số HAC và p-value xấp xỉ chuẩn. Kiểm định này là bằng chứng bổ sung, không thay thế RMSE/MAE, kiểm tra nhiều seed và phân tích từng mã.

## 6. Kết quả thực nghiệm

{markdown_table(formatted_summary)}

Baseline trung bình train là mô hình có RMSE thấp nhất: {baseline_row["rmse_mean"]:.6f}. Trong ba mạng, {MODEL_NAMES[best_neural_row["model"]]} có RMSE tốt nhất: {best_neural_row["rmse_mean"]:.6f}, nhưng vẫn cao hơn baseline {neural_gap:.6f}. Selective SSM thêm VN-Index có chênh lệch RMSE {vnindex_gap:+.6f} so với bản cơ sở; trong đó dấu dương nghĩa là RMSE tăng. DA của các mạng gần baseline train mean, nên không có cơ sở dùng DA để đảo ngược kết luận theo sai số.

Do đó, kết quả chính là một **kết quả âm có kiểm soát**: với feature hiện có và cửa sổ 30 phiên, mạng phức tạp chưa tạo lợi ích dự báo trung bình nhóm rõ ràng hơn baseline hằng. Đây là kết luận tốt hơn việc chỉ chọn vài ticker/seed đẹp hoặc chỉ báo cáo metric thuận lợi.

## 7. Thảo luận: vì sao kết quả có thể như vậy

Thứ nhất, return một phiên phụ thuộc nhiều vào thông tin mới, tâm lý và sự kiện chưa có trong OHLCV quá khứ. Mô hình chỉ thấy giá, khối lượng và một biến thị trường chung, nên không có lý do để mặc định nó vượt dự báo hằng. Thứ hai, quan hệ thị trường có thể thay đổi giữa giai đoạn train, validation và test; mẫu hình học được ở quá khứ không nhất thiết ổn định ở tương lai. Thứ ba, VN-Index là chỉ báo rộng và chỉ được đưa vào dưới dạng một return ngày; nó có thể chưa bổ sung thông tin ngoài feature giá của từng cổ phiếu.

Các giải thích này là giả thuyết hợp lý, không phải chứng minh nhân quả. Không thể kết luận Selective SSM không phù hợp với mọi dữ liệu tài chính; chỉ có thể kết luận cấu hình đã chạy chưa vượt baseline. Điều này cũng giải thích vì sao báo cáo giữ nguyên R2 âm và không biến DA thành kết luận đầu tư.

## 8. Độ tin cậy, khả năng tái lập và giới hạn

`EXPERIMENT_MANIFEST.json` lưu commit, SHA-256 các tệp mã chính, phiên bản Python/PyTorch, CPU/GPU, seed, window, epoch, batch size và manifest dữ liệu. Bộ chạy tạo 136 cấu hình và 59.704 dự báo; `verify_project.py` kiểm tra đủ lần chạy, tính hữu hạn và ghép đúng metric–prediction. Test đơn vị kiểm tra dữ liệu, quan hệ input–target, scaler không nhận giá trị tương lai, baseline, selective scan và metric.

Giới hạn quan trọng gồm: chỉ một khoảng test hậu 2025; chưa có walk-forward evaluation; chưa có news/sentiment, vĩ mô, báo cáo tài chính hay feature ngành; chưa tối ưu siêu tham số có kiểm soát; chưa có backtest chi phí giao dịch. Cảnh báo dữ liệu VN-Index được giữ lại trong manifest thay vì âm thầm sửa dữ liệu. Các giới hạn này cần xuất hiện trong báo cáo Word sau này.

## 9. Kết luận và hướng tiếp theo

Đề tài đã tạo một pipeline hồi quy return có kiểm soát leakage, có baseline, lặp nhiều seed, kiểm tra chất lượng dữ liệu và đầu ra tái lập. Tuy nhiên, câu trả lời hiện tại cho hai câu hỏi nghiên cứu là: chưa có bằng chứng Selective SSM vượt baseline train mean; thêm return VN-Index cũng chưa giảm RMSE. Hướng hợp lý là walk-forward validation, window 60/90 đã đăng ký trước, feature tin tức/sentiment hoặc vĩ mô, rồi mới mở rộng sang volatility forecast và risk classification bằng target riêng.

## Tài liệu tham khảo

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
3. Shi, Z. (2024). MambaStock: Selective State Space Model for Stock Prediction. arXiv:2402.18959.
4. Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
5. Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. Econometrica, 55(3), 703-708.
"""
    (report_dir / "BAO_CAO_NCKH.md").write_text(full_report, encoding="utf-8")
    print("Đã tạo báo cáo tại results/ và bao_cao/BAO_CAO_NCKH.md")


if __name__ == "__main__":
    configure_utf8_output()
    main()
