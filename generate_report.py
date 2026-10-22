"""Sinh báo cáo Markdown trực tiếp từ kết quả thí nghiệm đã lưu."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import RESULTS_DIR, save_group_summary


MODEL_ORDER = {"lstm": 0, "mamba_baseline": 1, "improved_mamba": 2}
MODEL_NAMES = {
    "lstm": "LSTM",
    "mamba_baseline": "Mamba baseline (OHLCV)",
    "improved_mamba": "Improved Mamba (OHLCV + VN-Index)",
}
FEATURE_NAMES = {
    "ohlcv_derived": "OHLCV và feature dẫn xuất",
    "ohlcv_derived_plus_vnindex": "OHLCV, feature dẫn xuất và return VN-Index",
}


def markdown_table(frame: pd.DataFrame) -> str:
    """Tạo bảng Markdown không cần thêm thư viện tabulate."""
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def data_inventory(data_dir: Path, tickers: list[str]) -> pd.DataFrame:
    records = []
    for ticker in tickers:
        frame = pd.read_csv(data_dir / f"{ticker}.csv")
        records.append(
            {
                "ticker": ticker,
                "rows": len(frame),
                "from": frame["time"].iloc[0],
                "to": frame["time"].iloc[-1],
                "missing_values": int(frame.isna().sum().sum()),
            }
        )
    return pd.DataFrame(records)


def vietnamese_detail_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Đổi nhãn kỹ thuật sang tiếng Việt cho bảng dùng trong báo cáo và Excel."""
    result = metrics.copy()
    result["model"] = result["model"].map(MODEL_NAMES)
    result["feature_set"] = result["feature_set"].map(FEATURE_NAMES)
    return result.rename(
        columns={
            "ticker": "Mã cổ phiếu",
            "model": "Mô hình",
            "window": "Cửa sổ (phiên)",
            "feature_set": "Tập đặc trưng",
            "rmse": "RMSE",
            "mae": "MAE",
            "r2": "R²",
            "directional_accuracy": "Độ chính xác hướng",
        }
    )


def vietnamese_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["model"] = result["model"].map(MODEL_NAMES)
    result["feature_set"] = result["feature_set"].map(FEATURE_NAMES)
    return result.rename(
        columns={
            "model": "Mô hình",
            "window": "Cửa sổ (phiên)",
            "feature_set": "Tập đặc trưng",
            "tickers": "Số mã",
            "rmse_mean": "RMSE trung bình",
            "mae_mean": "MAE trung bình",
            "r2_mean": "R² trung bình",
            "directional_accuracy_mean": "Độ chính xác hướng trung bình",
            "rmse_std": "Độ lệch chuẩn RMSE",
            "mae_std": "Độ lệch chuẩn MAE",
            "r2_std": "Độ lệch chuẩn R²",
            "directional_accuracy_std": "Độ lệch chuẩn độ chính xác hướng",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo báo cáo Markdown từ kết quả đã chạy.")
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    metrics = pd.read_csv(RESULTS_DIR / "metrics.csv")
    metrics = metrics.loc[metrics["window"] == args.window].copy()
    metrics["_order"] = metrics["model"].map(MODEL_ORDER)
    metrics = metrics.sort_values(["ticker", "_order"]).drop(columns="_order")
    if metrics["ticker"].nunique() != 8 or len(metrics) != 24:
        raise ValueError("Báo cáo toàn nhóm cần đúng 24 dòng: 8 mã × 3 mô hình.")

    summary = save_group_summary(args.window).copy()
    summary["_order"] = summary["model"].map(MODEL_ORDER)
    summary = summary.sort_values("_order").drop(columns="_order")
    tickers = sorted(metrics["ticker"].unique())
    inventory = data_inventory(Path("data"), tickers + ["VNINDEX"])

    # Các CSV tiếng Việt có BOM để Excel trên Windows hiển thị đúng dấu.
    detail_vi = vietnamese_detail_table(metrics)
    summary_vi = vietnamese_summary_table(summary)
    detail_vi.to_csv(RESULTS_DIR / "BANG_KET_QUA_CHI_TIET_VI.csv", index=False, encoding="utf-8-sig")
    summary_vi.to_csv(RESULTS_DIR / "BANG_TONG_HOP_NHOM_VI.csv", index=False, encoding="utf-8-sig")

    predictions = pd.read_csv(RESULTS_DIR / "predictions.csv")
    predictions["model"] = predictions["model"].map(MODEL_NAMES)
    predictions["feature_set"] = predictions["feature_set"].map(FEATURE_NAMES)
    predictions.rename(
        columns={
            "model": "Mô hình",
            "ticker": "Mã cổ phiếu",
            "window": "Cửa sổ (phiên)",
            "feature_set": "Tập đặc trưng",
            "date": "Ngày mục tiêu",
            "actual_return": "Log return thực tế",
            "predicted_return": "Log return dự báo",
        }
    ).to_csv(RESULTS_DIR / "DU_BAO_THEO_NGAY_VI.csv", index=False, encoding="utf-8-sig")

    inventory_vi = inventory.rename(
        columns={
            "ticker": "Mã",
            "rows": "Số phiên",
            "from": "Từ ngày",
            "to": "Đến ngày",
            "missing_values": "Giá trị thiếu",
        }
    )

    display_metrics = detail_vi.copy()
    for column in ("RMSE", "MAE", "R²", "Độ chính xác hướng"):
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.6f}")
    display_summary = summary_vi.copy()
    for column in display_summary.columns:
        if "trung bình" in column.lower() or "Độ lệch chuẩn" in column:
            display_summary[column] = display_summary[column].map(lambda value: f"{value:.6f}")

    error_winner = MODEL_NAMES[summary.loc[summary["rmse_mean"].idxmin(), "model"]]
    direction_winner = MODEL_NAMES[summary.loc[summary["directional_accuracy_mean"].idxmax(), "model"]]

    report = f"""# Báo cáo kết quả thí nghiệm Mamba — nhóm ngân hàng Việt Nam

## Phạm vi lần chạy

- **Mã cổ phiếu:** {", ".join(tickers)}.
- **Mô hình:** (1) LSTM; (2) Mamba baseline dùng lõi MambaStock với OHLCV; (3) Improved Mamba dùng cùng lõi Mamba và thêm `vnindex_return`.
- **Cửa sổ:** {args.window} phiên. Input kết thúc tại phiên *t*, target là log return ở *t+1*.
- **Chia thời gian:** train đến 31/12/2023, validation trong 2024, test từ 01/01/2025 đến dữ liệu giao nhau mới nhất.
- **Chống leakage:** scaler chỉ fit trên phần train; không random split; target được dịch một phiên về phía tương lai.
- **Huấn luyện:** seed 42, Adam (`lr=0.001`), hidden size 16, 2 layers, batch size 128, tối đa 75 epoch và early stopping sau 12 epoch validation không cải thiện.

## Kiểm kê dữ liệu đầu vào

{markdown_table(inventory_vi)}

## Kết quả chi tiết theo mã

{markdown_table(display_metrics)}

## Trung bình nhóm ngân hàng

{markdown_table(display_summary)}

## Cách đọc kết quả

- RMSE và MAE càng thấp càng tốt.
- R² càng lớn càng tốt; giá trị âm cho thấy mô hình kém hơn dự báo hằng theo mean của tập test ở cấu hình này.
- Directional Accuracy càng cao càng tốt; 0.50 là mốc tham chiếu trực quan gần với đoán hướng ngẫu nhiên cân bằng.
- Không suy ra nguyên nhân nhân quả chỉ từ một lần chạy. Khi viết báo cáo, đối chiếu Mamba baseline với Improved Mamba trên từng ticker và trung bình nhóm; không lựa chọn mô hình dựa trên tập test.

## Kết luận tường minh của lần chạy này

- Mô hình có **RMSE trung bình thấp nhất**: **{error_winner}**.
- Mô hình có **Directional Accuracy trung bình cao nhất**: **{direction_winner}**.
- Hai kết luận trên không mâu thuẫn: dự báo đúng chiều nhiều hơn không đảm bảo sai số độ lớn nhỏ hơn. Vì thế cần báo cáo đồng thời cả bốn metric.

## Tệp kết quả đi kèm

- `BANG_KET_QUA_CHI_TIET_VI.csv`: bảng đầy đủ theo mã, nhãn tiếng Việt, mở đúng dấu trong Excel.
- `BANG_TONG_HOP_NHOM_VI.csv`: trung bình/độ lệch chuẩn theo mô hình, nhãn tiếng Việt.
- `DU_BAO_THEO_NGAY_VI.csv`: return thực tế và dự báo ở từng ngày test, nhãn tiếng Việt.
- `metrics.csv`, `predictions.csv`, `group_summary.csv`: bảng gốc kỹ thuật để tái lập/chạy lại code.
- `bieu_do/`: biểu đồ predicted-vs-actual, loss, và so sánh metrics theo từng mã/cả nhóm.
"""
    output = RESULTS_DIR / "BAO_CAO_KET_QUA.md"
    output.write_text(report, encoding="utf-8")
    print(f"Created {output}")


if __name__ == "__main__":
    main()
