"""Đánh giá nhanh bộ chấm sentiment bằng mẫu do sinh viên gán nhãn thủ công."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from console import configure_utf8_output


LABELS = ["negative", "neutral", "positive"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mẫu sentiment đã gán nhãn thủ công.")
    parser.add_argument("--sample", default="data/news_validation_sample.csv")
    parser.add_argument("--minimum-reviewed", type=int, default=30)
    parser.add_argument(
        "--prediction-column",
        default="sentiment_label",
        help="Cột nhãn tự động cần so sánh (từ điển hoặc PhoBERT).",
    )
    args = parser.parse_args()
    path = Path(args.sample)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}; hãy kiểm tra snapshot dữ liệu trong thư mục data/.")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {args.prediction_column, "human_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} thiếu cột {sorted(missing)}.")
    frame["human_label"] = frame["human_label"].str.lower().str.strip()
    reviewed = frame.loc[frame["human_label"].isin(LABELS)].copy()
    invalid = frame.loc[(frame["human_label"] != "") & ~frame["human_label"].isin(LABELS)]
    if not invalid.empty:
        raise ValueError("human_label chỉ nhận negative, neutral hoặc positive.")
    if len(reviewed) < args.minimum_reviewed:
        raise ValueError(
            f"Mới gán nhãn {len(reviewed)} dòng; cần ít nhất {args.minimum_reviewed} dòng để kiểm tra."
        )
    print(f"Số tiêu đề đã kiểm tra: {len(reviewed)}")
    predicted = reviewed[args.prediction_column].astype(str).str.lower().str.strip()
    if not predicted.isin(LABELS).all():
        raise ValueError(f"{args.prediction_column} chỉ nhận negative, neutral hoặc positive.")
    print(f"Cột dự báo: {args.prediction_column}")
    print(f"Accuracy: {accuracy_score(reviewed['human_label'], predicted):.4f}")
    print("\nConfusion matrix (hàng=nhãn người, cột=nhãn mô hình):")
    print(pd.DataFrame(
        confusion_matrix(reviewed["human_label"], predicted, labels=LABELS),
        index=LABELS,
        columns=LABELS,
    ).to_string())
    print("\nBáo cáo theo lớp:")
    print(classification_report(
        reviewed["human_label"],
        predicted,
        labels=LABELS,
        digits=4,
        zero_division=0,
    ))


if __name__ == "__main__":
    configure_utf8_output()
    main()
