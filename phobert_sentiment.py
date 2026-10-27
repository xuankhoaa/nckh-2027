"""Fine-tune PhoBERT và chấm cảm xúc tiêu đề tin ngân hàng.

Quy trình có hai bước tách biệt để dễ tái lập:

1. ``train`` fine-tune ``vinai/phobert-base-v2`` trên tập sentiment của
   ``undertheseanlp/UTS2017_Bank`` và đánh giá một lần trên test chính thức.
2. ``score`` dùng checkpoint đã huấn luyện để tạo ``data/news_phobert.csv``.

Điểm đưa vào mô hình dự báo là ``P(positive) - P(negative)`` trong [-1, 1].
Checkpoint có kích thước lớn nên được lưu cục bộ trong ``artifacts/`` và không
đưa lên Git; metadata và metric nhỏ được lưu trong ``results/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

from console import configure_utf8_output
from data_quality import sha256_file


BASE_MODEL = "vinai/phobert-base-v2"
DATASET_ID = "undertheseanlp/UTS2017_Bank"
DATASET_CONFIG = "sentiment"
LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
DEFAULT_MODEL_DIR = Path("artifacts/phobert_sentiment")
DEFAULT_METRICS_PATH = Path("results/phobert_sentiment_metrics.json")


def normalize_label(value: object) -> str:
    """Chuẩn hóa nhãn từ dataset/model về ba nhãn dùng trong dự án."""
    label = str(value).strip().lower()
    aliases = {
        "neg": "negative",
        "negative": "negative",
        "label_0": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "label_1": "neutral",
        "pos": "positive",
        "positive": "positive",
        "label_2": "positive",
    }
    if label not in aliases:
        raise ValueError(f"Nhãn sentiment không được hỗ trợ: {value!r}.")
    return aliases[label]


def score_from_probabilities(negative: float, positive: float) -> float:
    """Đổi xác suất lớp thành điểm có dấu trong [-1, 1]."""
    score = float(positive) - float(negative)
    return float(np.clip(score, -1.0, 1.0))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _word_segment(texts: Sequence[str]) -> list[str]:
    """Tách từ theo yêu cầu đầu vào PhoBERT bằng Underthesea."""
    try:
        from underthesea import word_tokenize
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường chạy NLP
        raise RuntimeError(
            "Thiếu underthesea. Hãy cài requirements.txt trước khi chạy PhoBERT."
        ) from exc
    return [word_tokenize(str(text), format="text") for text in texts]


class EncodedTextDataset(Dataset):
    """Dataset tensor nhỏ, phù hợp với 2.471 mẫu UTS2017_Bank."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: object,
        *,
        labels: Sequence[int] | None = None,
        max_length: int = 128,
    ) -> None:
        segmented = _word_segment(texts)
        self.encodings = tokenizer(
            segmented,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {name: values[index] for name, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    predictions: list[int]
    targets: list[int]


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> EvaluationResult:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {name: values.to(device) for name, values in batch.items()}
            logits = model(**inputs).logits
            losses.append(float(loss_fn(logits, labels).item()))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return EvaluationResult(
        loss=float(np.mean(losses)),
        accuracy=float(accuracy_score(targets, predictions)),
        macro_f1=float(f1_score(targets, predictions, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        predictions=predictions,
        targets=targets,
    )


def _class_weights(labels: Sequence[int], device: torch.device) -> torch.Tensor:
    """Căn bậc hai nghịch đảo tần suất để giảm lệch lớp nhưng tránh trọng số quá lớn."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(LABELS)).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Tập train thiếu lớp sentiment; số lượng={counts.tolist()}.")
    weights = np.sqrt(len(labels) / (len(LABELS) * counts))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _load_uts2017() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường chạy NLP
        raise RuntimeError(
            "Thiếu datasets. Hãy cài requirements.txt trước khi fine-tune PhoBERT."
        ) from exc
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)

    def to_frame(split: str) -> pd.DataFrame:
        frame = dataset[split].to_pandas().loc[:, ["text", "sentiment"]].copy()
        frame["sentiment"] = frame["sentiment"].map(normalize_label)
        if frame["text"].isna().any() or frame["sentiment"].isna().any():
            raise ValueError(f"UTS2017_Bank/{split} có dữ liệu thiếu.")
        return frame

    return to_frame("train"), to_frame("test")


def train_phobert(args: argparse.Namespace) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Thiết bị: {device}; mô hình nền: {args.base_model}")
    train_full, test_frame = _load_uts2017()
    train_frame, validation_frame = train_test_split(
        train_full,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=train_full["sentiment"],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)
    if args.freeze_encoder:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "classifier" in name

    def encoded(frame: pd.DataFrame) -> EncodedTextDataset:
        labels = [LABEL_TO_ID[label] for label in frame["sentiment"]]
        return EncodedTextDataset(
            frame["text"].astype(str).tolist(),
            tokenizer,
            labels=labels,
            max_length=args.max_length,
        )

    train_set = encoded(train_frame)
    validation_set = encoded(validation_frame)
    test_set = encoded(test_frame)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size)
    weights = _class_weights(train_set.labels.tolist(), device)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -math.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {name: values.to(device) for name, values in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
        validation = _evaluate(model, validation_loader, device, weights)
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation.loss,
            "validation_accuracy": validation.accuracy,
            "validation_macro_f1": validation.macro_f1,
            "validation_weighted_f1": validation.weighted_f1,
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch}/{args.epochs}: loss={epoch_row['train_loss']:.4f}, "
            f"val macro-F1={validation.macro_f1:.4f}"
        )
        if validation.macro_f1 > best_macro_f1:
            best_macro_f1 = validation.macro_f1
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    test_result = _evaluate(best_model, test_loader, device, weights)
    report = classification_report(
        test_result.targets,
        test_result.predictions,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "seed": args.seed,
        "device": str(device),
        "freeze_encoder": bool(args.freeze_encoder),
        "train_size": len(train_frame),
        "validation_size": len(validation_frame),
        "test_size": len(test_frame),
        "class_labels": list(LABELS),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
        },
        "history": history,
        "official_test": {
            "accuracy": test_result.accuracy,
            "macro_f1": test_result.macro_f1,
            "weighted_f1": test_result.weighted_f1,
            "loss": test_result.loss,
            "confusion_matrix": confusion_matrix(
                test_result.targets,
                test_result.predictions,
                labels=list(range(len(LABELS))),
            ).tolist(),
            "classification_report": report,
        },
        "limitations": [
            "UTS2017_Bank chủ yếu là phản hồi khách hàng, không phải tiêu đề chứng khoán.",
            "Lớp neutral rất ít; Macro-F1 và kết quả từng lớp phải được báo cáo cùng Accuracy.",
            "Cần kiểm định thêm trên mẫu tiêu đề tin do con người gán nhãn.",
        ],
    }
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Test: accuracy={test_result.accuracy:.4f}, macro-F1={test_result.macro_f1:.4f}; "
        f"đã lưu {metrics_path}."
    )


def _batched(values: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def score_news(args: argparse.Namespace) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tin: {input_path}.")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Chưa có checkpoint tại {model_dir}; hãy chạy lệnh train trước.")
    frame = pd.read_csv(input_path, keep_default_na=False)
    if "title" not in frame.columns:
        raise ValueError(f"{input_path.name} thiếu cột title.")
    if frame.empty:
        raise ValueError(f"{input_path.name} không có dữ liệu.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    probabilities: list[np.ndarray] = []
    titles = frame["title"].astype(str).tolist()
    completed = 0
    for raw_batch in _batched(titles, args.batch_size):
        segmented = _word_segment(raw_batch)
        inputs = tokenizer(
            segmented,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        inputs = {name: values.to(device) for name, values in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        completed += len(raw_batch)
        if completed % 1000 < len(raw_batch) or completed == len(titles):
            print(f"Đã chấm {completed}/{len(titles)} tiêu đề.")

    probs = np.concatenate(probabilities, axis=0)
    labels = probs.argmax(axis=1)
    output = frame.copy()
    output["sentiment_negative_prob"] = probs[:, LABEL_TO_ID["negative"]]
    output["sentiment_neutral_prob"] = probs[:, LABEL_TO_ID["neutral"]]
    output["sentiment_positive_prob"] = probs[:, LABEL_TO_ID["positive"]]
    output["sentiment_score"] = (
        output["sentiment_positive_prob"] - output["sentiment_negative_prob"]
    ).clip(-1, 1)
    output["sentiment_label"] = [ID_TO_LABEL[int(index)] for index in labels]
    output["sentiment_confidence"] = probs.max(axis=1)
    output["sentiment_method"] = "phobert_base_v2_uts2017_bank_v1"
    output["sentiment_model"] = str(model.config._name_or_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Đã lưu {len(output)} tin PhoBERT vào {output_path}.")


def audit_scored_news(args: argparse.Namespace) -> None:
    """Kiểm tra file đã chấm và ghi manifest nhỏ có checksum."""
    scored_path = Path(args.input)
    raw_path = Path(args.raw_input)
    metrics_path = Path(args.metrics_path)
    for path in (scored_path, raw_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {path}.")
    frame = pd.read_csv(scored_path)
    probability_columns = [
        "sentiment_negative_prob",
        "sentiment_neutral_prob",
        "sentiment_positive_prob",
    ]
    required = {
        "published_at", "ticker", "title", "sentiment_score", "sentiment_label",
        "sentiment_confidence", *probability_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{scored_path.name} thiếu cột {sorted(missing)}.")
    probs = frame[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("Xác suất PhoBERT phải hữu hạn và nằm trong [0,1].")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Ba xác suất PhoBERT không cộng thành 1.")
    if not frame["sentiment_score"].between(-1, 1).all():
        raise ValueError("sentiment_score nằm ngoài [-1,1].")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "phobert_base_v2_uts2017_bank_v1",
        "rows": int(len(frame)),
        "raw_rows": int(len(pd.read_csv(raw_path, usecols=["title"]))),
        "files": {
            str(raw_path): sha256_file(raw_path),
            str(scored_path): sha256_file(scored_path),
            str(metrics_path): sha256_file(metrics_path),
        },
        "label_counts": {str(key): int(value) for key, value in frame["sentiment_label"].value_counts().items()},
        "score_summary": {
            key: float(value)
            for key, value in frame["sentiment_score"].describe().items()
        },
        "probability_sum_max_absolute_error": float(np.max(np.abs(probs.sum(axis=1) - 1.0))),
        "uts2017_official_test": {
            key: metrics["official_test"][key]
            for key in ("accuracy", "macro_f1", "weighted_f1", "confusion_matrix")
        },
        "limitations": metrics["limitations"],
    }
    if payload["rows"] != payload["raw_rows"]:
        raise ValueError("Số dòng news PhoBERT không khớp news raw.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit hợp lệ: {payload['rows']} dòng; đã lưu {output_path}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune và áp dụng PhoBERT sentiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune trên UTS2017_Bank.")
    train_parser.add_argument("--base-model", default=BASE_MODEL)
    train_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    train_parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.1)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--max-length", type=int, default=128)
    train_parser.add_argument("--validation-size", type=float, default=0.15)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--freeze-encoder", action="store_true")
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(func=train_phobert)

    score_parser = subparsers.add_parser("score", help="Chấm sentiment cho dữ liệu tin.")
    score_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    score_parser.add_argument("--input", default="data/news_raw.csv")
    score_parser.add_argument("--output", default="data/news_phobert.csv")
    score_parser.add_argument("--batch-size", type=int, default=32)
    score_parser.add_argument("--max-length", type=int, default=128)
    score_parser.add_argument("--cpu", action="store_true")
    score_parser.set_defaults(func=score_news)

    audit_parser = subparsers.add_parser("audit", help="Kiểm tra file PhoBERT và ghi manifest.")
    audit_parser.add_argument("--input", default="data/news_phobert.csv")
    audit_parser.add_argument("--raw-input", default="data/news_raw.csv")
    audit_parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    audit_parser.add_argument("--output", default="data/NEWS_PHOBERT_MANIFEST.json")
    audit_parser.set_defaults(func=audit_scored_news)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    configure_utf8_output()
    main()
