"""Lưu dấu vết dữ liệu, môi trường và cấu hình của một lần thực nghiệm."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from console import configure_utf8_output
from config import DATA_SOURCE, TICKERS
from data_quality import sha256_file, sha256_text_file, write_data_manifest
from evaluate import RESULTS_DIR


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILES = (
    "config.py",
    "data_quality.py",
    "phobert_sentiment.py",
    "preprocess.py",
    "baselines.py",
    "models/lstm.py",
    "models/mamba_model.py",
    "console.py",
    "train.py",
    "evaluate.py",
    "run_experiments.py",
    "generate_report.py",
    "compare_sentiment_methods.py",
    "verify_project.py",
)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_dirty() -> bool | None:
    """Ghi rõ khi HEAD chưa bao gồm thay đổi ở các tệp mã nghiên cứu."""
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", *SOURCE_FILES],
                cwd=PROJECT_DIR,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_experiment_manifest(
    *,
    windows: Iterable[int],
    seeds: Iterable[int],
    tickers: Iterable[str],
    targets: Iterable[str],
    feature_sets: Iterable[str],
    return_horizons: Iterable[int],
    volatility_horizon: int,
    news_path: str,
    news_cutoff_hours: Iterable[int],
    tail_weights: Iterable[float],
    return_loss: str,
    volatility_loss: str,
    weight_decay: float,
    validation_shrinkage: bool,
    ridge_alphas: Iterable[float],
    include_ridge: bool,
    epochs: int,
    batch_size: int,
    started_at: str,
) -> Path:
    """Ghi metadata không nhạy cảm để kiểm chứng hoặc tái lập lần chạy."""
    symbols = [*dict.fromkeys([*tickers, "VNINDEX"])]
    manifest_path = write_data_manifest(Path("data"), symbols, DATA_SOURCE)
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    news_file = Path(news_path)
    news_manifest_file = news_file.parent / "NEWS_MANIFEST.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "source_sha256": {
            path: sha256_text_file(PROJECT_DIR / path) for path in SOURCE_FILES
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in ("torch", "pandas", "numpy", "scikit-learn", "transformers", "datasets", "underthesea")
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_access": "Snapshot CSV được công bố cùng repository; lần huấn luyện không gọi nguồn dữ liệu bên ngoài.",
        "experiment": {
            "tickers": list(tickers),
            "seeds": list(seeds),
            "windows": list(windows),
            "targets": list(targets),
            "feature_sets": list(feature_sets),
            "return_horizons": list(return_horizons),
            "volatility_horizon": volatility_horizon,
            "news_cutoff_hours": list(news_cutoff_hours),
            "tail_weights": list(tail_weights),
            "return_loss": return_loss,
            "volatility_loss": volatility_loss,
            "weight_decay": weight_decay,
            "validation_shrinkage": validation_shrinkage,
            "ridge_alphas": list(ridge_alphas),
            "include_ridge": include_ridge,
            "epochs": epochs,
            "batch_size": batch_size,
            "models": ["naive_zero", "train_mean", "ridge", "lstm", "mamba"],
        },
        "news_data": {
            "path": news_file.as_posix(),
            "exists": news_file.is_file(),
            "sha256": sha256_file(news_file) if news_file.is_file() else None,
            "collection_manifest_path": news_manifest_file.as_posix(),
            "collection_manifest_sha256": (
                sha256_file(news_manifest_file) if news_manifest_file.is_file() else None
            ),
        },
        "data_manifest": data_manifest,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    configure_utf8_output()
    write_experiment_manifest(
        windows=(30,),
        seeds=(42, 52, 62, 72, 82),
        tickers=TICKERS,
        targets=("return", "volatility"),
        feature_sets=("market_only",),
        return_horizons=(1, 5),
        volatility_horizon=5,
        news_path="data/news_phobert.csv",
        news_cutoff_hours=(15,),
        tail_weights=(1.0,),
        return_loss="huber",
        volatility_loss="mse",
        weight_decay=1e-4,
        validation_shrinkage=True,
        ridge_alphas=(1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0),
        include_ridge=True,
        epochs=40,
        batch_size=128,
        started_at=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    main()
