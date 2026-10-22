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

from config import DATA_SOURCE, TICKERS
from data_quality import sha256_file, write_data_manifest
from evaluate import RESULTS_DIR


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILES = (
    "config.py",
    "data_quality.py",
    "preprocess.py",
    "baselines.py",
    "models/lstm.py",
    "models/mamba_model.py",
    "train.py",
    "evaluate.py",
    "run_experiments.py",
)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_dirty() -> bool | None:
    """Ghi rõ khi HEAD chưa bao gồm thay đổi mã đã dùng để chạy thí nghiệm."""
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL
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
    window: int,
    seeds: Iterable[int],
    tickers: Iterable[str],
    epochs: int,
    batch_size: int,
    started_at: str,
) -> Path:
    """Ghi metadata không nhạy cảm để kiểm chứng hoặc tái lập lần chạy."""
    symbols = [*dict.fromkeys([*tickers, "VNINDEX"])]
    manifest_path = write_data_manifest(Path("data"), symbols, DATA_SOURCE)
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "source_sha256": {
            path: sha256_file(PROJECT_DIR / path) for path in SOURCE_FILES
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in ("torch", "pandas", "numpy", "scikit-learn", "vnstock")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_access": "CSV cache cục bộ; Vnstock không được import trong lần huấn luyện này.",
        "experiment": {
            "tickers": list(tickers),
            "seeds": list(seeds),
            "window": window,
            "epochs": epochs,
            "batch_size": batch_size,
            "models": ["naive_zero", "train_mean", "lstm", "mamba_baseline", "improved_mamba"],
        },
        "data_manifest": data_manifest,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "EXPERIMENT_MANIFEST.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    write_experiment_manifest(window=30, seeds=(42, 52, 62, 72, 82), tickers=TICKERS, epochs=40, batch_size=128, started_at=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
