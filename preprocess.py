"""Leakage-safe feature engineering and sequence creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import BASE_FEATURES, VNINDEX_FEATURE, TimeSplit
from data_quality import REQUIRED_OHLCV, validate_ohlcv



@dataclass
class SequenceSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    dates_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    dates_valid: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    dates_test: np.ndarray
    feature_names: tuple[str, ...]
    alignment: dict[str, object]


def _read_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_OHLCV).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    frame = frame.loc[:, REQUIRED_OHLCV].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    for column in REQUIRED_OHLCV[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["time"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate dates.")
    frame = frame.sort_values("time").reset_index(drop=True)
    validate_ohlcv(frame, path.stem)
    return frame


def build_feature_frame(ticker: str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Create features known at market close on t and target return at t+1."""
    data_dir = Path(data_dir)
    stock = _read_ohlcv(data_dir / f"{ticker}.csv")
    vnindex = _read_ohlcv(data_dir / "VNINDEX.csv")

    stock["return"] = np.log(stock["close"] / stock["close"].shift(1))
    stock["volume_change"] = stock["volume"] / stock["volume"].shift(1) - 1.0
    stock["high_low_range"] = (stock["high"] - stock["low"]) / stock["close"]

    vnindex = vnindex.loc[:, ["time", "close"]].rename(columns={"close": "vnindex_close"})
    vnindex[VNINDEX_FEATURE] = np.log(vnindex["vnindex_close"] / vnindex["vnindex_close"].shift(1))

    frame = stock.merge(
        vnindex.loc[:, ["time", VNINDEX_FEATURE]],
        on="time",
        how="inner",
        validate="one_to_one",
    ).sort_values("time").reset_index(drop=True)

    # Both columns are shifted backward: row t holds the outcome and date of t+1.
    frame["target_return"] = frame["return"].shift(-1)
    frame["target_date"] = frame["time"].shift(-1)
    needed = [*BASE_FEATURES, VNINDEX_FEATURE, "target_return", "target_date"]
    return frame.dropna(subset=needed).reset_index(drop=True)


def _split_name(target_date: pd.Timestamp, split: TimeSplit) -> str:
    if target_date <= pd.Timestamp(split.train_end):
        return "train"
    if target_date <= pd.Timestamp(split.validation_end):
        return "valid"
    return "test"


def _stack(items: list[np.ndarray], width: int, window: int) -> np.ndarray:
    if not items:
        return np.empty((0, window, width), dtype=np.float32)
    return np.asarray(items, dtype=np.float32)


def make_sequence_split(
    frame: pd.DataFrame,
    include_vnindex: bool,
    window: int = 30,
    split: TimeSplit = TimeSplit(),
) -> SequenceSplit:
    """Build [sample, lookback, feature] tensors without fitting on future data."""
    if window < 2:
        raise ValueError("window must be at least 2.")

    feature_names = tuple(BASE_FEATURES + ((VNINDEX_FEATURE,) if include_vnindex else ()))
    train_end_rows = frame.loc[
        frame["target_date"] <= pd.Timestamp(split.train_end), "time"
    ]
    if train_end_rows.empty:
        raise ValueError("No rows are available for the requested training period.")

    # The scaler sees only market observations at or before the final training input day.
    final_train_input_day = train_end_rows.max()
    scaler = StandardScaler()
    scaler.fit(frame.loc[frame["time"] <= final_train_input_day, feature_names])
    features = scaler.transform(frame.loc[:, feature_names]).astype(np.float32)

    sequences: dict[str, list[np.ndarray]] = {"train": [], "valid": [], "test": []}
    targets: dict[str, list[float]] = {"train": [], "valid": [], "test": []}
    dates: dict[str, list[np.datetime64]] = {"train": [], "valid": [], "test": []}
    alignment: dict[str, object] | None = None

    for end_index in range(window - 1, len(frame)):
        input_end = frame.iloc[end_index]
        target_date = pd.Timestamp(input_end["target_date"])
        label = _split_name(target_date, split)
        sequences[label].append(features[end_index - window + 1 : end_index + 1])
        targets[label].append(float(input_end["target_return"]))
        dates[label].append(target_date.to_datetime64())

        if alignment is None:
            alignment = {
                "input_start_date": frame.iloc[end_index - window + 1]["time"].date().isoformat(),
                "input_end_date": input_end["time"].date().isoformat(),
                "target_date": target_date.date().isoformat(),
                "target_return": float(input_end["target_return"]),
            }

    if not all(sequences[name] for name in sequences):
        counts = {name: len(values) for name, values in sequences.items()}
        raise ValueError(f"At least one chronological split has no sequences: {counts}")
    assert alignment is not None
    if alignment["input_end_date"] >= alignment["target_date"]:
        raise AssertionError("Target must be after the final input day.")

    width = len(feature_names)
    return SequenceSplit(
        x_train=_stack(sequences["train"], width, window),
        y_train=np.asarray(targets["train"], dtype=np.float32),
        dates_train=np.asarray(dates["train"]),
        x_valid=_stack(sequences["valid"], width, window),
        y_valid=np.asarray(targets["valid"], dtype=np.float32),
        dates_valid=np.asarray(dates["valid"]),
        x_test=_stack(sequences["test"], width, window),
        y_test=np.asarray(targets["test"], dtype=np.float32),
        dates_test=np.asarray(dates["test"]),
        feature_names=feature_names,
        alignment=alignment,
    )


def print_split_summary(dataset: SequenceSplit) -> None:
    """Print the verifiable t -> t+1 alignment and chronological split sizes."""
    print("Input shape [samples, sequence_length, features]:", dataset.x_train.shape)
    print(
        "Chronological samples:",
        f"train={len(dataset.y_train)}, valid={len(dataset.y_valid)}, test={len(dataset.y_test)}",
    )
    print(
        "Alignment check: input",
        f"{dataset.alignment['input_start_date']} ... {dataset.alignment['input_end_date']}",
        "-> target",
        f"{dataset.alignment['target_date']}",
        f"(log return={dataset.alignment['target_return']:.6f})",
    )
