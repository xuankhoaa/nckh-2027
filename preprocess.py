"""Leakage-safe feature engineering and sequence creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    BASE_FEATURES,
    DEFAULT_NEWS_CUTOFF_HOUR,
    DEFAULT_RETURN_HORIZON,
    DEFAULT_VOLATILITY_HORIZON,
    FEATURE_SETS,
    NEWS_FEATURES,
    VNINDEX_FEATURE,
    TimeSplit,
)
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


def _news_timestamp(value: object) -> pd.Timestamp:
    """Chuẩn hóa thời gian tin; thời gian không có múi giờ được hiểu là giờ Việt Nam."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("published_at chứa thời gian không hợp lệ.")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("Asia/Ho_Chi_Minh")
    return timestamp.tz_convert("Asia/Ho_Chi_Minh")


def aggregate_news_features(
    news_path: Path | str,
    ticker: str,
    trading_dates: pd.Series,
    cutoff_hour: int = DEFAULT_NEWS_CUTOFF_HOUR,
) -> pd.DataFrame:
    """Gộp tin theo phiên đầu tiên mà tin đã thực sự sẵn có, không nhìn trước.

    CSV đầu vào ở mức bài báo phải có ``published_at``, ``ticker`` và
    ``sentiment_score`` trong [-1, 1]. Tin xuất bản từ giờ đóng cửa trở đi được
    chuyển sang phiên giao dịch kế tiếp; tin cuối tuần cũng được chuyển sang
    phiên kế tiếp.
    """
    if not 0 <= cutoff_hour <= 23:
        raise ValueError("cutoff_hour phải nằm trong [0, 23].")
    path = Path(news_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Thiếu dữ liệu tin tức {path}. Xem NEWS_DATA_SCHEMA.md; không được thay bằng dữ liệu giả."
        )
    news = pd.read_csv(path)
    required = {"published_at", "ticker", "sentiment_score"}
    missing = required.difference(news.columns)
    if missing:
        raise ValueError(f"{path.name} thiếu các cột: {sorted(missing)}")
    news = news.loc[:, ["published_at", "ticker", "sentiment_score"]].copy()
    news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
    news = news.loc[news["ticker"].eq(ticker.upper())].copy()
    if news.empty:
        raise ValueError(f"Không có bài báo nào cho {ticker} trong {path.name}.")
    news["sentiment_score"] = pd.to_numeric(news["sentiment_score"], errors="raise")
    if not np.isfinite(news["sentiment_score"]).all() or not news["sentiment_score"].between(-1, 1).all():
        raise ValueError("sentiment_score phải là số hữu hạn trong [-1, 1].")
    news["published_local"] = news["published_at"].map(_news_timestamp)
    news["available_date"] = news["published_local"].map(
        lambda value: value.normalize().tz_localize(None)
        + pd.Timedelta(days=int(value.hour >= cutoff_hour))
    )

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates).drop_duplicates().sort_values())
    positions = calendar.searchsorted(pd.DatetimeIndex(news["available_date"]), side="left")
    news = news.loc[positions < len(calendar)].copy()
    positions = positions[positions < len(calendar)]
    if news.empty:
        raise ValueError(f"Tin của {ticker} không giao với khoảng ngày thị trường đang nghiên cứu.")
    news["time"] = calendar.take(positions).to_numpy()
    daily = news.groupby("time", as_index=False).agg(
        sentiment_score=("sentiment_score", "mean"),
        news_count=("sentiment_score", "size"),
    )
    daily["news_count"] = daily["news_count"].astype(float)
    return daily


def build_feature_frame(
    ticker: str,
    data_dir: Path | str = "data",
    *,
    include_news: bool = False,
    news_path: Path | str | None = None,
    news_cutoff_hour: int = DEFAULT_NEWS_CUTOFF_HOUR,
    return_horizon: int = DEFAULT_RETURN_HORIZON,
    volatility_horizon: int = DEFAULT_VOLATILITY_HORIZON,
) -> pd.DataFrame:
    """Tạo feature tại t và hai nhãn tương lai, không dùng thông tin sau t."""
    if return_horizon < 1 or volatility_horizon < 1:
        raise ValueError("return_horizon và volatility_horizon phải dương.")
    data_dir = Path(data_dir)
    stock = _read_ohlcv(data_dir / f"{ticker}.csv")
    vnindex = _read_ohlcv(data_dir / "VNINDEX.csv")

    stock["return"] = np.log(stock["close"] / stock["close"].shift(1))
    stock["volume_change"] = stock["volume"] / stock["volume"].shift(1) - 1.0
    stock["high_low_range"] = (stock["high"] - stock["low"]) / stock["close"]
    stock["intraday_return"] = np.log(stock["close"] / stock["open"])
    stock["overnight_return"] = np.log(stock["open"] / stock["close"].shift(1))
    stock["historical_volatility_5d"] = stock["return"].rolling(5).apply(
        lambda values: float(np.sqrt(np.mean(np.square(values)))), raw=True
    )
    stock["historical_volatility_20d"] = stock["return"].rolling(20).apply(
        lambda values: float(np.sqrt(np.mean(np.square(values)))), raw=True
    )
    stock["momentum_5d"] = stock["return"].rolling(5).sum()
    stock["momentum_20d"] = stock["return"].rolling(20).sum()
    volume_mean = stock["volume"].rolling(20).mean()
    volume_std = stock["volume"].rolling(20).std(ddof=0)
    stock["volume_zscore_20d"] = (stock["volume"] - volume_mean) / volume_std.replace(0, np.nan)

    vnindex = vnindex.loc[:, ["time", "close"]].rename(columns={"close": "vnindex_close"})
    vnindex[VNINDEX_FEATURE] = np.log(vnindex["vnindex_close"] / vnindex["vnindex_close"].shift(1))

    frame = stock.merge(
        vnindex.loc[:, ["time", VNINDEX_FEATURE]],
        on="time",
        how="inner",
        validate="one_to_one",
    ).sort_values("time").reset_index(drop=True)

    frame["vnindex_momentum_5d"] = frame[VNINDEX_FEATURE].rolling(5).sum()
    frame["relative_return"] = frame["return"] - frame[VNINDEX_FEATURE]

    if include_news:
        resolved_news_path = Path(news_path) if news_path is not None else data_dir / "news.csv"
        news_daily = aggregate_news_features(
            resolved_news_path,
            ticker,
            frame["time"],
            cutoff_hour=news_cutoff_hour,
        )
        frame = frame.merge(news_daily, on="time", how="left", validate="one_to_one")
        frame.loc[:, NEWS_FEATURES] = frame.loc[:, NEWS_FEATURES].fillna(0.0)

    # Row t holds outcomes strictly after the final input observation at t.
    future_returns = pd.concat(
        [frame["return"].shift(-step) for step in range(1, return_horizon + 1)],
        axis=1,
    )
    frame["target_return"] = future_returns.sum(axis=1, min_count=return_horizon)
    future_squared_returns = pd.concat(
        [frame["return"].shift(-step).pow(2) for step in range(1, volatility_horizon + 1)],
        axis=1,
    )
    frame["target_volatility"] = np.sqrt(future_squared_returns.mean(axis=1))
    frame["target_return_date"] = frame["time"].shift(-return_horizon)
    frame["target_volatility_date"] = frame["time"].shift(-volatility_horizon)
    needed = [*BASE_FEATURES, VNINDEX_FEATURE]
    if include_news:
        needed.extend(NEWS_FEATURES)
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
    include_vnindex: bool | None = None,
    window: int = 30,
    split: TimeSplit = TimeSplit(),
    *,
    feature_set: str | None = None,
    target: str = "return",
) -> SequenceSplit:
    """Build [sample, lookback, feature] tensors without fitting on future data."""
    if window < 2:
        raise ValueError("window must be at least 2.")

    if target not in ("return", "volatility"):
        raise ValueError("target phải là 'return' hoặc 'volatility'.")
    if feature_set is None:
        # Giữ tương thích với lời gọi cũ trong khi API mới dùng tên tập feature.
        feature_names = tuple(BASE_FEATURES + ((VNINDEX_FEATURE,) if include_vnindex else ()))
    else:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set phải thuộc {tuple(FEATURE_SETS)}")
        feature_names = tuple(FEATURE_SETS[feature_set])
    missing_features = set(feature_names).difference(frame.columns)
    if missing_features:
        raise ValueError(f"Khung dữ liệu thiếu feature: {sorted(missing_features)}")
    target_column = f"target_{target}"
    target_date_column = f"target_{target}_date"
    frame = frame.dropna(subset=[*feature_names, target_column, target_date_column]).reset_index(drop=True)
    train_end_rows = frame.loc[
        frame[target_date_column] <= pd.Timestamp(split.train_end), "time"
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
        target_date = pd.Timestamp(input_end[target_date_column])
        label = _split_name(target_date, split)
        sequences[label].append(features[end_index - window + 1 : end_index + 1])
        targets[label].append(float(input_end[target_column]))
        dates[label].append(target_date.to_datetime64())

        if alignment is None:
            alignment = {
                "input_start_date": frame.iloc[end_index - window + 1]["time"].date().isoformat(),
                "input_end_date": input_end["time"].date().isoformat(),
                "target_date": target_date.date().isoformat(),
                "target": target,
                "target_value": float(input_end[target_column]),
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
        f"({dataset.alignment['target']}={dataset.alignment['target_value']:.6f})",
    )
