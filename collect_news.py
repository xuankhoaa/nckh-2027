"""Thu thập tiêu đề tin ngân hàng từ Google News RSS và chấm sentiment minh bạch.

Script chỉ lưu tiêu đề, nguồn, URL và thời điểm xuất bản; không sao chép toàn văn
bài báo. Các khoảng thời gian bị chạm giới hạn kết quả sẽ tự được chia nhỏ để
giảm mất mẫu. Dữ liệu đầu ra tương thích trực tiếp với ``preprocess.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
import unicodedata
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from config import DATA_END, DATA_START, TICKERS, TimeSplit
from console import configure_utf8_output


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GOOGLE_RESULT_LIMIT = 100

# Dùng tên thương hiệu đầy đủ thay vì mã ba chữ cái để tránh ACB/MBB/CTG xuất
# hiện trong ngữ cảnh không liên quan. Mỗi truy vấn được lưu lại trong manifest.
BANK_QUERIES = {
    "VCB": '(Vietcombank OR "Ngân hàng Ngoại thương")',
    "BID": '(BIDV OR "Ngân hàng Đầu tư và Phát triển")',
    "CTG": '(VietinBank OR "Ngân hàng Công thương")',
    "MBB": '("MB Bank" OR MBBank OR "Ngân hàng Quân đội")',
    "TCB": "Techcombank",
    "ACB": '"Ngân hàng ACB"',
    "VPB": "VPBank",
    "STB": "Sacombank",
}

BANK_ALIASES = {
    "VCB": ("vietcombank", " vcb ", "ngân hàng ngoại thương"),
    "BID": ("bidv", "ngân hàng đầu tư và phát triển"),
    "CTG": ("vietinbank", " ctg ", "ngân hàng công thương"),
    "MBB": ("mb bank", "mbbank", " mbb ", "ngân hàng quân đội", "ngân hàng mb"),
    "TCB": ("techcombank", " tcb "),
    "ACB": (" acb ",),
    "VPB": ("vpbank", " vpb "),
    "STB": ("sacombank", " stb "),
}

# Từ điển nhỏ, có chủ đích và có thể kiểm toán. Tránh các từ đơn mơ hồ như
# "tăng" hoặc "giảm"; ưu tiên cụm từ mô tả trực tiếp tình hình ngân hàng.
POSITIVE_PHRASES = {
    "lợi nhuận tăng": 2.0,
    "lãi tăng": 2.0,
    "tăng trưởng lợi nhuận": 2.0,
    "lợi nhuận vượt": 2.0,
    "vượt kế hoạch": 1.5,
    "vượt kỳ vọng": 1.5,
    "lãi kỷ lục": 2.0,
    "lợi nhuận kỷ lục": 2.0,
    "kết quả tích cực": 1.5,
    "triển vọng tích cực": 1.5,
    "nâng hạng": 1.0,
    "tăng vốn thành công": 1.0,
    "chia cổ tức": 1.0,
    "trả cổ tức": 1.0,
    "được chấp thuận": 1.0,
    "được phê duyệt": 1.0,
    "hoàn thành kế hoạch": 1.5,
    "giảm nợ xấu": 2.0,
    "kiểm soát nợ xấu": 1.5,
    "chất lượng tài sản cải thiện": 2.0,
    "hợp tác chiến lược": 1.0,
    "tăng trưởng tín dụng": 1.0,
}

NEGATIVE_PHRASES = {
    "lợi nhuận giảm": 2.0,
    "lãi giảm": 2.0,
    "sụt giảm lợi nhuận": 2.0,
    "thua lỗ": 2.0,
    "báo lỗ": 2.0,
    "nợ xấu tăng": 2.0,
    "gia tăng nợ xấu": 2.0,
    "nợ xấu cao": 1.5,
    "nợ quá hạn": 1.5,
    "bị xử phạt": 2.0,
    "xử phạt": 1.5,
    "vi phạm": 1.5,
    "khởi tố": 2.0,
    "bắt tạm giam": 2.0,
    "bắt giữ": 2.0,
    "điều tra": 1.5,
    "cảnh báo": 1.0,
    "hạ hạng": 1.5,
    "rủi ro gia tăng": 1.5,
    "giảm mạnh": 1.5,
    "bán tháo": 1.5,
    "gian lận": 2.0,
    "đình chỉ": 1.5,
}

POSITIVE_WORDS = {
    "tích cực": 0.75,
    "kỷ lục": 0.75,
    "tăng trưởng": 0.5,
    "bứt phá": 0.75,
    "khởi sắc": 0.75,
    "vinh danh": 0.5,
    "giải thưởng": 0.5,
    "ưu đãi": 0.5,
    "miễn phí": 0.5,
    "thành công": 0.5,
    "an toàn": 0.5,
    "hiệu quả": 0.5,
    "dẫn đầu": 0.5,
    "ổn định": 0.5,
    "phục hồi": 0.5,
}

NEGATIVE_WORDS = {
    "tiêu cực": 0.75,
    "sụt giảm": 0.75,
    "thua lỗ": 0.75,
    "xử phạt": 0.75,
    "vi phạm": 0.75,
    "khởi tố": 1.0,
    "bắt giữ": 1.0,
    "điều tra": 0.75,
    "cảnh báo": 0.5,
    "rủi ro": 0.5,
    "gian lận": 1.0,
    "giả mạo": 0.75,
    "lừa đảo": 1.0,
    "mất tiền": 0.75,
    "đình chỉ": 0.75,
    "bán tháo": 0.75,
}


@dataclass(frozen=True)
class FetchStats:
    ticker: str
    request_count: int
    saturated_intervals: int
    article_count: int
    first_published_at: str | None
    last_published_at: str | None


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^0-9a-zà-ỹđ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_sentiment(title: str) -> float:
    """Điểm từ điển trong [-1, 1]; 0 nghĩa là không có tín hiệu rõ ràng."""
    text = f" {_normalise_text(title)} "
    positive = sum(weight for phrase, weight in POSITIVE_PHRASES.items() if f" {phrase} " in text)
    negative = sum(weight for phrase, weight in NEGATIVE_PHRASES.items() if f" {phrase} " in text)
    positive += sum(weight for phrase, weight in POSITIVE_WORDS.items() if f" {phrase} " in text)
    negative += sum(weight for phrase, weight in NEGATIVE_WORDS.items() if f" {phrase} " in text)
    if positive + negative == 0:
        return 0.0
    return float(round((positive - negative) / (positive + negative), 6))


def title_mentions_bank(title: str, ticker: str) -> bool:
    """Chỉ giữ bài có thực thể ngân hàng ngay trong tiêu đề được chấm điểm."""
    text = f" {_normalise_text(title)} "
    return any(alias in text for alias in BANK_ALIASES[ticker.upper()])


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(start.normalize().replace(day=1), end.normalize(), freq="MS")
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for month_start in starts:
        left = max(start.normalize(), month_start)
        right = min(end.normalize() + pd.Timedelta(days=1), month_start + pd.offsets.MonthBegin(1))
        if left < right:
            intervals.append((left, right))
    return intervals


def _parse_rss(content: bytes, ticker: str, query: str) -> list[dict[str, object]]:
    root = ET.fromstring(content)
    records: list[dict[str, object]] = []
    for item in root.findall(".//item"):
        source_node = item.find("source")
        source = source_node.text.strip() if source_node is not None and source_node.text else ""
        title = (item.findtext("title") or "").strip()
        suffix = f" - {source}"
        if source and title.endswith(suffix):
            title = title[: -len(suffix)].strip()
        published = pd.to_datetime(item.findtext("pubDate"), errors="coerce", utc=True)
        if pd.isna(published) or not title:
            continue
        records.append(
            {
                "published_at": published.tz_convert("Asia/Ho_Chi_Minh").isoformat(),
                "ticker": ticker,
                "title": title,
                "source": source,
                "source_url": source_node.attrib.get("url", "") if source_node is not None else "",
                "url": (item.findtext("link") or "").strip(),
                "query": query,
            }
        )
    return records


def _request_rss(
    session: requests.Session,
    ticker: str,
    left: pd.Timestamp,
    right: pd.Timestamp,
    *,
    timeout: float,
    retries: int,
) -> list[dict[str, object]]:
    # Mở rộng một ngày rồi lọc lại cục bộ để không mất bài tại biên do toán tử
    # after/before của công cụ tìm kiếm là bất đẳng thức nghiêm ngặt.
    after = (left - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    before = (right + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    query = f"{BANK_QUERIES[ticker]} after:{after} before:{before}"
    url = f"{GOOGLE_NEWS_RSS}?{urlencode({'q': query, 'hl': 'vi', 'gl': 'VN', 'ceid': 'VN:vi'})}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            records = _parse_rss(response.content, ticker, query)
            kept = []
            for record in records:
                published = pd.Timestamp(record["published_at"]).tz_localize(None)
                if left <= published.normalize() < right:
                    kept.append(record)
            return kept
        except (requests.RequestException, ET.ParseError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Không tải được RSS cho {ticker}, {left.date()}–{right.date()}: {last_error}")


def fetch_ticker_news(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    delay: float = 0.5,
    timeout: float = 30.0,
    retries: int = 2,
) -> tuple[pd.DataFrame, FetchStats]:
    """Tải theo tháng; tự chia đôi khoảng chạm trần 100 kết quả."""
    ticker = ticker.upper()
    if ticker not in BANK_QUERIES:
        raise ValueError(f"Chưa có truy vấn tin đáng tin cậy cho {ticker}.")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (NCKH university research; RSS metadata only)"})
    records: list[dict[str, object]] = []
    request_count = 0
    saturated_intervals = 0

    def collect(left: pd.Timestamp, right: pd.Timestamp) -> None:
        nonlocal request_count, saturated_intervals
        items = _request_rss(session, ticker, left, right, timeout=timeout, retries=retries)
        request_count += 1
        if len(items) >= GOOGLE_RESULT_LIMIT - 5 and (right - left).days > 2:
            saturated_intervals += 1
            middle = left + pd.Timedelta(days=max(1, (right - left).days // 2))
            time.sleep(delay)
            collect(left, middle)
            time.sleep(delay)
            collect(middle, right)
            return
        records.extend(items)

    for left, right in _month_starts(start, end):
        collect(left, right)
        time.sleep(delay)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        columns = ["published_at", "ticker", "title", "source", "source_url", "url", "query"]
        frame = pd.DataFrame(columns=columns)
    else:
        # Google có thể khớp từ khóa trong thân bài. Vì pipeline chỉ chấm điểm
        # tiêu đề, ta bắt buộc thực thể ngân hàng phải xuất hiện ngay ở tiêu đề.
        frame = frame.loc[
            [title_mentions_bank(title, ticker) for title in frame["title"]]
        ].copy()
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True).dt.tz_convert(
            "Asia/Ho_Chi_Minh"
        )
        frame["title_key"] = frame["title"].map(_normalise_text)
        frame["published_date"] = frame["published_at"].dt.date.astype(str)
        frame = (
            frame.sort_values(["published_at", "url"])
            .drop_duplicates(subset=["ticker", "url"], keep="first")
            .drop_duplicates(subset=["ticker", "published_date", "title_key"], keep="first")
            .drop(columns=["title_key", "published_date"])
            .reset_index(drop=True)
        )
        frame["published_at"] = frame["published_at"].map(lambda value: value.isoformat())
    dates = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    stats = FetchStats(
        ticker=ticker,
        request_count=request_count,
        saturated_intervals=saturated_intervals,
        article_count=len(frame),
        first_published_at=None if frame.empty else dates.min().isoformat(),
        last_published_at=None if frame.empty else dates.max().isoformat(),
    )
    return frame, stats


def validate_news_coverage(frame: pd.DataFrame, tickers: list[str], minimum_per_split: int = 20) -> pd.DataFrame:
    """Bắt buộc mỗi mã có dữ liệu trong train, validation và test."""
    split = TimeSplit()
    news = frame.copy()
    news["published_at"] = pd.to_datetime(news["published_at"], errors="raise", utc=True)
    local_date = (
        news["published_at"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None).dt.normalize()
    )
    news["split"] = pd.cut(
        local_date,
        bins=[pd.Timestamp.min, pd.Timestamp(split.train_end), pd.Timestamp(split.validation_end), pd.Timestamp.max],
        labels=["train", "validation", "test"],
    )
    coverage = (
        news.groupby(["ticker", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=tickers, columns=["train", "validation", "test"], fill_value=0)
        .astype(int)
    )
    insufficient = coverage.lt(minimum_per_split)
    if insufficient.any().any():
        details = []
        for row, column in zip(*insufficient.to_numpy().nonzero()):
            details.append(f"{coverage.index[row]}/{coverage.columns[column]}={coverage.iloc[row, column]}")
        raise ValueError(
            f"Dữ liệu tin chưa đạt tối thiểu {minimum_per_split} bài cho mỗi mã/mỗi tập: "
            + ", ".join(details)
        )
    return coverage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_validation_sample(frame: pd.DataFrame, path: Path, seed: int = 42) -> None:
    random.seed(seed)
    sample_parts = []
    for label, subset in frame.groupby("sentiment_label"):
        count = min(30, len(subset))
        if count:
            sample_parts.append(subset.sample(count, random_state=seed))
    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else frame.head(0)
    columns = ["published_at", "ticker", "title", "sentiment_score", "sentiment_label"]
    sample = sample.loc[:, columns].copy()
    sample["human_label"] = ""
    sample["review_note"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thu thập metadata tin ngân hàng từ Google News RSS.")
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS), choices=tuple(BANK_QUERIES))
    parser.add_argument("--start", default=DATA_START)
    parser.add_argument("--end", default=DATA_END)
    parser.add_argument("--output", default="data/news.csv")
    parser.add_argument("--raw-output", default="data/news_raw.csv")
    parser.add_argument("--manifest", default="data/NEWS_MANIFEST.json")
    parser.add_argument("--validation-sample", default="data/news_validation_sample.csv")
    parser.add_argument("--minimum-per-split", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start >= end:
        raise ValueError("--start phải nhỏ hơn --end.")
    frames: list[pd.DataFrame] = []
    stats: list[FetchStats] = []
    for ticker in args.tickers:
        frame, ticker_stats = fetch_ticker_news(ticker, start, end, delay=max(args.delay, 0.0))
        frames.append(frame)
        stats.append(ticker_stats)
        print(f"{ticker}: {ticker_stats.article_count} bài, {ticker_stats.request_count} yêu cầu RSS")

    raw = pd.concat(frames, ignore_index=True).sort_values(["ticker", "published_at"]).reset_index(drop=True)
    if raw.empty:
        raise ValueError("Không thu được tin nào; không tạo dữ liệu giả.")
    output = raw.copy()
    output["sentiment_score"] = output["title"].map(score_sentiment)
    output["sentiment_label"] = pd.cut(
        output["sentiment_score"], [-1.000001, -0.000001, 0.000001, 1.000001], labels=["negative", "neutral", "positive"]
    ).astype(str)
    output["sentiment_method"] = "vi_finance_lexicon_v1_title"
    output["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    coverage = validate_news_coverage(output, list(args.tickers), args.minimum_per_split)

    raw_path = Path(args.raw_output)
    output_path = Path(args.output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    _write_validation_sample(output, Path(args.validation_sample))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Google News RSS",
        "content_policy": "Chỉ lưu metadata, tiêu đề và liên kết; không lưu toàn văn.",
        "period": {"start": str(start.date()), "end": str(end.date())},
        "queries": {ticker: BANK_QUERIES[ticker] for ticker in args.tickers},
        "sentiment_method": "vi_finance_lexicon_v1_title",
        "minimum_articles_per_ticker_per_split": args.minimum_per_split,
        "coverage": coverage.reset_index().to_dict(orient="records"),
        "fetch_stats": [asdict(item) for item in stats],
        "files": {
            str(raw_path): {"rows": len(raw), "sha256": _sha256(raw_path)},
            str(output_path): {"rows": len(output), "sha256": _sha256(output_path)},
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nĐộ phủ theo tập:")
    print(coverage.to_string())
    print(f"\nĐã tạo {output_path}, {raw_path}, {manifest_path} và mẫu kiểm tra thủ công.")


if __name__ == "__main__":
    configure_utf8_output()
    main()
