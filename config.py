"""Cấu hình tập trung cho hai bài toán dự báo của đề tài."""

from dataclasses import dataclass


PROJECT_NAME = "NCKH 2027 - Mamba đa biến cho return và volatility cổ phiếu ngân hàng"
DATA_SOURCE = "KBS"
DATA_START = "2020-01-01"
# Mốc đóng băng dữ liệu của lần thực nghiệm chính. Không thay đổi mốc này khi tái lập kết quả.
DATA_END = "2026-10-12"
DEFAULT_WINDOW = 30
DEFAULT_VOLATILITY_HORIZON = 5
DEFAULT_NEWS_CUTOFF_HOUR = 15
RESEARCH_SEEDS = (42, 52, 62, 72, 82)


TICKERS = ("VCB", "BID", "CTG", "MBB", "TCB", "ACB", "VPB", "STB")
BASE_FEATURES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return",
    "volume_change",
    "high_low_range",
    "historical_volatility_5d",
)
VNINDEX_FEATURE = "vnindex_return"
NEWS_FEATURES = ("sentiment_score", "news_count")

TARGETS = ("return", "volatility")
FEATURE_SETS = {
    "market_only": (*BASE_FEATURES, VNINDEX_FEATURE),
    "market_news": (*BASE_FEATURES, VNINDEX_FEATURE, *NEWS_FEATURES),
    "market_news_no_sentiment": (*BASE_FEATURES, VNINDEX_FEATURE, "news_count"),
    "market_news_no_count": (*BASE_FEATURES, VNINDEX_FEATURE, "sentiment_score"),
}
MAIN_FEATURE_SETS = ("market_only", "market_news")
ABLATION_FEATURE_SETS = ("market_news_no_sentiment", "market_news_no_count")


@dataclass(frozen=True)
class TimeSplit:
    """A chronological split; a sample belongs to the date it predicts."""

    train_end: str = "2023-12-31"
    validation_end: str = "2024-12-31"

