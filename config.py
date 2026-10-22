"""Cấu hình tập trung cho thí nghiệm dự báo tỷ suất sinh lợi."""

from dataclasses import dataclass


PROJECT_NAME = "NCKH 2027 - Dự báo return cổ phiếu ngân hàng bằng Selective SSM"
DATA_SOURCE = "KBS"
DATA_START = "2020-01-01"
# Mốc đóng băng dữ liệu của lần thực nghiệm chính. Không thay đổi mốc này khi tái lập kết quả.
DATA_END = "2026-10-12"
DEFAULT_WINDOW = 30
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
)
VNINDEX_FEATURE = "vnindex_return"


@dataclass(frozen=True)
class TimeSplit:
    """A chronological split; a sample belongs to the date it predicts."""

    train_end: str = "2023-12-31"
    validation_end: str = "2024-12-31"

