"""Small, explicit configuration for the bank-stock forecasting experiments."""

from dataclasses import dataclass


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

