"""Tiện ích giữ đầu ra dòng lệnh tiếng Việt hoạt động trên Windows."""

from __future__ import annotations

import sys


def configure_utf8_output() -> None:
    """Dùng UTF-8 cho stdout/stderr khi luồng cho phép cấu hình lại."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # Một số môi trường notebook hoặc luồng đã đóng không hỗ trợ reconfigure.
            continue
