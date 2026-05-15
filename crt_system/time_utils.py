from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

DEFAULT_KEY_TIMES = (1, 5, 9, 13, 15, 18, 21)
DEFAULT_SESSION_WINDOWS = (
    ("asia", 0, 8),
    ("london", 8, 13),
    ("new_york", 13, 17),
    ("late_session", 17, 24),
)


def session_for_timestamp(timestamp: pd.Timestamp, session_windows=DEFAULT_SESSION_WINDOWS) -> str:
    hour = timestamp.hour
    for session_name, start_hour, end_hour in session_windows:
        if start_hour <= hour < end_hour:
            return session_name
    return "late_session"


def minutes_since_session_open(timestamp: pd.Timestamp, session_windows=DEFAULT_SESSION_WINDOWS) -> int:
    hour = timestamp.hour
    minute = timestamp.minute
    for _, start_hour, end_hour in session_windows:
        if start_hour <= hour < end_hour:
            return max(0, (hour - start_hour) * 60 + minute)
    return hour * 60 + minute


def key_time_flag(timestamp: pd.Timestamp, key_times: Iterable[int] = DEFAULT_KEY_TIMES, window_minutes: int = 30) -> int:
    total_minutes = timestamp.hour * 60 + timestamp.minute
    for key_hour in key_times:
        key_minutes = key_hour * 60
        if abs(total_minutes - key_minutes) <= window_minutes:
            return 1
    return 0


def expected_amd_alignment(
    session_name: str,
    direction: str,
    sweep_direction: str,
    asian_range_atr: float,
    accumulation_threshold: float = 1.2,
) -> int:
    if asian_range_atr > accumulation_threshold:
        return 0
    if session_name not in {"london", "new_york"}:
        return 0
    if direction == "bullish" and sweep_direction == "down":
        return 1
    if direction == "bearish" and sweep_direction == "up":
        return 1
    return 0
