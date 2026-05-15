from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

TIMEFRAME_ALIASES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
    "1M": "MS",
}

TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
    "1M": 43200,
}


@dataclass(frozen=True)
class MultiTimeframeFrames:
    base: pd.DataFrame
    setup: pd.DataFrame
    execution: pd.DataFrame


def normalize_timeframe(timeframe: str) -> str:
    cleaned = timeframe.strip()
    if cleaned.endswith("M") and cleaned[:-1].isdigit():
        return cleaned
    return cleaned.lower()


def timeframe_to_minutes(timeframe: str) -> int:
    normalized = normalize_timeframe(timeframe)
    if normalized not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return TIMEFRAME_MINUTES[normalized]


def validate_ohlcv(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"open", "high", "low", "close"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise ValueError("OHLCV dataframe must use a DatetimeIndex")
    ordered = dataframe.sort_index().copy()
    if "volume" not in ordered.columns:
        ordered["volume"] = 0.0
    return ordered


def _infer_base_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        raise ValueError("At least three bars are required to infer base timeframe")
    deltas = index.to_series().diff().dropna()
    median_delta = deltas.median()
    return max(1, int(round(median_delta.total_seconds() / 60.0)))


def load_ohlcv_csv(path: str, timestamp_column: str = "timestamp", timezone: str | None = "UTC") -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    if timestamp_column not in dataframe.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in {path}")
    dataframe[timestamp_column] = pd.to_datetime(dataframe[timestamp_column], utc=True if timezone else False)
    dataframe = dataframe.set_index(timestamp_column)
    return validate_ohlcv(dataframe)


def resample_ohlcv(dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    validated = validate_ohlcv(dataframe)
    normalized = normalize_timeframe(timeframe)
    if normalized not in TIMEFRAME_ALIASES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    aggregated = validated.resample(TIMEFRAME_ALIASES[normalized], label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return aggregated.dropna(subset=["open", "high", "low", "close"])


def build_timeframes(base_dataframe: pd.DataFrame, setup_timeframe: str, execution_timeframe: str) -> MultiTimeframeFrames:
    base = validate_ohlcv(base_dataframe)
    setup_tf = normalize_timeframe(setup_timeframe)
    execution_tf = normalize_timeframe(execution_timeframe)
    base_minutes = _infer_base_minutes(base.index)
    if base_minutes > timeframe_to_minutes(execution_tf):
        raise ValueError("Base input is too coarse for the requested execution timeframe")
    if base_minutes > timeframe_to_minutes(setup_tf):
        raise ValueError("Base input is too coarse for the requested setup timeframe")
    setup = base.copy() if timeframe_to_minutes(setup_tf) == base_minutes else resample_ohlcv(base, setup_tf)
    execution = base.copy() if timeframe_to_minutes(execution_tf) == base_minutes else resample_ohlcv(base, execution_tf)
    return MultiTimeframeFrames(base=base, setup=setup, execution=execution)
