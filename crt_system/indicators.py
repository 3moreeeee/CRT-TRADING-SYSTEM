from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(dataframe: pd.DataFrame) -> pd.Series:
    prev_close = dataframe["close"].shift(1)
    high_low = dataframe["high"] - dataframe["low"]
    high_prev_close = (dataframe["high"] - prev_close).abs()
    low_prev_close = (dataframe["low"] - prev_close).abs()
    return pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)


def atr(dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(dataframe).rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def candle_components(dataframe: pd.DataFrame) -> pd.DataFrame:
    body = (dataframe["close"] - dataframe["open"]).abs()
    candle_range = dataframe["high"] - dataframe["low"]
    upper_wick = dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
    lower_wick = dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
    return pd.DataFrame(
        {
            "body": body,
            "range": candle_range,
            "upper_wick": upper_wick.clip(lower=0.0),
            "lower_wick": lower_wick.clip(lower=0.0),
        },
        index=dataframe.index,
    )


def rolling_swing_levels(dataframe: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "swing_high": dataframe["high"].rolling(lookback, min_periods=lookback).max(),
            "swing_low": dataframe["low"].rolling(lookback, min_periods=lookback).min(),
        },
        index=dataframe.index,
    )


def normalize_series(series: pd.Series, scale: float) -> pd.Series:
    if scale == 0 or np.isnan(scale):
        return series * np.nan
    return series / scale
