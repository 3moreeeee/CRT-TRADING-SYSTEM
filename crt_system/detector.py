from __future__ import annotations

from dataclasses import asdict
from typing import List

import numpy as np
import pandas as pd

from .config import CRTConfig
from .indicators import atr, candle_components, ema, rolling_swing_levels
from .time_utils import expected_amd_alignment, key_time_flag, minutes_since_session_open, session_for_timestamp
from .types import CRTSetup


def _close_inside_range(close_value: float, range_low: float, range_high: float) -> bool:
    return range_low <= close_value <= range_high


def _detect_sweep(
    dataframe: pd.DataFrame,
    range_high: float,
    range_low: float,
    range_index: int,
    config: CRTConfig,
    candle_atr: float,
):
    start_index = range_index + 1
    end_index = min(len(dataframe), range_index + 1 + config.max_setup_lookahead_bars)
    for sweep_index in range(start_index, end_index):
        row = dataframe.iloc[sweep_index]
        low_sweep = row["low"] < range_low - config.min_sweep_atr * candle_atr
        high_sweep = row["high"] > range_high + config.min_sweep_atr * candle_atr
        if low_sweep and _close_inside_range(float(row["close"]), range_low, range_high):
            return "bullish", sweep_index, "wick"
        if high_sweep and _close_inside_range(float(row["close"]), range_low, range_high):
            return "bearish", sweep_index, "wick"
        if low_sweep:
            return "bullish", sweep_index, "body"
        if high_sweep:
            return "bearish", sweep_index, "body"
    return None


def _reclaim_bar(
    dataframe: pd.DataFrame,
    sweep_index: int,
    direction: str,
    range_high: float,
    range_low: float,
    config: CRTConfig,
):
    start_index = sweep_index if direction == "bullish" else sweep_index
    end_index = min(len(dataframe), sweep_index + 1 + config.reclaim_window_bars)
    for reclaim_index in range(start_index, end_index):
        row = dataframe.iloc[reclaim_index]
        close_value = float(row["close"])
        if _close_inside_range(close_value, range_low, range_high):
            return reclaim_index, row
    return None


def detect_crt_setups(dataframe: pd.DataFrame, symbol: str, setup_timeframe: str, execution_timeframe: str, config: CRTConfig | None = None) -> List[CRTSetup]:
    config = config or CRTConfig(setup_timeframe=setup_timeframe, execution_timeframe=execution_timeframe)
    market = dataframe.sort_index().copy()
    market["atr"] = atr(market, config.atr_period)
    components = candle_components(market)
    market = pd.concat([market, components], axis=1)
    market["ema_fast"] = ema(market["close"], config.ema_fast_period)
    market["ema_slow"] = ema(market["close"], config.ema_slow_period)
    market["ema_trend"] = ema(market["close"], config.ema_trend_period)
    swings = rolling_swing_levels(market, config.liquidity_lookback_bars)
    market = pd.concat([market, swings], axis=1)

    setups: List[CRTSetup] = []
    for range_index in range(config.liquidity_lookback_bars, len(market) - config.max_setup_lookahead_bars - 1):
        candle = market.iloc[range_index]
        candle_atr = float(candle["atr"])
        if np.isnan(candle_atr) or candle_atr <= 0:
            continue

        range_high = float(candle["high"])
        range_low = float(candle["low"])
        range_size = range_high - range_low
        if range_size <= 0:
            continue
        range_ratio = range_size / candle_atr
        if not (config.min_range_atr <= range_ratio <= config.max_range_atr):
            continue

        sweep = _detect_sweep(market, range_high, range_low, range_index, config, candle_atr)
        if sweep is None:
            continue

        direction, sweep_index, sweep_type = sweep
        reclaim = _reclaim_bar(market, sweep_index, direction, range_high, range_low, config)
        if reclaim is None:
            continue

        reclaim_index, reclaim_row = reclaim
        reclaim_close = float(reclaim_row["close"])
        close_position = (reclaim_close - range_low) / range_size if direction == "bullish" else (range_high - reclaim_close) / range_size
        close_position = float(np.clip(close_position, 0.0, 1.0))
        reclaim_strength = close_position if direction == "bullish" else close_position
        bars_to_reclaim = reclaim_index - sweep_index
        setup_time = market.index[range_index]
        sweep_time = market.index[sweep_index]
        session_name = session_for_timestamp(setup_time)
        key_flag = key_time_flag(setup_time, config.key_times, config.key_time_window_minutes)
        time_since_session_open = minutes_since_session_open(setup_time)
        asian_range = market[(market.index.hour < 8) & (market.index.date == setup_time.date())]
        asian_range_atr = float((asian_range["high"].max() - asian_range["low"].min()) / candle_atr) if not asian_range.empty else 0.0
        amd_alignment = expected_amd_alignment(session_name, direction, "down" if direction == "bullish" else "up", asian_range_atr)
        if pd.notna(candle["swing_high"]) and pd.notna(candle["swing_low"]):
            liquidity_distance = float(min(abs(range_high - float(candle["swing_high"])), abs(range_low - float(candle["swing_low"]))) / candle_atr)
            near_key_level = int(liquidity_distance <= config.key_level_distance_atr)
        else:
            liquidity_distance = float("nan")
            near_key_level = 0
        higher_tf_bias = 1 if float(candle["close"]) > float(candle["ema_fast"]) > float(candle["ema_slow"]) else -1 if float(candle["close"]) < float(candle["ema_fast"]) < float(candle["ema_slow"]) else 0
        setup = CRTSetup(
            setup_id=f"{symbol}_{setup_timeframe}_{setup_time.isoformat()}",
            symbol=symbol,
            setup_timeframe=setup_timeframe,
            execution_timeframe=execution_timeframe,
            range_index=range_index,
            range_time=setup_time,
            signal_time=market.index[reclaim_index],
            direction=direction,  # type: ignore[arg-type]
            range_high=range_high,
            range_low=range_low,
            range_mid=(range_high + range_low) / 2.0,
            range_size=range_size,
            range_body_size=float(candle["body"]),
            range_atr=candle_atr,
            range_direction=1 if float(candle["close"]) > float(candle["open"]) else -1 if float(candle["close"]) < float(candle["open"]) else 0,
            sweep_index=sweep_index,
            sweep_time=sweep_time,
            sweep_high=float(market.iloc[sweep_index]["high"]),
            sweep_low=float(market.iloc[sweep_index]["low"]),
            sweep_close=float(market.iloc[sweep_index]["close"]),
            sweep_direction=1 if direction == "bullish" else -1,
            sweep_depth=float((range_low - float(market.iloc[sweep_index]["low"])) if direction == "bullish" else (float(market.iloc[sweep_index]["high"]) - range_high)),
            sweep_depth_atr=float(abs((range_low - float(market.iloc[sweep_index]["low"])) if direction == "bullish" else (float(market.iloc[sweep_index]["high"]) - range_high)) / candle_atr),
            sweep_type=sweep_type,
            close_position_after_sweep=close_position,
            reclaim_strength=float(reclaim_strength),
            bars_to_reclaim=bars_to_reclaim,
            key_time_flag=key_flag,
            session_name=session_name,
            time_since_session_open_minutes=time_since_session_open,
            amd_alignment=amd_alignment,
            near_key_level=near_key_level,
            liquidity_distance_atr=liquidity_distance,
            higher_tf_bias=higher_tf_bias,
            setup_ema_fast=float(candle["ema_fast"]),
            setup_ema_slow=float(candle["ema_slow"]),
            setup_ema_trend=float(candle["ema_trend"]),
        )
        setups.append(setup)
    return setups
