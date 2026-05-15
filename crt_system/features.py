from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import CRTConfig
from .indicators import atr, candle_components, ema, rolling_swing_levels
from .types import CRTSetup, ExecutionPlan

SESSION_NAMES = ("asia", "london", "new_york", "late_session")


def _series_or_nan(value: float) -> float:
    return float(value) if value is not None and np.isfinite(value) else np.nan


def _volatility_regime(setup_row: pd.Series, dataframe: pd.DataFrame, config: CRTConfig) -> int:
    atr_series = dataframe["atr"].rolling(50, min_periods=20).median()
    if pd.isna(setup_row["atr"]) or pd.isna(atr_series.loc[setup_row.name]):
        return 0
    ratio = float(setup_row["atr"] / atr_series.loc[setup_row.name])
    if ratio < 0.85:
        return -1
    if ratio > 1.25:
        return 1
    return 0


def build_feature_row(
    setup: CRTSetup,
    execution_plan: ExecutionPlan,
    setup_frame: pd.DataFrame,
    execution_frame: pd.DataFrame,
    config: CRTConfig | None = None,
) -> Dict[str, float]:
    config = config or CRTConfig(setup_timeframe=setup.setup_timeframe, execution_timeframe=setup.execution_timeframe)
    setup_row = setup_frame.iloc[setup.range_index]
    execution_atr_series = atr(execution_frame, config.atr_period)
    setup_ema_fast = ema(setup_frame["close"], config.ema_fast_period)
    setup_ema_slow = ema(setup_frame["close"], config.ema_slow_period)
    setup_ema_trend = ema(setup_frame["close"], config.ema_trend_period)
    swings = rolling_swing_levels(setup_frame, config.swing_lookback_bars)
    candle = candle_components(setup_frame.iloc[[setup.range_index]]).iloc[0]

    feature_row: Dict[str, float] = {
        "range_high": setup.range_high,
        "range_low": setup.range_low,
        "range_midpoint": setup.range_mid,
        "range_size": setup.range_size,
        "range_size_atr": setup.range_size / setup.range_atr if setup.range_atr else np.nan,
        "range_body_size": setup.range_body_size,
        "range_upper_wick": float(candle["upper_wick"]),
        "range_lower_wick": float(candle["lower_wick"]),
        "body_to_range_ratio": setup.range_body_size / setup.range_size if setup.range_size else np.nan,
        "range_direction": float(setup.range_direction),
        "sweep_direction": float(setup.sweep_direction),
        "sweep_depth": setup.sweep_depth,
        "sweep_depth_atr": setup.sweep_depth_atr,
        "close_position_after_sweep": setup.close_position_after_sweep,
        "reclaim_strength": setup.reclaim_strength,
        "bars_to_reclaim": float(setup.bars_to_reclaim),
        "wick_only_sweep": 1.0 if setup.sweep_type == "wick" else 0.0,
        "distance_to_midpoint": abs(execution_plan.tp1 - execution_plan.entry_price),
        "distance_to_full_target": abs(execution_plan.tp2 - execution_plan.entry_price),
        "expected_rr_tp1": execution_plan.rr_tp1,
        "expected_rr_tp2": execution_plan.rr_tp2,
        "hour_of_day": float(setup.range_time.hour),
        "minute_of_hour": float(setup.range_time.minute),
        "key_time_flag": float(setup.key_time_flag),
        "time_since_session_open_minutes": float(setup.time_since_session_open_minutes),
        "amd_alignment": float(setup.amd_alignment),
        "session_asia": 1.0 if setup.session_name == "asia" else 0.0,
        "session_london": 1.0 if setup.session_name == "london" else 0.0,
        "session_new_york": 1.0 if setup.session_name == "new_york" else 0.0,
        "session_late_session": 1.0 if setup.session_name == "late_session" else 0.0,
        "order_block_position": _order_block_position(execution_plan),
        "entry_to_ob_distance": abs(execution_plan.entry_price - execution_plan.ob_mid),
        "entry_to_ob_mid_distance": abs(execution_plan.entry_price - execution_plan.ob_mid),
        "confirmation_candle_strength": execution_plan.confirmation_strength,
        "lower_tf_displacement": execution_plan.displacement_strength,
        "micro_bos_flag": float(execution_plan.micro_bos_flag),
        "atr": setup.range_atr,
        "execution_atr": float(execution_atr_series.iloc[-1]) if len(execution_atr_series) else np.nan,
        "spread_ratio": execution_plan.spread_ratio,
        "spread_points": execution_plan.spread_points,
        "stop_loss_distance": execution_plan.risk_distance,
        "volatility_regime": float(_volatility_regime(setup_row, setup_frame.assign(atr=atr(setup_frame, config.atr_period)), config)),
        "impulse_strength": execution_plan.displacement_strength / execution_plan.risk_distance if execution_plan.risk_distance else np.nan,
        "higher_tf_bias": float(setup.higher_tf_bias),
        "trend_alignment": float(1 if (setup.direction == "bullish" and setup.higher_tf_bias >= 0) or (setup.direction == "bearish" and setup.higher_tf_bias <= 0) else 0),
        "distance_to_nearest_liquidity_atr": float(setup.liquidity_distance_atr),
        "near_key_level_flag": float(setup.near_key_level),
        "setup_ema_fast": float(setup.setup_ema_fast),
        "setup_ema_slow": float(setup.setup_ema_slow),
        "setup_ema_trend": float(setup.setup_ema_trend),
        "setup_close_vs_fast_ema": float(setup_row["close"] - setup_ema_fast.iloc[setup.range_index]),
        "setup_close_vs_slow_ema": float(setup_row["close"] - setup_ema_slow.iloc[setup.range_index]),
        "setup_close_vs_trend_ema": float(setup_row["close"] - setup_ema_trend.iloc[setup.range_index]),
        "swing_high_distance": float(setup_row["high"] - swings["swing_high"].iloc[setup.range_index]) if pd.notna(swings["swing_high"].iloc[setup.range_index]) else np.nan,
        "swing_low_distance": float(setup_row["low"] - swings["swing_low"].iloc[setup.range_index]) if pd.notna(swings["swing_low"].iloc[setup.range_index]) else np.nan,
        "direction": 1.0 if setup.direction == "bullish" else 0.0,
    }
    return feature_row


def _order_block_position(execution_plan: ExecutionPlan) -> float:
    ob_size = execution_plan.ob_high - execution_plan.ob_low
    if ob_size <= 0:
        return np.nan
    if execution_plan.order_block_type == "bullish":
        return (execution_plan.entry_price - execution_plan.ob_low) / ob_size
    return (execution_plan.ob_high - execution_plan.entry_price) / ob_size


def build_feature_frame(
    setups: List[CRTSetup],
    execution_plans: List[ExecutionPlan],
    setup_frame: pd.DataFrame,
    execution_frame: pd.DataFrame,
    config: CRTConfig | None = None,
) -> pd.DataFrame:
    rows = []
    for setup, plan in zip(setups, execution_plans):
        if not plan.valid:
            continue
        row = build_feature_row(setup, plan, setup_frame, execution_frame, config=config)
        row["setup_id"] = setup.setup_id
        row["direction_label"] = 1 if setup.direction == "bullish" else 0
        rows.append(row)
    return pd.DataFrame(rows)
