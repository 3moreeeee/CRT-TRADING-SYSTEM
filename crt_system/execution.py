from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import CRTConfig
from .indicators import atr, candle_components
from .types import CRTSetup, ExecutionPlan


def _first_order_block(dataframe: pd.DataFrame, start_index: int, direction: str, lookback: int = 8):
    if direction == "bullish":
        for index in range(start_index - 1, max(-1, start_index - lookback - 1), -1):
            row = dataframe.iloc[index]
            if float(row["close"]) < float(row["open"]):
                return index, row
    else:
        for index in range(start_index - 1, max(-1, start_index - lookback - 1), -1):
            row = dataframe.iloc[index]
            if float(row["close"]) > float(row["open"]):
                return index, row
    return None


def build_execution_plan(setup: CRTSetup, execution_frame: pd.DataFrame, config: CRTConfig | None = None) -> ExecutionPlan:
    config = config or CRTConfig(setup_timeframe=setup.setup_timeframe, execution_timeframe=setup.execution_timeframe)
    market = execution_frame.loc[execution_frame.index >= setup.signal_time].copy()
    if market.empty:
        return ExecutionPlan(
            setup_id=setup.setup_id,
            valid=False,
            invalid_reason="no_future_execution_data",
            entry_time=None,
            entry_price=np.nan,
            stop_loss=np.nan,
            tp1=np.nan,
            tp2=np.nan,
            ob_high=np.nan,
            ob_low=np.nan,
            ob_mid=np.nan,
            ob_quality=0.0,
            order_block_type=setup.direction,
            structure_break_level=np.nan,
            displacement_strength=0.0,
            confirmation_strength=0.0,
            risk_distance=np.nan,
            rr_tp1=np.nan,
            rr_tp2=np.nan,
            spread_ratio=np.nan,
            spread_points=np.nan,
            execution_atr=np.nan,
            micro_bos_flag=0,
            retrace_required=1,
        )

    market["atr"] = atr(market, config.atr_period)
    components = candle_components(market)
    market = pd.concat([market, components], axis=1)
    market["rolling_high"] = market["high"].shift(1).rolling(5, min_periods=5).max()
    market["rolling_low"] = market["low"].shift(1).rolling(5, min_periods=5).min()

    direction = setup.direction
    displacement_index = None
    structure_break_level = np.nan
    displacement_strength = 0.0
    confirmation_strength = 0.0
    order_block_index = None
    order_block_row = None
    micro_bos_flag = 0

    for index in range(5, len(market)):
        row = market.iloc[index]
        row_atr = float(row["atr"])
        if np.isnan(row_atr) or row_atr <= 0:
            continue
        body = float(row["body"])
        candle_range = float(row["range"])
        strong_displacement = body >= config.min_displacement_atr * row_atr and candle_range >= body * 1.3
        if direction == "bullish":
            bos = pd.notna(row["rolling_high"]) and float(row["close"]) > float(row["rolling_high"]) + 0.05 * row_atr
            if row["close"] > row["open"] and strong_displacement and bos:
                displacement_index = index
                structure_break_level = float(row["rolling_high"])
                displacement_strength = body / row_atr
                confirmation_strength = min(1.0, float(row["close"] - row["open"]) / candle_range)
                micro_bos_flag = 1
                break
        else:
            bos = pd.notna(row["rolling_low"]) and float(row["close"]) < float(row["rolling_low"]) - 0.05 * row_atr
            if row["close"] < row["open"] and strong_displacement and bos:
                displacement_index = index
                structure_break_level = float(row["rolling_low"])
                displacement_strength = body / row_atr
                confirmation_strength = min(1.0, float(row["open"] - row["close"]) / candle_range)
                micro_bos_flag = 1
                break

    if displacement_index is None:
        return ExecutionPlan(
            setup_id=setup.setup_id,
            valid=False,
            invalid_reason="no_micro_bos_or_displacement",
            entry_time=None,
            entry_price=np.nan,
            stop_loss=np.nan,
            tp1=np.nan,
            tp2=np.nan,
            ob_high=np.nan,
            ob_low=np.nan,
            ob_mid=np.nan,
            ob_quality=0.0,
            order_block_type=direction,
            structure_break_level=np.nan,
            displacement_strength=0.0,
            confirmation_strength=0.0,
            risk_distance=np.nan,
            rr_tp1=np.nan,
            rr_tp2=np.nan,
            spread_ratio=np.nan,
            spread_points=np.nan,
            execution_atr=float(market["atr"].iloc[min(len(market) - 1, 5)]),
            micro_bos_flag=0,
            retrace_required=1,
        )

    order_block = _first_order_block(market, displacement_index, direction)
    if order_block is None:
        return ExecutionPlan(
            setup_id=setup.setup_id,
            valid=False,
            invalid_reason="no_order_block_found",
            entry_time=None,
            entry_price=np.nan,
            stop_loss=np.nan,
            tp1=np.nan,
            tp2=np.nan,
            ob_high=np.nan,
            ob_low=np.nan,
            ob_mid=np.nan,
            ob_quality=0.0,
            order_block_type=direction,
            structure_break_level=structure_break_level,
            displacement_strength=displacement_strength,
            confirmation_strength=confirmation_strength,
            risk_distance=np.nan,
            rr_tp1=np.nan,
            rr_tp2=np.nan,
            spread_ratio=np.nan,
            spread_points=np.nan,
            execution_atr=float(market["atr"].iloc[displacement_index]),
            micro_bos_flag=micro_bos_flag,
            retrace_required=1,
        )

    ob_index, ob_row = order_block
    ob_high = float(ob_row["high"])
    ob_low = float(ob_row["low"])
    ob_mid = (ob_high + ob_low) / 2.0
    ob_quality = min(1.0, abs(float(ob_row["close"]) - float(ob_row["open"])) / float(ob_row["range"]) if float(ob_row["range"]) else 0.0)
    execution_atr = float(market["atr"].iloc[displacement_index])
    buffer = config.stop_buffer_atr * execution_atr

    if direction == "bullish":
        entry_price = ob_high
        stop_loss = ob_low - buffer
        tp1 = setup.range_mid
        tp2 = setup.range_high
        order_block_type = "bullish"
    else:
        entry_price = ob_low
        stop_loss = ob_high + buffer
        tp1 = setup.range_mid
        tp2 = setup.range_low
        order_block_type = "bearish"

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        return ExecutionPlan(
            setup_id=setup.setup_id,
            valid=False,
            invalid_reason="invalid_risk_distance",
            entry_time=None,
            entry_price=np.nan,
            stop_loss=np.nan,
            tp1=np.nan,
            tp2=np.nan,
            ob_high=np.nan,
            ob_low=np.nan,
            ob_mid=np.nan,
            ob_quality=ob_quality,
            order_block_type=order_block_type,
            structure_break_level=structure_break_level,
            displacement_strength=displacement_strength,
            confirmation_strength=confirmation_strength,
            risk_distance=np.nan,
            rr_tp1=np.nan,
            rr_tp2=np.nan,
            spread_ratio=np.nan,
            spread_points=np.nan,
            execution_atr=execution_atr,
            micro_bos_flag=micro_bos_flag,
            retrace_required=1,
        )

    rr_tp1 = abs(tp1 - entry_price) / risk_distance
    rr_tp2 = abs(tp2 - entry_price) / risk_distance
    spread_points = 0.0
    spread_ratio = 0.0

    valid = True
    invalid_reason = ""
    if ob_quality < config.min_ob_quality:
        valid = False
        invalid_reason = "weak_ob_quality"
    if rr_tp1 < config.min_rr_tp1 or rr_tp2 < config.min_rr_tp2:
        valid = False
        invalid_reason = invalid_reason or "poor_reward_risk"

    return ExecutionPlan(
        setup_id=setup.setup_id,
        valid=valid,
        invalid_reason=invalid_reason,
        entry_time=market.index[ob_index],
        entry_price=float(entry_price),
        stop_loss=float(stop_loss),
        tp1=float(tp1),
        tp2=float(tp2),
        ob_high=ob_high,
        ob_low=ob_low,
        ob_mid=float(ob_mid),
        ob_quality=float(ob_quality),
        order_block_type=order_block_type,
        structure_break_level=float(structure_break_level),
        displacement_strength=float(displacement_strength),
        confirmation_strength=float(confirmation_strength),
        risk_distance=float(risk_distance),
        rr_tp1=float(rr_tp1),
        rr_tp2=float(rr_tp2),
        spread_ratio=float(spread_ratio),
        spread_points=float(spread_points),
        execution_atr=float(execution_atr),
        micro_bos_flag=micro_bos_flag,
        retrace_required=1,
    )
