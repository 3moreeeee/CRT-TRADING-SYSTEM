from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CRTConfig
from .types import CRTSetup, ExecutionPlan, TradeLabel


def _touches_trade_level(row: pd.Series, level: float) -> bool:
    return float(row["low"]) <= level <= float(row["high"])


def simulate_trade(
    execution_frame: pd.DataFrame,
    setup: CRTSetup,
    plan: ExecutionPlan,
    config: CRTConfig | None = None,
    target_mode: str = "tp2",
) -> TradeLabel:
    config = config or CRTConfig(setup_timeframe=setup.setup_timeframe, execution_timeframe=setup.execution_timeframe)
    if not plan.valid:
        return TradeLabel(
            setup_id=setup.setup_id,
            target_mode=target_mode,
            label=0,
            entry_filled=0,
            tp1_hit=0,
            tp2_hit=0,
            stop_hit=0,
            bars_to_exit=0,
            exit_reason=plan.invalid_reason,
            mfe_r=0.0,
            mae_r=0.0,
            realized_r=0.0,
        )

    market = execution_frame.loc[execution_frame.index >= setup.signal_time].copy()
    if market.empty:
        return TradeLabel(setup_id=setup.setup_id, target_mode=target_mode, label=0, entry_filled=0, tp1_hit=0, tp2_hit=0, stop_hit=0, bars_to_exit=0, exit_reason="no_future_data", mfe_r=0.0, mae_r=0.0, realized_r=0.0)

    entry_filled = False
    entry_index = None
    entry_price = plan.entry_price
    risk = plan.risk_distance
    tp1_hit = False
    tp2_hit = False
    stop_hit = False
    mfe = 0.0
    mae = 0.0

    for idx, (_, row) in enumerate(market.iterrows()):
        high = float(row["high"])
        low = float(row["low"])
        if not entry_filled:
            if _touches_trade_level(row, entry_price):
                entry_filled = True
                entry_index = idx
                continue
            continue

        if setup.direction == "bullish":
            adverse = max(0.0, entry_price - low)
            favorable = max(0.0, high - entry_price)
            stop_touched = low <= plan.stop_loss
            tp1_touched = high >= plan.tp1
            tp2_touched = high >= plan.tp2
        else:
            adverse = max(0.0, high - entry_price)
            favorable = max(0.0, entry_price - low)
            stop_touched = high >= plan.stop_loss
            tp1_touched = low <= plan.tp1
            tp2_touched = low <= plan.tp2

        mfe = max(mfe, favorable / risk if risk else 0.0)
        mae = max(mae, adverse / risk if risk else 0.0)

        if stop_touched and (tp1_touched or tp2_touched):
            stop_hit = True
            break
        if stop_touched:
            stop_hit = True
            break
        if tp1_touched:
            tp1_hit = True
            if target_mode == "tp1":
                tp2_hit = False
                break
        if tp2_touched:
            tp2_hit = True
            break

    if not entry_filled:
        return TradeLabel(setup_id=setup.setup_id, target_mode=target_mode, label=0, entry_filled=0, tp1_hit=0, tp2_hit=0, stop_hit=0, bars_to_exit=0, exit_reason="not_filled", mfe_r=0.0, mae_r=0.0, realized_r=0.0)

    if target_mode == "tp1":
        label = 1 if tp1_hit and not stop_hit else 0
        exit_reason = "tp1" if label else "stop_or_timeout"
    else:
        label = 1 if tp2_hit and not stop_hit else 0
        exit_reason = "tp2" if label else "stop_or_timeout"

    realized_r = 0.0
    if label == 1:
        realized_r = plan.rr_tp1 if target_mode == "tp1" else plan.rr_tp2
    elif stop_hit:
        realized_r = -1.0

    return TradeLabel(
        setup_id=setup.setup_id,
        target_mode=target_mode,
        label=label,
        entry_filled=1,
        tp1_hit=1 if tp1_hit else 0,
        tp2_hit=1 if tp2_hit else 0,
        stop_hit=1 if stop_hit else 0,
        bars_to_exit=int(len(market)),
        exit_reason=exit_reason,
        mfe_r=float(mfe),
        mae_r=float(mae),
        realized_r=float(realized_r),
    )


def build_label_frame(
    setups: List[CRTSetup],
    execution_plans: List[ExecutionPlan],
    execution_frame: pd.DataFrame,
    config: CRTConfig | None = None,
    target_mode: str = "tp2",
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for setup, plan in zip(setups, execution_plans):
        label = simulate_trade(execution_frame, setup, plan, config=config, target_mode=target_mode)
        row = asdict(label)
        row["direction_label"] = 1 if setup.direction == "bullish" else 0
        rows.append(row)
    return pd.DataFrame(rows)
