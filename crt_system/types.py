from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

Direction = Literal["bullish", "bearish"]
TargetMode = Literal["tp1", "tp2"]


@dataclass(frozen=True)
class CRTSetup:
    setup_id: str
    symbol: str
    setup_timeframe: str
    execution_timeframe: str
    range_index: int
    range_time: pd.Timestamp
    signal_time: pd.Timestamp
    direction: Direction
    range_high: float
    range_low: float
    range_mid: float
    range_size: float
    range_body_size: float
    range_atr: float
    range_direction: int
    sweep_index: int
    sweep_time: pd.Timestamp
    sweep_high: float
    sweep_low: float
    sweep_close: float
    sweep_direction: int
    sweep_depth: float
    sweep_depth_atr: float
    sweep_type: str
    close_position_after_sweep: float
    reclaim_strength: float
    bars_to_reclaim: int
    key_time_flag: int
    session_name: str
    time_since_session_open_minutes: int
    amd_alignment: int
    near_key_level: int
    liquidity_distance_atr: float
    higher_tf_bias: int
    setup_ema_fast: float
    setup_ema_slow: float
    setup_ema_trend: float


@dataclass(frozen=True)
class ExecutionPlan:
    setup_id: str
    valid: bool
    invalid_reason: str
    entry_time: Optional[pd.Timestamp]
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    ob_high: float
    ob_low: float
    ob_mid: float
    ob_quality: float
    order_block_type: str
    structure_break_level: float
    displacement_strength: float
    confirmation_strength: float
    risk_distance: float
    rr_tp1: float
    rr_tp2: float
    spread_ratio: float
    spread_points: float
    execution_atr: float
    micro_bos_flag: int
    retrace_required: int


@dataclass(frozen=True)
class TradeLabel:
    setup_id: str
    target_mode: str
    label: int
    entry_filled: int
    tp1_hit: int
    tp2_hit: int
    stop_hit: int
    bars_to_exit: int
    exit_reason: str
    mfe_r: float
    mae_r: float
    realized_r: float
