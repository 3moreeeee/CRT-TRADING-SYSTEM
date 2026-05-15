from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from .config import CRTConfig
from .types import CRTSetup, ExecutionPlan


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    reasons: Tuple[str, ...]


def quality_filter(setup: CRTSetup, plan: ExecutionPlan, config: CRTConfig | None = None) -> FilterResult:
    config = config or CRTConfig(setup_timeframe=setup.setup_timeframe, execution_timeframe=setup.execution_timeframe)
    reasons: List[str] = []

    if setup.key_time_flag == 0:
        reasons.append("not_key_time")
    if setup.amd_alignment == 0:
        reasons.append("amd_misaligned")
    if setup.reclaim_strength < config.min_reclaim_strength:
        reasons.append("weak_reclaim")
    if setup.sweep_depth_atr < config.min_sweep_atr:
        reasons.append("weak_sweep")
    if setup.range_atr < config.min_range_atr:
        reasons.append("small_range")
    if plan.valid:
        if plan.spread_ratio > config.max_spread_ratio:
            reasons.append("spread_too_wide")
        if plan.confirmation_strength < config.min_confirmation_strength:
            reasons.append("weak_confirmation")
        if plan.displacement_strength < config.min_displacement_atr:
            reasons.append("weak_displacement")
        if plan.ob_quality < config.min_ob_quality:
            reasons.append("weak_order_block")
        if plan.risk_distance / setup.range_atr < config.min_stop_atr:
            reasons.append("stop_too_small")
        if plan.risk_distance / setup.range_atr > config.max_stop_atr:
            reasons.append("stop_too_large")
        if plan.rr_tp1 < config.min_rr_tp1:
            reasons.append("poor_rr_tp1")
        if plan.rr_tp2 < config.min_rr_tp2:
            reasons.append("poor_rr_tp2")
    else:
        reasons.append(plan.invalid_reason)

    return FilterResult(passed=len(reasons) == 0, reasons=tuple(reasons))
