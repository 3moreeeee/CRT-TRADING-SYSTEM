from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class CRTConfig:
    """System-wide configuration for the CRT strategy."""

    setup_timeframe: str = "1h"
    execution_timeframe: str = "5m"
    target_mode: str = "tp2"
    atr_period: int = 14
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    ema_trend_period: int = 200
    liquidity_lookback_bars: int = 20
    swing_lookback_bars: int = 10
    max_setup_lookahead_bars: int = 6
    reclaim_window_bars: int = 4
    post_signal_horizon_bars: int = 96
    min_range_atr: float = 0.75
    max_range_atr: float = 3.5
    min_sweep_atr: float = 0.12
    key_level_distance_atr: float = 0.35
    key_time_window_minutes: int = 30
    stop_buffer_atr: float = 0.05
    min_rr_tp1: float = 1.2
    min_rr_tp2: float = 2.0
    min_displacement_atr: float = 0.5
    min_confirmation_strength: float = 0.35
    min_reclaim_strength: float = 0.45
    max_spread_ratio: float = 0.15
    min_stop_atr: float = 0.08
    max_stop_atr: float = 5.0
    min_ob_quality: float = 0.35
    key_times: Tuple[int, ...] = (1, 5, 9, 13, 15, 18, 21)
    session_windows: Tuple[Tuple[str, int, int], ...] = (
        ("asia", 0, 8),
        ("london", 8, 13),
        ("new_york", 13, 17),
        ("late_session", 17, 24),
    )
    setup_execution_map: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "1M": ("1D",),
            "1D": ("1h",),
            "4h": ("15m",),
            "1h": ("5m", "1m"),
            "15m": ("1m",),
        }
    )


def default_timeframe_map() -> Dict[str, Tuple[str, ...]]:
    return CRTConfig().setup_execution_map
