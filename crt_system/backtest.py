from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import CRTConfig
from .data import build_timeframes, resample_ohlcv, validate_ohlcv
from .execution import build_execution_plan
from .features import build_feature_frame
from .labels import simulate_trade
from .types import CRTSetup, ExecutionPlan, TradeLabel
from .detector import detect_crt_setups


@dataclass
class CRTBacktestResult:
    summary: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame

    def save(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.trades.to_csv(output_path / "crt_trades.csv", index=False)
        self.equity_curve.to_csv(output_path / "crt_equity_curve.csv", index=False)
        pd.Series(self.summary).to_json(output_path / "crt_summary.json")


@dataclass(frozen=True)
class _SimulatedTrade:
    setup: CRTSetup
    plan: ExecutionPlan
    label: TradeLabel
    entry_time: Optional[pd.Timestamp]
    exit_time: Optional[pd.Timestamp]
    exit_reason: str
    realized_r: float
    equity_before: float
    equity_after: float


def _touches_level(row: pd.Series, level: float) -> bool:
    return float(row["low"]) <= level <= float(row["high"])


def _simulate_trade_with_timestamps(
    execution_frame: pd.DataFrame,
    setup: CRTSetup,
    plan: ExecutionPlan,
    config: CRTConfig,
) -> _SimulatedTrade:
    label = simulate_trade(execution_frame, setup, plan, config=config, target_mode=config.target_mode)
    if not plan.valid:
        return _SimulatedTrade(setup, plan, label, None, None, plan.invalid_reason, 0.0, 0.0, 0.0)

    market = execution_frame.loc[execution_frame.index >= setup.signal_time].copy()
    if market.empty:
        return _SimulatedTrade(setup, plan, label, None, None, "no_future_data", 0.0, 0.0, 0.0)

    entry_time = None
    exit_time = None
    entry_price = plan.entry_price
    filled = False
    for timestamp, row in market.iterrows():
        if not filled:
            if _touches_level(row, entry_price):
                entry_time = timestamp
                filled = True
            continue

        high = float(row["high"])
        low = float(row["low"])
        if setup.direction == "bullish":
            stop_touched = low <= plan.stop_loss
            tp1_touched = high >= plan.tp1
            tp2_touched = high >= plan.tp2
        else:
            stop_touched = high >= plan.stop_loss
            tp1_touched = low <= plan.tp1
            tp2_touched = low <= plan.tp2

        if stop_touched:
            exit_time = timestamp
            return _SimulatedTrade(setup, plan, label, entry_time, exit_time, "stop", -1.0, 0.0, 0.0)
        if config.target_mode == "tp1" and tp1_touched:
            exit_time = timestamp
            return _SimulatedTrade(setup, plan, label, entry_time, exit_time, "tp1", plan.rr_tp1, 0.0, 0.0)
        if config.target_mode == "tp2" and tp2_touched:
            exit_time = timestamp
            return _SimulatedTrade(setup, plan, label, entry_time, exit_time, "tp2", plan.rr_tp2, 0.0, 0.0)
        if config.target_mode == "tp1" and tp2_touched:
            exit_time = timestamp
            return _SimulatedTrade(setup, plan, label, entry_time, exit_time, "tp2", plan.rr_tp2, 0.0, 0.0)

    if filled and exit_time is None:
        exit_time = market.index[-1]
    return _SimulatedTrade(setup, plan, label, entry_time, exit_time, "timeout", 0.0, 0.0, 0.0)


def _max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    running_max = equity_series.cummax()
    drawdown = equity_series / running_max - 1.0
    return float(drawdown.min())


def run_crt_backtest(
    base_dataframe: pd.DataFrame,
    config: CRTConfig | None = None,
    symbol: str = "MARKET",
    initial_equity: float = 100000.0,
    risk_per_trade: float = 0.01,
    execution_dataframe: pd.DataFrame | None = None,
) -> CRTBacktestResult:
    config = config or CRTConfig()
    base_frame = validate_ohlcv(base_dataframe)
    setup_frame = base_frame.copy() if config.setup_timeframe.lower() == "1m" else resample_ohlcv(base_frame, config.setup_timeframe)

    if execution_dataframe is not None:
        execution_frame = validate_ohlcv(execution_dataframe)
    else:
        frames = build_timeframes(base_frame, config.setup_timeframe, config.execution_timeframe)
        setup_frame = frames.setup
        execution_frame = frames.execution

    setups = detect_crt_setups(setup_frame, symbol, config.setup_timeframe, config.execution_timeframe, config=config)
    plans = [build_execution_plan(setup, execution_frame, config) for setup in setups]

    trade_records: List[Dict[str, float]] = []
    equity_rows: List[Dict[str, float]] = []
    equity = initial_equity
    equity_rows.append({"time": execution_frame.index[0], "equity": equity})

    paired = sorted(zip(setups, plans), key=lambda item: item[0].signal_time)
    for setup, plan in paired:
        simulated = _simulate_trade_with_timestamps(execution_frame, setup, plan, config)
        if simulated.entry_time is None:
            continue
        risk_amount = equity * risk_per_trade
        pnl = risk_amount * simulated.realized_r
        equity_before = equity
        equity += pnl
        equity_rows.append({"time": simulated.exit_time or simulated.entry_time, "equity": equity})
        trade_records.append(
            {
                "setup_id": setup.setup_id,
                "symbol": setup.symbol,
                "direction": setup.direction,
                "signal_time": setup.signal_time,
                "entry_time": simulated.entry_time,
                "exit_time": simulated.exit_time,
                "exit_reason": simulated.exit_reason,
                "entry_price": plan.entry_price,
                "stop_loss": plan.stop_loss,
                "tp1": plan.tp1,
                "tp2": plan.tp2,
                "risk_distance": plan.risk_distance,
                "rr_tp1": plan.rr_tp1,
                "rr_tp2": plan.rr_tp2,
                "realized_r": simulated.realized_r,
                "risk_amount": risk_amount,
                "pnl": pnl,
                "equity_before": equity_before,
                "equity_after": equity,
                "valid_plan": int(plan.valid),
                "label": simulated.label.label,
                "tp1_hit": simulated.label.tp1_hit,
                "tp2_hit": simulated.label.tp2_hit,
                "stop_hit": simulated.label.stop_hit,
            }
        )

    trades = pd.DataFrame(trade_records)
    equity_curve = pd.DataFrame(equity_rows)
    if not equity_curve.empty:
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["peak"] - 1.0

    if trades.empty:
        summary = {
            "trade_count": 0,
            "win_rate": 0.0,
            "average_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "final_equity": float(initial_equity),
        }
        return CRTBacktestResult(summary=summary, trades=trades, equity_curve=equity_curve)

    wins = trades[trades["realized_r"] > 0]
    losses = trades[trades["realized_r"] < 0]
    gross_profit = float(wins["realized_r"].sum())
    gross_loss = abs(float(losses["realized_r"].sum()))
    summary = {
        "trade_count": float(len(trades)),
        "win_rate": float((trades["realized_r"] > 0).mean()),
        "average_r": float(trades["realized_r"].mean()),
        "expectancy_r": float(trades["realized_r"].mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": _max_drawdown(equity_curve["equity"]),
        "final_equity": float(equity_curve["equity"].iloc[-1]),
    }
    return CRTBacktestResult(summary=summary, trades=trades, equity_curve=equity_curve)
