from __future__ import annotations

import argparse
from pathlib import Path

from crt_system import CRTConfig, run_crt_backtest
from crt_system.data import load_ohlcv_csv


def resolve_input_path(raw_input: str) -> Path:
    candidate = Path(raw_input)
    if candidate.exists():
        return candidate

    search_roots = [Path.cwd(), Path.cwd() / "data_crt_h1", Path.cwd() / "data_crt", Path.cwd() / "data"]

    for root in search_roots:
        direct_match = root / raw_input
        if direct_match.exists():
            return direct_match

    for root in search_roots:
        if root.exists():
            matches = list(root.rglob(candidate.name))
            if matches:
                return matches[0]

    raise FileNotFoundError(f"Could not find input file: {raw_input}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay/backtest the CRT strategy on historical OHLCV data")
    parser.add_argument("--input", required=True, help="CSV file with OHLCV data")
    parser.add_argument("--execution-input", default="", help="Optional lower-timeframe CSV for execution confirmation")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--setup-timeframe", default="1h")
    parser.add_argument("--execution-timeframe", default="5m")
    parser.add_argument("--symbol", default="MARKET")
    parser.add_argument("--target-mode", default="tp2", choices=["tp1", "tp2"])
    parser.add_argument("--initial-equity", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--output-dir", default="crt_backtest_output")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    base_frame = load_ohlcv_csv(str(input_path), timestamp_column=args.timestamp_column)
    execution_frame = None
    if args.execution_input:
        execution_path = resolve_input_path(args.execution_input)
        execution_frame = load_ohlcv_csv(str(execution_path), timestamp_column=args.timestamp_column)
    config = CRTConfig(setup_timeframe=args.setup_timeframe, execution_timeframe=args.execution_timeframe, target_mode=args.target_mode)
    result = run_crt_backtest(
        base_frame,
        config=config,
        symbol=args.symbol,
        initial_equity=args.initial_equity,
        risk_per_trade=args.risk_per_trade,
        execution_dataframe=execution_frame,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(str(output_dir))

    print("Backtest summary:")
    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print(f"Trades saved to {output_dir / 'crt_trades.csv'}")
    print(f"Equity curve saved to {output_dir / 'crt_equity_curve.csv'}")
    print(f"Summary saved to {output_dir / 'crt_summary.json'}")


if __name__ == "__main__":
    main()
