from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crt_system import CRTConfig, CRTPipeline, train_crt_model
from crt_system.data import load_ohlcv_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh CRT strategy pipeline")
    parser.add_argument("--input", required=True, help="CSV file with OHLCV data")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--setup-timeframe", default="1h")
    parser.add_argument("--execution-timeframe", default="5m")
    parser.add_argument("--symbol", default="MARKET")
    parser.add_argument("--target-mode", default="tp2", choices=["tp1", "tp2"])
    parser.add_argument("--save-model", default="", help="Optional path to save the trained model bundle")
    args = parser.parse_args()

    base_frame = load_ohlcv_csv(args.input, timestamp_column=args.timestamp_column)
    config = CRTConfig(setup_timeframe=args.setup_timeframe, execution_timeframe=args.execution_timeframe, target_mode=args.target_mode)
    pipeline = CRTPipeline(config)
    dataset = pipeline.build_dataset(base_frame, symbol=args.symbol)
    if dataset.empty:
        raise SystemExit("No CRT samples were generated from the supplied data")

    bundle = train_crt_model(dataset, target_column="target")
    print("Training metrics:")
    for key, value in bundle.metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) and value == value else f"{key}: {value}")

    if args.save_model:
        bundle.save(args.save_model)
        print(f"Saved model bundle to {args.save_model}")

    output_path = Path(args.input).with_suffix(".crt_dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
