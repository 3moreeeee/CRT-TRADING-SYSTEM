from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import List

import MetaTrader5 as mt5
import pandas as pd


DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "XAUUSD", "BTCUSD", "ETHUSD", "NZDUSD"]

TIMEFRAME_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
}


def parse_symbols(symbols_value: str) -> List[str]:
    return [symbol.strip() for symbol in symbols_value.split(",") if symbol.strip()]


def load_mt5_credentials() -> tuple[int, str, str]:
    try:
        from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    except ImportError as exc:
        raise RuntimeError("config.py with MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER is required") from exc
    return MT5_LOGIN, MT5_PASSWORD, MT5_SERVER


def download_symbol_rates(symbol: str, timeframe, date_from: datetime | None, date_to: datetime | None) -> pd.DataFrame:
    if date_from is None or date_to is None:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10_000_000)
    else:
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned for {symbol} | error: {mt5.last_error()}")

    dataframe = pd.DataFrame(rates)
    dataframe["timestamp"] = pd.to_datetime(dataframe["time"], unit="s", utc=True).dt.tz_convert(None)
    dataframe = dataframe.drop(columns=["time"], errors="ignore")
    dataframe = dataframe.rename(columns={"tick_volume": "volume"})
    dataframe = dataframe[["timestamp", "open", "high", "low", "close", "volume"]]
    dataframe = dataframe.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MT5 OHLCV data for the CRT system")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated symbols")
    parser.add_argument("--timeframe", default="1h", choices=sorted(TIMEFRAME_MAP.keys()))
    parser.add_argument("--date-from", default="2026-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--date-to", default="2026-03-22", help="End date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", default="data_crt_h1", help="Folder to store the CSV files")
    parser.add_argument("--max-bars", type=int, default=10_000_000, help="Maximum bars to request when no date range is supplied")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    timeframe = TIMEFRAME_MAP[args.timeframe]
    date_from = pd.to_datetime(args.date_from) if args.date_from else None
    date_to = pd.to_datetime(args.date_to) if args.date_to else None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    login, password, server = load_mt5_credentials()

    print("===================================================")
    print("MT5 HISTORICAL DATA DOWNLOADER FOR CRT")
    print(f"Timeframe : {args.timeframe}")
    print(f"Range     : {date_from.date() if date_from is not None else 'full'} -> {date_to.date() if date_to is not None else 'full'}")
    print(f"Symbols   : {symbols}")
    print("===================================================\n")

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed | error: {mt5.last_error()}")

    if not mt5.login(login=login, password=password, server=server):
        mt5.shutdown()
        raise RuntimeError(f"MT5 login failed | error: {mt5.last_error()}")

    account = mt5.account_info()
    if account is None:
        mt5.shutdown()
        raise RuntimeError(f"No account is logged in | error: {mt5.last_error()}")

    print(f"Connected to MT5 account {account.login} on {account.server}")
    print(f"Currency: {account.currency}\n")

    success = []
    failed = []

    for index, symbol in enumerate(symbols, start=1):
        print("---------------------------------------------------")
        print(f"[{index}/{len(symbols)}] Downloading {symbol} ({args.timeframe})")
        start_time = time.time()

        if not mt5.symbol_select(symbol, True):
            print(f"FAILED: symbol not found | error: {mt5.last_error()}")
            failed.append(symbol)
            continue

        mt5.symbol_info_tick(symbol)
        time.sleep(1.0)

        try:
            dataframe = download_symbol_rates(symbol, timeframe, date_from, date_to)
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed.append(symbol)
            continue

        output_path = os.path.join(output_dir, f"{symbol}_{args.timeframe}.csv")
        dataframe.to_csv(output_path, index=False)

        elapsed = time.time() - start_time
        print("SUCCESS")
        print(f"Bars saved : {len(dataframe):,}")
        print(f"Range      : {dataframe['timestamp'].iloc[0]} -> {dataframe['timestamp'].iloc[-1]}")
        print(f"File       : {output_path}")
        print(f"Elapsed    : {elapsed:.2f}s")
        success.append(symbol)

    print("\n===================================================")
    print(f"Finished | success: {len(success)} | failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {failed}")
    print("===================================================")

    mt5.shutdown()


if __name__ == "__main__":
    main()