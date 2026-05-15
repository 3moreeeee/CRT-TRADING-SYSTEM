# CRT Trading System

A fresh, production-oriented trading research system built around **Candle Range Theory (CRT)**, **Power of 3**, **AMD behavior**, **liquidity sweeps**, **key-time execution**, and **lower-timeframe confirmation**.

This project is designed to do one thing well: turn CRT market structure into a fully systematic pipeline that can **detect setups, build objective features, generate labels, train a machine learning filter, and replay/backtest trades** using historical OHLCV data.

## Core Idea

CRT treats each higher-timeframe candle as a **range**.
Price is then analyzed through the three market phases:

1. **Accumulation** - price builds a range and compresses.
2. **Manipulation** - price sweeps liquidity beyond one side of the range.
3. **Distribution** - price reclaims back inside and moves toward the opposite side of the range.

The strategy in this repository is built to detect that sequence mechanically:
- identify the higher-timeframe range candle,
- detect the sweep,
- confirm the reclaim,
- zoom into a lower timeframe for execution,
- place objective entry, stop, and take-profit rules,
- label the outcome,
- and optionally train a machine learning model to filter low-quality setups.

## What This Project Includes

### 1. Data pipeline
- MT5 OHLCV downloader for CRT data.
- CSV ingestion with timestamp normalization.
- Multi-timeframe support.
- Setup timeframe and execution timeframe separation.

### 2. CRT setup detection
- Range candle high, low, midpoint, size.
- Sweep detection above or below the range.
- Reclaim detection back inside the range.
- Bullish and bearish CRT classification.
- Time-window and session-aware logic.

### 3. Objective execution logic
- Lower-timeframe order-block style entry selection.
- Stop-loss placement beyond invalidation.
- TP1 at midpoint of the range.
- TP2 at the opposite side of the range.
- Single-target and multi-target support.

### 4. Feature engineering
The system builds features from scratch across these groups:
- range structure,
- sweep and reclaim quality,
- distribution and target distance,
- key-time and session behavior,
- lower-timeframe entry confirmation,
- volatility and spread context,
- higher-timeframe trend and liquidity context.

### 5. Labeling
The main target is whether price reaches the chosen target before stop loss.
This gives a clean supervised label for ML training:
- `tp1` hit before stop,
- `tp2` hit before stop,
- or invalid / failed setup.

### 6. Machine learning
ML is used as a **filter**, not as a replacement for the CRT strategy.
The model learns from objective CRT labels and features so it can rank or reject weak setups before execution.

### 7. Backtesting and replay
- Sequential historical replay.
- Trade log output.
- Equity curve output.
- Summary statistics.
- Optional separate execution-timeframe CSV support.

---

## Strategy Logic

This project uses a clean mechanical interpretation of CRT.

### CRT setup rules
A valid setup usually includes:
- a higher-timeframe candle acting as the range candle,
- a sweep beyond one side of that candle,
- a reclaim back inside the range,
- confirmation that the move is likely to continue toward the opposite side.

### Bullish CRT
A bullish CRT generally looks like this:
- price sweeps **below** the range candle low,
- price reclaims back inside the range,
- price is expected to move upward toward the midpoint or full range high.

### Bearish CRT
A bearish CRT generally looks like this:
- price sweeps **above** the range candle high,
- price reclaims back inside the range,
- price is expected to move downward toward the midpoint or full range low.

### Power of 3 and AMD
The system also models the classic three-phase behavior:
- **Accumulation**
- **Manipulation**
- **Distribution**

It uses session and key-time features to capture the idea that certain windows are more likely to produce clean CRT behavior.

### Key times
The strategy pays attention to these windows:
- 1am
- 5am
- 9am
- 1pm
- 3pm
- 6pm
- 9pm

These are encoded as time features and are used by the filtering logic.

### Multi-timeframe design
The project supports the standard CRT execution logic:
- monthly CRT -> daily execution
- daily CRT -> 1H execution
- 4H CRT -> 15m execution
- 1H CRT -> 5m or 1m execution
- 15m CRT -> 1m execution

For this repository, the most common use case is:
- **1H setup timeframe**
- **5m or 1m execution timeframe**

---

## Machine Learning Role

The ML part of the project is intentionally narrow and disciplined.

### What the model learns
The classifier is trained on CRT examples using features like:
- range size and ATR ratio,
- sweep depth,
- reclaim strength,
- time-of-day and session alignment,
- order-block quality,
- reward-to-risk potential,
- trend and liquidity context,
- volatility regime.

### What the model predicts
The target is built from trade outcomes:
- did the setup reach the target before the stop?
- was the setup worth taking?

This makes the model a **quality filter** for CRT setups.
It does not generate random signals on its own.
It learns which CRT conditions historically produced better outcomes.

### Why this matters
This keeps the system closer to the market structure thesis:
- rules define the strategy,
- ML improves selectivity,
- backtesting verifies the behavior.

---

## Repository Structure

### Main CRT package
- [crt_system/config.py](crt_system/config.py) - strategy configuration.
- [crt_system/data.py](crt_system/data.py) - OHLCV loading and timeframe handling.
- [crt_system/detector.py](crt_system/detector.py) - CRT setup detection.
- [crt_system/execution.py](crt_system/execution.py) - lower-timeframe execution planning.
- [crt_system/features.py](crt_system/features.py) - feature engineering.
- [crt_system/labels.py](crt_system/labels.py) - label generation and trade simulation.
- [crt_system/filters.py](crt_system/filters.py) - objective trade-quality filters.
- [crt_system/modeling.py](crt_system/modeling.py) - ML training and evaluation.
- [crt_system/backtest.py](crt_system/backtest.py) - historical replay/backtest engine.
- [crt_system/pipeline.py](crt_system/pipeline.py) - end-to-end orchestration.

### Command-line tools
- [download_crt_data_mt5.py](download_crt_data_mt5.py) - download MT5 OHLCV CSV files.
- [run_crt_backtest.py](run_crt_backtest.py) - replay/backtest the CRT system.
- [run_crt_pipeline.py](run_crt_pipeline.py) - build a dataset and train the ML model.

---

## How To Use It

### 1. Activate the environment
```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Download H1 setup data
```powershell
python download_crt_data_mt5.py --symbols EURUSD --timeframe 1h
```

### 4. Download lower-timeframe execution data
```powershell
python download_crt_data_mt5.py --symbols EURUSD --timeframe 5m
```

### 5. Run the backtest
```powershell
python run_crt_backtest.py --input data_crt_h1/EURUSD_1h.csv --execution-input data_crt/EURUSD_5m.csv --setup-timeframe 1h --execution-timeframe 5m
```

### 6. Review outputs
The backtest writes:
- `crt_backtest_output/crt_trades.csv`
- `crt_backtest_output/crt_equity_curve.csv`
- `crt_backtest_output/crt_summary.json`

---

## How The Backtest Works

The replay engine processes the strategy in the same order a trader would:

1. load the setup-timeframe CSV,
2. detect CRT range candles,
3. identify sweep/reclaim structures,
4. load the execution timeframe if provided,
5. compute objective entries, stops, and targets,
6. simulate trade outcomes bar by bar,
7. collect trade statistics,
8. build the equity curve.

This makes the backtest suitable for research, filtering, and strategy refinement.

---

## Output Metrics

The backtest reports:
- trade count,
- win rate,
- average R,
- expectancy,
- profit factor,
- maximum drawdown,
- final equity.

These are the primary metrics used to judge whether a CRT configuration is worth further refinement.

---

## Design Philosophy

This system is built with a few strict principles:

### 1. Rule-first, ML-second
The market structure defines the trade.
ML only refines the quality of the trade.

### 2. Objective over discretionary
Every major decision is encoded as a rule:
- range selection,
- sweep validation,
- reclaim validation,
- execution confirmation,
- SL/TP definition.

### 3. Multi-timeframe consistency
The setup timeframe and execution timeframe are always treated separately.
That is essential for CRT.

### 4. Time matters
Key session windows are not decoration.
They are part of the system.

### 5. Trade quality must be filterable
Not every CRT setup is equal.
The system uses filters and features to reject weak conditions.

---

## Notes On Current Results

The pipeline is working end to end, but raw CRT performance depends on:
- symbol,
- timeframe,
- session,
- spread,
- date range,
- target mode,
- filter strictness.

If the strategy is too loose, the backtest will show that immediately.
That is useful, because it tells you which conditions need tightening.

---

## Suggested Next Steps

1. Tighten the CRT filters for session and sweep quality.
2. Add symbol-specific parameter presets.
3. Compare `tp1` vs `tp2` target modes.
4. Add walk-forward validation for robust evaluation.
5. Add a live scanner that only emits high-quality CRT setups.

---

## Short Version

This is a CRT research and execution framework that combines:
- market structure logic,
- session-aware behavior,
- lower-timeframe confirmation,
- objective risk management,
- machine learning filtering,
- and historical replay/backtesting.

It is designed to be readable, systematic, and extensible.
