# 📘 vAlgo Trading System

## 📌 Project Overview

A modular, Excel-configurable, multi-instrument algo trading system designed for both backtesting and live execution using multiple brokers. Supports multi-threaded parallel execution, dynamic indicators, rule-based signal generation, and complete strategy control from Excel.

---

## 🎯 Objectives

- Excel-based strategy config (indicators, rules, capital, instruments)
- Multi-instrument, multi-threaded trading
- Indicator and rule engine built on Pandas
- Live and backtest support
- Visual trade reports and equity curves

---

## 🔧 Key Features

| Feature             | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| Excel Config        | Define all strategy logic via `config.xlsx`                     |
| Indicator Engine    | Modular EMA, RSI, VWAP, CPR, Supertrend                         |
| Rule Engine         | Evaluates expressions like `EMA_9 > EMA_21 AND RSI < 60`        |
| Backtest Engine     | Simulates trades, logs results, and calculates metrics          |
| Live Trading Engine | Subscribes to Angel One WebSocket and executes trades           |
| Data Manager        | Fetches & stores OHLCV data using Angel One API                 |
| Parallel Execution  | ThreadPoolExecutor for multi-symbol backtest and live execution |
| Reports             | Generates Excel trade logs, summary stats, and equity curves    |

---

## 🔢 Folder Structure

## 🔧 Modules to Build (Folder-Level View)
```
vAlgo/
├── config/               # Excel configs
├── data_manager/         # Historical data storage/fetch
├── indicators/           # Indicator calculations
├── rules/                # Rule parsing & evaluation
├── strategies/           # Strategy base class/logic
├── backtest_engine/      # Simulated backtesting
├── live_trading/         # WebSocket & order placement
├── utils/                # Logging, file handling, Telegram
├── outputs/              # Trade logs & reports
├── logs/                 # Runtime logs
├── main_backtest.py      # Entry point for backtest
├── main_live.py          # Entry point for live trading
├── main_fetch_data.py    # Standalone OHLC fetcher
├── requirements.txt
└── README.md

```
vAlgo/
├── config/
│   └── config.xlsx
├── data_manager/
│   ├── ohlc_fetcher.py
│   ├── duckdb_handler.py
│   └── main_fetch_data.py
├── indicators/
│   ├── ema.py
│   ├── rsi.py
│   ├── vwap.py
│   ├── cpr.py
│   └── supertrend.py
├── rules/
│   ├── rule_parser.py
│   └── rule_evaluator.py
├── strategies/
│   └── base_strategy.py
├── backtest_engine/
│   ├── backtester.py
│   ├── trade_logger.py
│   ├── performance_metrics.py
│   └── multi_symbol_runner.py
├── live_trading/
│   ├── websocket_listener.py
│   ├── order_manager.py
│   └── executor.py
├── utils/
│   ├── config_loader.py
│   ├── logger.py
│   ├── file_utils.py
│   └── telegram_alert.py
├── outputs/
│   ├── trade_logs/
│   └── reports/
├── logs/
├── main_backtest.py
├── main_live.py
├── main_fetch_data.py
├── requirements.txt
└── README.md
```

---

## 📊 Config Sheets (`config.xlsx`)

### Sheet: `Initialize`

| Parameter  | Value         |
| ---------- | ------------- |
| Mode       | backtest/live |
| Start Date | DD-MM-YYY    |
| End Date   | DD-MM-YYY    |
| Capital    | 150000        |

### Sheet: `Instruments`

| Symbol    | Timeframe | Status |
| --------- | --------- | ------ |
| NIFTY     | 1min      | Active |
| BANKNIFTY | 1min      | Active |

### Sheet: `Indicators`

| Indicator | Status | Parameters |
| --------- | ------ | ---------- |
| EMA       | Active | 9,21,50    |
| RSI       | Active | 14         |
| CPR       | Active | -          |

### Sheet: `Rules`

| Rule ID | Type  | Condition                         |
| ------- | ----- | --------------------------------- |
| COND1   | Entry | EMA_9 > EMA_21 AND RSI_14 < 60 |
| COND2   | Exit  | Close < Supertrend                |

---

## 🛠️ Development Phases

### Phase 0: Git & Environment

- `git init` project and add `.gitignore`
- Setup `requirements.txt`

### Phase 1: Config Loader

- Load Excel config from `/config/config.xlsx`

### Phase 2: Data Manager

- Fetch OHLCV using Angel One
- Store in DuckDB/Parquet

### Phase 3: Indicator Engine

- Calculate indicators from config

### Phase 4: Rule Engine

- Parse rule expressions with Rule IDs
- Evaluate and return signal columns

### Phase 5: Backtest Engine

- Run simulations for each instrument in parallel

### Phase 6: Live Trading Engine

- Subscribes to WebSocket
- Evaluates and executes trades

### Phase 7: Metrics & CLI

- Output metrics (Sharpe, PnL, drawdown)
- CLI using `argparse`

### Phase 8: Logging, Alerts, Reports

- Logging via `logger.py`
- Telegram alerts
- Excel-based trade logs and equity curves

---

## 📈 Example CLI

```
python main_backtest.py --symbol NIFTY --start-date 2023-01-01 --end-date 2023-12-31
```

---

## 🥊 Tech Stack

- **Language**: Python 3.10+
- **Data**: Pandas, DuckDB or Parquet
- **Broker API**: Angel One SmartAPI or other broker api
- **Execution**: ThreadPoolExecutor or Asyncio
- **Config UI**: Excel (openpyxl/pandas)
- **Visualization**: Matplotlib, Plotly

---

## 🔧 Roadmap Enhancements

- Parameter optimizer (grid search)
- Streamlit dashboard (optional)
- Strategy factory to mix indicators
- Paper trade mode

---

## 🚀 Quick Start

1. Install requirements: `pip install -r requirements.txt`
2. Update `config/config.xlsx`
3. Run `main_fetch_data.py`
4. Run `main_backtest.py` or `main_live.py`
5. Check `outputs/` for trade logs and reports

---

> Built for personal quant research with flexibility, modularity, and rapid testing in mind. Fully editable, extensible, and AI-development-friendly.

---

Feel free to contribute or fork for personal usage. For any doubts or bugs, reach out via GitHub Issues.