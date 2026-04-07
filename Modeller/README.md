# Modeller

Python simulator for backtesting strategies on L2 orderbook snapshots.

## Current workflow

The project now uses one canonical orderbook input format:

- `wide CSV` for L2 snapshots
- optional `CSV` for trades
- JSON config file for strategy and run parameters

The intended pipeline is:

1. Record raw data with a dedicated loader script.
2. Convert raw orderbook data to wide CSV.
3. Run the backtest from `backtest_config.json` or another config file.

## Main scripts

- `recorder_bybit.py` - live Bybit orderbook recorder
- `recorder_binance.py` - Binance recorder/downloader
- `convert_l2_to_wide.py` - converts raw Bybit JSON or legacy L2 into canonical wide CSV
- `run_backtest.py` - headless backtest runner
- `run_backtest_dash.py` - Dash UI on top of the same backtest core

## Canonical L2 format

Backtests read only wide L2 CSV files.

Required columns:

- `time` or `ts`
- `symbol`
- `ask_price_1`, `ask_size_1`, `bid_price_1`, `bid_size_1`
- the same pattern for deeper levels: `..._2`, `..._3`, and so on

Optional notes:

- if a CSV contains multiple symbols, set `data.symbol` in config
- missing levels should be `NaN`
- trades CSV, when used, must contain `ts`, `price`, `size`, `side`

## Config-driven runs

The default config file is `backtest_config.json`.
There is also a ready-to-load example: `backtest_config.example.json`.

Minimal launch:

```bash
python run_backtest.py
```

Use another config:

```bash
python run_backtest.py --config path/to/my_backtest.json
python run_backtest_dash.py --config path/to/my_backtest.json
python run_backtest.py --config backtest_config.example.json
```

Config structure:

```json
{
  "data": {
    "l2_path": "test_data/test.csv",
    "trades_path": null,
    "symbol": null
  },
  "strategy": {
    "name": "mm",
    "mm": { "spread": 1.0, "quote_size": 1.0 },
    "taker": {
      "window": 20,
      "std_mult": 2.0,
      "qty": 1.0,
      "cooldown": 0.0,
      "max_position": 1.0
    },
    "ema": {
      "fast": 10,
      "slow": 30,
      "qty": 1.0,
      "max_position": 1.0,
      "offset": 0.0
    },
    "imbalance": {
      "depth": 5,
      "threshold": 0.3,
      "smoothing": 3,
      "qty": 1.0,
      "max_position": 1.0
    }
  },
  "dashboard": {
    "host": "127.0.0.1",
    "port": 860,
    "debug": false
  },
  "console": {
    "level": 1,
    "book_levels": 1
  }
}
```

If some strategy fields are omitted, code defaults are used automatically.

Console output modes:

- `console.level = 1` keeps the compact summary output
- `console.level = 2` prints every strategy action to console
- `console.level = 3` includes strategy actions plus fills, position, cash and equity
- `console.book_levels` controls how many bid/ask levels are printed for each action

## Bybit pipeline example

1. Record raw orderbook:

```bash
python recorder_bybit.py --symbol BTCUSDT --duration 1800
```

2. Convert raw JSON to wide CSV:

```bash
python convert_l2_to_wide.py -i test_data/bybit_custom_loader/orderbook_BTCUSDT_spot_2026-03-04.json -o orderbook_BTCUSDT_wide.csv
```

3. Point `data.l2_path` in config to the generated CSV and run:

```bash
python run_backtest.py --config backtest_config.json
```

## Optional Dash visualization

Install optional UI dependencies:

```bash
pip install dash plotly
```

Then run:

```bash
python run_backtest_dash.py
```

The UI uses the same config model and reads the same wide CSV files.

## Tests

```bash
pytest -q
```
