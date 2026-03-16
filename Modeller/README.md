# Modeller

Python simulator for strategy backtesting on L2 snapshots (trades optional) with pluggable execution models.

## Input data and assumptions

Supported inputs: `CSV` or `Parquet`.

1. `l2_snapshots`:
- required columns: `ts`, `bids`, `asks`
- `bids` / `asks` are arrays of `(price, size)` levels
2. `trades` (optional):
- required columns: `ts`, `price`, `size`, `side`
- `side` is `buyer_initiated` or `seller_initiated`

Assumption:
- snapshots are full L2 replacements (book is overwritten by each snapshot)
- no incremental book deltas are required

## Architecture

1. `sim/core`: event definitions and replay engine
2. `sim/data`: loaders and stream merge/normalization
3. `sim/market`: `OrderBookL2` state model (no fill logic)
4. `sim/execution`: fill model interface + `Touch` and `Queue` implementations
5. `sim/exchange`: simulated exchange, order lifecycle, fees
6. `sim/portfolio`: portfolio accounting and metrics
7. `sim/strategy`: strategy interface, basic MM, and taker Bollinger strategy

## Event ordering

`merge_streams(...)` guarantees deterministic ordering by:
1. `ts`
2. event type priority: `snapshot -> trade -> timer`
3. stable sequence index inside each type

## Project layout

```text
sim/
  core/
  data/
  market/
  exchange/
  execution/
  strategy/
  portfolio/
  tests/
run_backtest.py
README.md
```

## Run tests

```bash
pytest -q
```

## Minimal run

```bash
python run_backtest.py --l2 path/to/l2.csv
```

Optional trades merge:

```bash
python run_backtest.py --l2 path/to/l2.csv --trades path/to/trades.csv
```

`run_backtest.py` is headless and contains no visualization dependencies.

## Strategy selection

Default strategy is market maker (`mm`).

```bash
python run_backtest.py --l2 path/to/l2.csv --strategy mm --mm-spread 1.0 --mm-quote-size 1.0
```

Taker Bollinger breakout/revert example:

```bash
python run_backtest.py --l2 path/to/l2.csv --strategy taker --taker-window 20 --taker-std-mult 2.0 --taker-qty 1.0 --taker-cooldown 5 --taker-max-position 1.0
```

`taker` uses market-only orders with Bollinger breakout entries and mean-revert exit/reverse logic.
Risk controls:
- `--taker-cooldown`: seconds between new entries from flat
- `--taker-max-position`: max absolute position cap

## Optional Dash visualization

Install optional UI dependencies:

```bash
pip install dash plotly
```

Run the simulator with a Dash dashboard:

```bash
python run_backtest_dash.py --l2 path/to/l2.csv --strategy taker
```

Dashboard includes:
- run form with strategy and loader parameters
- `Equity` tab with equity curve
- `Stats` tab with key performance numbers
- `Fills` tab with executed fills table

Architecture note:
- simulation core works independently from visualization
- Dash app consumes `MetricsCollector` output as an optional adapter layer

## Bybit custom loader example

```bash
python run_backtest.py ^
  --loader bybit ^
  --l2 test_data/bybit_custom_loader/orderbook_BTCUSDT_spot_2026-03-04.json ^
  --trades test_data/bybit_custom_loader/trades_BTCUSDT_spot_2026-03-04.json ^
  --strategy taker
```

`test_data` loader mode resolves files inside project `test_data` automatically:

```bash
python run_backtest.py --loader test_data --l2 orderbook_BTCUSDT_spot_2026-03-04.json --strategy taker
```
