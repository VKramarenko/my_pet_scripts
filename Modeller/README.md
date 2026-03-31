# Modeller

Python simulator for strategy backtesting on L2 snapshots (trades optional) with pluggable execution models.

## Input data and assumptions

Supported inputs: `CSV` or `Parquet`.

1. `l2_snapshots` (choose one layout):

**Wide L2 (canonical)** — use `--loader wide`:

- `time` — snapshot time (epoch seconds as float, or parseable datetime); alias: `ts`
- `symbol` — instrument id (string)
- For each level `i` from 1 to N: `ask_price_i`, `bid_price_i`, `ask_size_i`, `bid_size_i`
- Missing levels: use NaN for all four columns of that level; partial invalid pairs are skipped when building the book
- If the file has more than one distinct `symbol`, pass `--symbol SYMBOL` (or `symbol_filter` in code) so the backtest replays a single instrument

**Legacy list columns** — default loader:

- required columns: `ts`, `bids`, `asks`
- `bids` / `asks` are arrays of `(price, size)` levels

Conversion helpers (in `sim.data.book_converters`): `legacy_bids_asks_to_wide`, `bybit_snapshots_to_wide`, `wide_to_legacy_lists`.

CLI: from the Modeller directory, convert Bybit orderbook JSON to wide CSV (auto depth, auto-detect format from `.json`):

```bash
python convert_l2_to_wide.py -i test_data/bybit_custom_loader/orderbook_BTCUSDT_spot_2026-03-04.json -o orderbook_BTCUSDT_wide.csv
```

Legacy `ts`+`bids`+`asks` file:

```bash
python convert_l2_to_wide.py --format legacy -i path/to/l2.csv -o out_wide.csv --symbol BTCUSDT
```

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

Wide L2 example:

```bash
python run_backtest.py --loader wide --l2 path/to/l2_wide.csv --symbol BTCUSDT
```

(`--symbol` is only required when the wide file contains multiple symbols.)

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
