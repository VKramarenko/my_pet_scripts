from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
import pandas as pd

from sim.core.events import MarketSnapshot, MarketTrade
from sim.data.book_converters import (
    dataframe_rows_to_snapshots,
    row_time_to_float,
    validate_wide_book_columns,
)
from sim.data.book_schema import detect_book_depth, resolve_time_column
from sim.data.level_utils import bybit_row_ts_bids_asks, parse_levels


def _read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}")


def _read_json_records(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}")
        return payload
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _resolve_test_data_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested.exists():
        return requested

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "test_data" / requested,
        project_root / "test_data" / "bybit_custom_loader" / requested.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve test_data file for '{path}'. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def load_l2_snapshots(path: str | Path) -> Iterator[MarketSnapshot]:
    frame = _read_frame(path).sort_values("ts")
    for row in frame.itertuples(index=False):
        yield MarketSnapshot(
            ts=float(row.ts),
            bids=parse_levels(row.bids),
            asks=parse_levels(row.asks),
        )


def load_trades(path: str | Path) -> Iterator[MarketTrade]:
    frame = _read_frame(path).sort_values("ts")
    for row in frame.itertuples(index=False):
        yield MarketTrade(
            ts=float(row.ts),
            price=float(row.price),
            size=float(row.size),
            side=str(row.side),
        )


def load_bybit_l2_snapshots(path: str | Path) -> Iterator[MarketSnapshot]:
    records = _read_json_records(path)
    rows: list[tuple[float, object, object, str | None]] = []
    for row in records:
        ts, bids, asks = bybit_row_ts_bids_asks(row)
        sym = row.get("symbol")
        symbol = None if sym is None else str(sym)
        rows.append((ts, bids, asks, symbol))
    rows.sort(key=lambda x: x[0])
    for ts, bids, asks, symbol in rows:
        yield MarketSnapshot(
            ts=ts, bids=parse_levels(bids), asks=parse_levels(asks), symbol=symbol
        )


def load_wide_l2_snapshots(
    path: str | Path,
    symbol_filter: str | None = None,
) -> Iterator[MarketSnapshot]:
    frame = _read_frame(path)
    depth = detect_book_depth(list(frame.columns))
    if depth < 1:
        raise ValueError(
            "Wide L2 frame must include level columns like ask_price_1, bid_price_1, ..."
        )
    validate_wide_book_columns(frame, depth)
    time_col = resolve_time_column(list(frame.columns))
    if "symbol" not in frame.columns:
        raise ValueError("Wide L2 frame must include a 'symbol' column")
    symbols = frame["symbol"].astype(str).unique()
    if len(symbols) > 1 and symbol_filter is None:
        raise ValueError(
            "Wide L2 file contains multiple symbols "
            f"{sorted(symbols.tolist())!r}; pass symbol_filter=... or --symbol"
        )
    if symbol_filter is not None:
        frame = frame[frame["symbol"].astype(str) == symbol_filter]
        if frame.empty:
            raise ValueError(f"No rows for symbol_filter={symbol_filter!r}")
    frame = frame.copy()
    frame["_ts_sort"] = frame[time_col].map(row_time_to_float)
    frame = frame.sort_values("_ts_sort")
    yield from dataframe_rows_to_snapshots(frame, depth, time_col)


def load_test_data_l2_snapshots(path: str | Path) -> Iterator[MarketSnapshot]:
    resolved_path = _resolve_test_data_path(path)
    yield from load_bybit_l2_snapshots(resolved_path)


def load_test_data_orderbooks(path: str | Path) -> Iterator[MarketSnapshot]:
    """
    Backward-compatible alias for test_data orderbook snapshots.
    """
    yield from load_test_data_l2_snapshots(path)


def load_test_data_trades(path: str | Path) -> Iterator[MarketTrade]:
    """Load trades from test_data with path resolution."""
    resolved_path = _resolve_test_data_path(path)
    yield from load_bybit_trades(resolved_path)


def load_l2_binance(
    data_dir: str,
    symbol: str,
    depth: int = 25,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Iterator[MarketSnapshot]:
    """Загружает Binance L2 из JSONL.gz (snapshot + diff) → MarketSnapshot."""
    from sim.data.binance_loader import load_binance_l2

    yield from load_binance_l2(
        data_dir=data_dir,
        symbol=symbol,
        depth=depth,
        start_ts=start_ts,
        end_ts=end_ts,
    )


def load_bybit_trades(path: str | Path) -> Iterator[MarketTrade]:
    records = _read_json_records(path)
    records.sort(key=lambda x: float(x["timestamp"]))
    for row in records:
        side_raw = str(row["side"]).lower()
        if side_raw == "buy":
            side = "buyer_initiated"
        elif side_raw == "sell":
            side = "seller_initiated"
        else:
            raise ValueError(f"Unsupported Bybit trade side: {row['side']}")
        yield MarketTrade(
            ts=float(row["timestamp"]),
            price=float(row["price"]),
            size=float(row["size"]),
            side=side,
        )
