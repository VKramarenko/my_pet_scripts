from __future__ import annotations

import datetime
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from sim.core.events import MarketSnapshot
from sim.data.book_schema import (
    detect_book_depth,
    level_column_names,
    required_wide_level_columns,
    resolve_time_column,
)
from sim.data.level_utils import bybit_row_ts_bids_asks, parse_levels


def row_time_to_float(v: object) -> float:
    if pd.isna(v):
        raise ValueError("snapshot time is NaN")
    if isinstance(v, pd.Timestamp):
        return float(v.timestamp())
    if isinstance(v, datetime.datetime):
        return v.timestamp()
    if isinstance(v, (float, int)) and not isinstance(v, bool):
        return float(v)
    return float(pd.Timestamp(v).timestamp())


def _finite_pair(price: object, size: object) -> tuple[float, float] | None:
    if pd.isna(price) or pd.isna(size):
        return None
    try:
        p = float(price)
        s = float(size)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or not math.isfinite(s):
        return None
    return (p, s)


def wide_row_to_levels(
    row: pd.Series, depth: int
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    bids: list[tuple[float, float]] = []
    asks: list[tuple[float, float]] = []
    for i in range(1, depth + 1):
        ap, bp, asz, bsz = level_column_names(i)
        pair_b = _finite_pair(row[bp], row[bsz])
        if pair_b is not None:
            bids.append(pair_b)
        pair_a = _finite_pair(row[ap], row[asz])
        if pair_a is not None:
            asks.append(pair_a)
    return bids, asks


def wide_row_to_market_snapshot(
    row: pd.Series, depth: int, time_col: str
) -> MarketSnapshot:
    ts = row_time_to_float(row[time_col])
    sym_raw = row["symbol"]
    symbol = None if pd.isna(sym_raw) else str(sym_raw)
    bids, asks = wide_row_to_levels(row, depth)
    return MarketSnapshot(ts=ts, bids=bids, asks=asks, symbol=symbol)


def validate_wide_book_columns(df: pd.DataFrame, depth: int) -> None:
    missing = [n for n in required_wide_level_columns(depth) if n not in df.columns]
    if missing:
        preview = ", ".join(missing[:12])
        more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        raise ValueError(f"Wide L2 frame missing columns: {preview}{more}")


def dataframe_rows_to_snapshots(
    frame: pd.DataFrame, depth: int, time_col: str
) -> Iterator[MarketSnapshot]:
    for _, row in frame.iterrows():
        yield wide_row_to_market_snapshot(row, depth, time_col)


def legacy_bids_asks_to_wide(
    df: pd.DataFrame,
    depth: int,
    symbol: str,
    time_col: str = "ts",
) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"legacy frame missing time column {time_col!r}")
    if "bids" not in df.columns or "asks" not in df.columns:
        raise ValueError("legacy frame must have 'bids' and 'asks' columns")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        bids = parse_levels(row["bids"])
        asks = parse_levels(row["asks"])
        rec: dict[str, Any] = {
            "time": row_time_to_float(row[time_col]),
            "symbol": symbol,
        }
        for i in range(1, depth + 1):
            ap, bp, asz, bsz = level_column_names(i)
            if i <= len(asks):
                rec[ap] = asks[i - 1][0]
                rec[asz] = asks[i - 1][1]
            else:
                rec[ap] = math.nan
                rec[asz] = math.nan
            if i <= len(bids):
                rec[bp] = bids[i - 1][0]
                rec[bsz] = bids[i - 1][1]
            else:
                rec[bp] = math.nan
                rec[bsz] = math.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {path}")
        return payload
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def bybit_snapshots_to_wide(
    path_or_records: str | Path | list[dict[str, Any]],
    depth: int | None = None,
    symbol_default: str | None = None,
) -> pd.DataFrame:
    if isinstance(path_or_records, list):
        records = path_or_records
    else:
        records = _read_json_records(path_or_records)
    parsed: list[tuple[float, list[tuple[float, float]], list[tuple[float, float]], str]] = []
    for row in records:
        ts, br, ar = bybit_row_ts_bids_asks(row)
        bids = parse_levels(br)
        asks = parse_levels(ar)
        sym = row.get("symbol")
        if sym is not None:
            symbol = str(sym)
        else:
            symbol = symbol_default or ""
        parsed.append((ts, bids, asks, symbol))
    parsed.sort(key=lambda x: x[0])
    if depth is None:
        depth = 1
        for _, bids, asks, _ in parsed:
            depth = max(depth, len(bids), len(asks))
    rows: list[dict[str, Any]] = []
    for ts, bids, asks, symbol in parsed:
        rec: dict[str, Any] = {"time": ts, "symbol": symbol}
        for i in range(1, depth + 1):
            ap, bp, asz, bsz = level_column_names(i)
            if i <= len(asks):
                rec[ap] = asks[i - 1][0]
                rec[asz] = asks[i - 1][1]
            else:
                rec[ap] = math.nan
                rec[asz] = math.nan
            if i <= len(bids):
                rec[bp] = bids[i - 1][0]
                rec[bsz] = bids[i - 1][1]
            else:
                rec[bp] = math.nan
                rec[bsz] = math.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def wide_to_legacy_lists(df: pd.DataFrame) -> pd.DataFrame:
    depth = detect_book_depth(list(df.columns))
    if depth < 1:
        raise ValueError("no ask_price_N / bid_price_N columns found")
    validate_wide_book_columns(df, depth)
    time_col = resolve_time_column(list(df.columns))
    out_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        bids, asks = wide_row_to_levels(row, depth)
        out_rows.append(
            {
                "ts": row_time_to_float(row[time_col]),
                "bids": json.dumps(bids),
                "asks": json.dumps(asks),
            }
        )
    return pd.DataFrame(out_rows)
