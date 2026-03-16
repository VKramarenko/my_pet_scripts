from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from sim.core.events import MarketSnapshot, MarketTrade


def _parse_levels(raw: object) -> list[tuple[float, float]]:
    if isinstance(raw, list):
        return [(float(p), float(s)) for p, s in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        return [(float(p), float(s)) for p, s in parsed]
    raise ValueError(f"Unsupported levels payload: {type(raw)}")


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


def _extract_snapshot_fields(row: dict[str, Any]) -> tuple[float, object, object]:
    if {"timestamp", "bids", "asks"}.issubset(row):
        return float(row["timestamp"]), row["bids"], row["asks"]
    if {"ts", "b", "a"}.issubset(row):
        return float(row["ts"]), row["b"], row["a"]
    payload = row.get("data")
    if isinstance(payload, dict):
        if {"timestamp", "bids", "asks"}.issubset(payload):
            return float(payload["timestamp"]), payload["bids"], payload["asks"]
        if {"ts", "b", "a"}.issubset(payload):
            return float(payload["ts"]), payload["b"], payload["a"]
    raise ValueError("Unsupported orderbook snapshot format")


def load_l2_snapshots(path: str | Path) -> Iterator[MarketSnapshot]:
    frame = _read_frame(path).sort_values("ts")
    for row in frame.itertuples(index=False):
        yield MarketSnapshot(
            ts=float(row.ts),
            bids=_parse_levels(row.bids),
            asks=_parse_levels(row.asks),
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
    rows = []
    for row in records:
        ts, bids, asks = _extract_snapshot_fields(row)
        rows.append((ts, bids, asks))
    rows.sort(key=lambda x: x[0])
    for ts, bids, asks in rows:
        yield MarketSnapshot(ts=ts, bids=_parse_levels(bids), asks=_parse_levels(asks))


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
