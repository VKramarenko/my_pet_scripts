from __future__ import annotations

import ast
import json
from typing import Any


def parse_levels(raw: object) -> list[tuple[float, float]]:
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


def bybit_row_ts_bids_asks(row: dict[str, Any]) -> tuple[float, object, object]:
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
