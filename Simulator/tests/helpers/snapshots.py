from __future__ import annotations

from datetime import datetime

from src.models import Level, Snapshot


def make_snapshot(
    timestamp: datetime,
    *,
    best_ask: float = 101.0,
    best_bid: float = 100.0,
    ask_qty: float = 1.0,
    bid_qty: float = 1.0,
) -> Snapshot:
    return Snapshot(
        timestamp=timestamp,
        asks=[Level(best_ask, ask_qty)],
        bids=[Level(best_bid, bid_qty)],
    )


def make_two_level_snapshot(
    timestamp: datetime,
    *,
    asks: list[tuple[float, float]] | None = None,
    bids: list[tuple[float, float]] | None = None,
) -> Snapshot:
    asks = asks or [(101.0, 1.0), (102.0, 2.0)]
    bids = bids or [(100.0, 1.0), (99.0, 2.0)]
    return Snapshot(
        timestamp=timestamp,
        asks=[Level(price, qty) for price, qty in asks],
        bids=[Level(price, qty) for price, qty in bids],
    )


def make_crossable_snapshot_for_buy(timestamp: datetime) -> Snapshot:
    return make_two_level_snapshot(
        timestamp,
        asks=[(100.0, 3.0), (101.0, 4.0)],
        bids=[(99.0, 5.0), (98.0, 6.0)],
    )


def make_crossable_snapshot_for_sell(timestamp: datetime) -> Snapshot:
    return make_two_level_snapshot(
        timestamp,
        asks=[(101.0, 3.0), (102.0, 4.0)],
        bids=[(100.0, 5.0), (99.0, 6.0)],
    )

