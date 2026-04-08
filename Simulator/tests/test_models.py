from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.enums import LiquidityRole, OrderStatus, OrderType, Side
from src.models import Level, Order, Snapshot, Trade


def make_snapshot() -> Snapshot:
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    return Snapshot(
        timestamp=ts,
        asks=[Level(price=101.0, qty=2.0), Level(price=102.0, qty=1.0)],
        bids=[Level(price=99.0, qty=3.0), Level(price=98.5, qty=4.0)],
    )


def test_level_accepts_valid_values() -> None:
    level = Level(price=100.0, qty=1.5)
    assert level.price == 100.0
    assert level.qty == 1.5


def test_level_rejects_negative_qty() -> None:
    with pytest.raises(ValueError, match="qty must be >= 0"):
        Level(price=100.0, qty=-1.0)


def test_snapshot_best_bid() -> None:
    snapshot = make_snapshot()
    assert snapshot.best_bid() == Level(price=99.0, qty=3.0)


def test_snapshot_best_ask() -> None:
    snapshot = make_snapshot()
    assert snapshot.best_ask() == Level(price=101.0, qty=2.0)


def test_snapshot_mid_price() -> None:
    snapshot = make_snapshot()
    assert snapshot.mid_price() == 100.0


def test_order_helpers() -> None:
    order = Order(
        order_id="o-1",
        strategy_id="s-1",
        side=Side.BUY,
        price=100.0,
        qty=5.0,
        remaining_qty=2.0,
        order_type=OrderType.LIMIT,
        status=OrderStatus.PARTIALLY_FILLED,
        created_at=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
    )

    assert order.filled_qty() == 3.0
    assert order.is_active() is True
    assert order.is_done() is False


def test_trade_accepts_valid_values() -> None:
    trade = Trade(
        trade_id="t-1",
        order_id="o-1",
        strategy_id="s-1",
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        side=Side.SELL,
        price=101.5,
        qty=0.5,
        liquidity_role=LiquidityRole.MAKER,
    )

    assert trade.trade_id == "t-1"
    assert trade.liquidity_role == LiquidityRole.MAKER

