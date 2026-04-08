from __future__ import annotations

from datetime import UTC, datetime

from src.enums import OrderStatus, OrderType, Side
from src.matching import execute_limit_against_snapshot
from src.models import Level, Order, Snapshot


def make_snapshot() -> Snapshot:
    return Snapshot(
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        asks=[Level(100.0, 3.0), Level(101.0, 4.0), Level(102.0, 10.0)],
        bids=[Level(99.0, 5.0), Level(98.0, 6.0), Level(97.0, 7.0)],
    )


def make_order(side: Side, price: float, qty: float) -> Order:
    return Order(
        order_id="o-1",
        strategy_id="s-1",
        side=side,
        price=price,
        qty=qty,
        remaining_qty=qty,
        order_type=OrderType.LIMIT,
        status=OrderStatus.NEW,
        created_at=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
    )


def test_buy_limit_fills_on_single_level() -> None:
    trades, remaining_qty = execute_limit_against_snapshot(make_order(Side.BUY, 100.0, 2.0), make_snapshot())
    assert [(trade.price, trade.qty) for trade in trades] == [(100.0, 2.0)]
    assert remaining_qty == 0.0


def test_buy_limit_fills_across_multiple_levels() -> None:
    trades, remaining_qty = execute_limit_against_snapshot(make_order(Side.BUY, 101.0, 7.0), make_snapshot())
    assert [(trade.price, trade.qty) for trade in trades] == [(100.0, 3.0), (101.0, 4.0)]
    assert remaining_qty == 0.0


def test_buy_limit_partially_fills_and_leaves_tail() -> None:
    trades, remaining_qty = execute_limit_against_snapshot(make_order(Side.BUY, 101.0, 10.0), make_snapshot())
    assert [(trade.price, trade.qty) for trade in trades] == [(100.0, 3.0), (101.0, 4.0)]
    assert remaining_qty == 3.0


def test_buy_limit_in_depth_stays_active() -> None:
    trades, remaining_qty = execute_limit_against_snapshot(make_order(Side.BUY, 99.0, 2.0), make_snapshot())
    assert trades == []
    assert remaining_qty == 2.0


def test_sell_limit_fills_immediately() -> None:
    trades, remaining_qty = execute_limit_against_snapshot(make_order(Side.SELL, 99.0, 4.0), make_snapshot())
    assert [(trade.price, trade.qty) for trade in trades] == [(99.0, 4.0)]
    assert remaining_qty == 0.0


def test_sell_limit_partially_fills() -> None:
    trades, remaining_qty = execute_limit_against_snapshot(make_order(Side.SELL, 98.0, 12.0), make_snapshot())
    assert [(trade.price, trade.qty) for trade in trades] == [(99.0, 5.0), (98.0, 6.0)]
    assert remaining_qty == 1.0


def test_sell_limit_in_depth_stays_active() -> None:
    trades, remaining_qty = execute_limit_against_snapshot(make_order(Side.SELL, 100.0, 2.0), make_snapshot())
    assert trades == []
    assert remaining_qty == 2.0

