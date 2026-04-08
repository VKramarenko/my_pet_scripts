from __future__ import annotations

from datetime import UTC, datetime

from src.enums import EventType, LiquidityRole, OrderStatus, Side
from src.events import MarketSnapshotEvent, OrderUpdateEvent, OwnTradeEvent
from src.models import Level, Snapshot, Trade


def test_market_snapshot_event_is_created_correctly() -> None:
    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        asks=[Level(price=101.0, qty=1.0)],
        bids=[Level(price=100.0, qty=1.0)],
    )

    event = MarketSnapshotEvent(timestamp=snapshot.timestamp, snapshot=snapshot)

    assert event.event_type == EventType.MARKET_SNAPSHOT
    assert event.snapshot is snapshot
    assert event.timestamp == snapshot.timestamp


def test_order_update_event_stores_statuses() -> None:
    event = OrderUpdateEvent(
        timestamp=datetime(2026, 4, 7, 12, 1, tzinfo=UTC),
        order_id="o-1",
        strategy_id="s-1",
        old_status=OrderStatus.NEW,
        new_status=OrderStatus.ACTIVE,
        remaining_qty=1.0,
    )

    assert event.event_type == EventType.ORDER_UPDATE
    assert event.old_status == OrderStatus.NEW
    assert event.new_status == OrderStatus.ACTIVE


def test_own_trade_event_wraps_trade() -> None:
    trade = Trade(
        trade_id="t-1",
        order_id="o-1",
        strategy_id="s-1",
        timestamp=datetime(2026, 4, 7, 12, 2, tzinfo=UTC),
        side=Side.BUY,
        price=100.5,
        qty=0.25,
        liquidity_role=LiquidityRole.UNKNOWN,
    )

    event = OwnTradeEvent(timestamp=trade.timestamp, trade=trade)

    assert event.event_type == EventType.OWN_TRADE
    assert event.trade is trade
    assert event.timestamp == trade.timestamp

