from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.actions import PlaceOrderAction
from src.engine import SimulationEngine
from src.enums import EventType, OrderStatus, OrderType, Side
from src.events import MarketSnapshotEvent, OrderUpdateEvent, OwnTradeEvent
from src.models import Level, Snapshot


def make_snapshot(ts: datetime, asks: list[tuple[float, float]], bids: list[tuple[float, float]]) -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(price, qty) for price, qty in asks],
        bids=[Level(price, qty) for price, qty in bids],
    )


def test_market_snapshot_event_created_each_step() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)

    engine.process_snapshot(make_snapshot(ts, [(101.0, 1.0)], [(100.0, 1.0)]))
    engine.process_snapshot(make_snapshot(ts + timedelta(seconds=1), [(101.5, 1.0)], [(100.5, 1.0)]))

    market_events = [event for event in engine.state.event_log if isinstance(event, MarketSnapshotEvent)]
    assert len(market_events) == 2


def test_active_order_creation_generates_order_update_event() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    events = engine.process_snapshot(
        make_snapshot(ts, [(105.0, 5.0)], [(104.0, 5.0)]),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=2.0, order_type=OrderType.LIMIT)],
    )

    updates = [event for event in events if isinstance(event, OrderUpdateEvent)]
    assert len(updates) == 1
    assert updates[0].new_status == OrderStatus.ACTIVE


def test_fill_generates_own_trade_event() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    events = engine.process_snapshot(
        make_snapshot(ts, [(100.0, 2.0)], [(99.0, 5.0)]),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=2.0, order_type=OrderType.FOK)],
    )

    assert any(isinstance(event, OwnTradeEvent) for event in events)


def test_multi_level_fill_creates_multiple_trades() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    events = engine.process_snapshot(
        make_snapshot(ts, [(100.0, 3.0), (101.0, 4.0)], [(99.0, 5.0)]),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=101.0, qty=7.0, order_type=OrderType.FOK)],
    )

    own_trades = [event for event in events if isinstance(event, OwnTradeEvent)]
    assert len(own_trades) == 2


def test_status_events_added_to_event_log() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(
        make_snapshot(ts, [(105.0, 5.0)], [(104.0, 5.0)]),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=2.0, order_type=OrderType.LIMIT)],
    )

    assert any(isinstance(event, OrderUpdateEvent) for event in engine.state.event_log)


def test_event_order_is_deterministic() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    events = engine.process_snapshot(
        make_snapshot(ts, [(100.0, 3.0), (101.0, 4.0)], [(99.0, 5.0)]),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=101.0, qty=7.0, order_type=OrderType.FOK)],
    )

    assert [event.event_type for event in events] == [
        EventType.MARKET_SNAPSHOT,
        EventType.OWN_TRADE,
        EventType.OWN_TRADE,
        EventType.ORDER_UPDATE,
    ]


def test_active_orders_processed_in_created_at_then_order_id_order() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(
        make_snapshot(ts, [(110.0, 5.0)], [(109.0, 5.0)]),
        [
            PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=1.0, order_type=OrderType.LIMIT),
            PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=1.0, order_type=OrderType.LIMIT),
        ],
    )

    events = engine.process_snapshot(
        make_snapshot(ts + timedelta(seconds=1), [(100.0, 2.0)], [(99.0, 5.0)]),
    )

    own_trades = [event for event in events if isinstance(event, OwnTradeEvent)]
    assert [event.trade.order_id for event in own_trades] == ["ORD-000001", "ORD-000002"]
