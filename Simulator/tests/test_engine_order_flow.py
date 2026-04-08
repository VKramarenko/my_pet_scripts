from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.actions import CancelOrderAction, ModifyOrderAction, PlaceOrderAction
from src.engine import SimulationEngine
from src.enums import OrderStatus, OrderType, Side
from src.events import OwnTradeEvent
from src.models import Level, Snapshot


def make_snapshot(ts: datetime, best_ask: float, best_bid: float, ask_qty: float = 5.0, bid_qty: float = 5.0) -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(best_ask, ask_qty), Level(best_ask + 1.0, 10.0)],
        bids=[Level(best_bid, bid_qty), Level(best_bid - 1.0, 10.0)],
    )


def test_resting_orders_processed_before_strategy_actions() -> None:
    engine = SimulationEngine()
    ts1 = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    ts2 = ts1 + timedelta(seconds=1)

    engine.process_snapshot(
        make_snapshot(ts1, best_ask=105.0, best_bid=104.0),
        [
            PlaceOrderAction(
                strategy_id="s-1",
                side=Side.BUY,
                price=100.0,
                qty=3.0,
                order_type=OrderType.LIMIT,
            )
        ],
    )

    events = engine.process_snapshot(
        make_snapshot(ts2, best_ask=100.0, best_bid=99.0, ask_qty=3.0),
        [
            PlaceOrderAction(
                strategy_id="s-1",
                side=Side.BUY,
                price=110.0,
                qty=1.0,
                order_type=OrderType.FOK,
            )
        ],
    )

    own_trades = [event for event in events if isinstance(event, OwnTradeEvent)]
    assert len(own_trades) == 2
    assert own_trades[0].trade.order_id == "ORD-000001"
    assert own_trades[1].trade.order_id == "ORD-000002"


def test_old_active_limit_fills_on_new_snapshot() -> None:
    engine = SimulationEngine()
    ts1 = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    ts2 = ts1 + timedelta(seconds=1)

    engine.process_snapshot(
        make_snapshot(ts1, best_ask=105.0, best_bid=104.0),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=3.0, order_type=OrderType.LIMIT)],
    )
    engine.process_snapshot(make_snapshot(ts2, best_ask=100.0, best_bid=99.0, ask_qty=3.0))

    assert "ORD-000001" not in engine.state.active_orders
    assert engine.state.completed_orders["ORD-000001"].status == OrderStatus.FILLED


def test_partially_filled_resting_order_stays_active() -> None:
    engine = SimulationEngine()
    ts1 = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    ts2 = ts1 + timedelta(seconds=1)

    engine.process_snapshot(
        make_snapshot(ts1, best_ask=105.0, best_bid=104.0),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=5.0, order_type=OrderType.LIMIT)],
    )
    engine.process_snapshot(make_snapshot(ts2, best_ask=100.0, best_bid=99.0, ask_qty=2.0))

    assert engine.state.active_orders["ORD-000001"].status == OrderStatus.PARTIALLY_FILLED
    assert engine.state.active_orders["ORD-000001"].remaining_qty == 3.0


def test_fully_filled_resting_order_removed_from_active_orders() -> None:
    engine = SimulationEngine()
    ts1 = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    ts2 = ts1 + timedelta(seconds=1)

    engine.process_snapshot(
        make_snapshot(ts1, best_ask=105.0, best_bid=104.0),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=2.0, order_type=OrderType.LIMIT)],
    )
    engine.process_snapshot(make_snapshot(ts2, best_ask=100.0, best_bid=99.0, ask_qty=2.0))

    assert "ORD-000001" not in engine.state.active_orders


def test_cancel_order_action_removes_active_order() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(
        make_snapshot(ts, best_ask=105.0, best_bid=104.0),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=2.0, order_type=OrderType.LIMIT)],
    )

    engine.process_snapshot(
        make_snapshot(ts + timedelta(seconds=1), best_ask=105.0, best_bid=104.0),
        [CancelOrderAction(strategy_id="s-1", order_id="ORD-000001")],
    )

    assert "ORD-000001" not in engine.state.active_orders
    assert engine.state.completed_orders["ORD-000001"].status == OrderStatus.CANCELED


def test_cancel_ignores_already_completed_order() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(
        make_snapshot(ts, best_ask=100.0, best_bid=99.0, ask_qty=2.0),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=2.0, order_type=OrderType.FOK)],
    )

    events = engine.process_snapshot(
        make_snapshot(ts + timedelta(seconds=1), best_ask=101.0, best_bid=100.0),
        [CancelOrderAction(strategy_id="s-1", order_id="ORD-000001")],
    )

    assert events[0].timestamp == ts + timedelta(seconds=1)
    assert engine.state.completed_orders["ORD-000001"].status == OrderStatus.FILLED


def test_modify_order_works_as_cancel_replace() -> None:
    engine = SimulationEngine()
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(
        make_snapshot(ts, best_ask=105.0, best_bid=104.0),
        [PlaceOrderAction(strategy_id="s-1", side=Side.BUY, price=100.0, qty=5.0, order_type=OrderType.LIMIT)],
    )

    engine.process_snapshot(
        make_snapshot(ts + timedelta(seconds=1), best_ask=105.0, best_bid=104.0),
        [ModifyOrderAction(strategy_id="s-1", order_id="ORD-000001", new_price=101.0, new_qty=4.0)],
    )

    assert engine.state.completed_orders["ORD-000001"].status == OrderStatus.CANCELED
    assert "ORD-000002" in engine.state.active_orders
    assert engine.state.active_orders["ORD-000002"].price == 101.0

