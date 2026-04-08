from __future__ import annotations

from datetime import UTC, datetime

from src.actions import PlaceOrderAction
from src.enums import OrderStatus
from src.events import OrderUpdateEvent
from src.models import Level, Snapshot, Trade
from src.strategy.context import StrategyContext
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy


def make_context(strategy: PassiveBuyOnceStrategy) -> StrategyContext:
    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        asks=[Level(101.0, 1.0)],
        bids=[Level(100.0, 1.0)],
    )
    return StrategyContext(
        timestamp=snapshot.timestamp,
        snapshot=snapshot,
        active_orders={},
        new_trades=[],
        order_updates=[],
        position=strategy.state.position,
        cash=strategy.state.cash,
        metrics=strategy.state.metrics,
    )


def test_passive_buy_once_places_order_on_first_snapshot() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 99.0, "qty": 2.0})
    actions = strategy.on_snapshot(make_context(strategy))
    assert len(actions) == 1
    assert isinstance(actions[0], PlaceOrderAction)


def test_strategy_does_not_place_duplicate_when_active_order_exists() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 99.0, "qty": 2.0})
    strategy.on_order_update(
        OrderUpdateEvent(
            timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
            order_id="o-1",
            strategy_id="s-1",
            old_status=None,
            new_status=OrderStatus.ACTIVE,
            remaining_qty=2.0,
        )
    )
    assert strategy.on_snapshot(make_context(strategy)) == []


def test_strategy_does_not_fail_after_fill() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 99.0, "qty": 2.0})
    strategy.state.position = 2.0
    assert strategy.on_snapshot(make_context(strategy)) == []


def test_strategy_uses_local_state_not_engine_reference() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 99.0, "qty": 2.0})
    strategy.on_order_update(
        OrderUpdateEvent(
            timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
            order_id="o-1",
            strategy_id="s-1",
            old_status=None,
            new_status=OrderStatus.ACTIVE,
            remaining_qty=2.0,
        )
    )
    context = make_context(strategy)
    context.active_orders.clear()
    assert strategy.has_active_orders() is True

