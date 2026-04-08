from __future__ import annotations

from datetime import UTC, datetime

from src.actions import BaseAction
from src.enums import OrderStatus
from src.events import OrderUpdateEvent
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext


class NoopStrategy(BaseStrategy):
    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        return []


def test_base_strategy_initializes_empty_state() -> None:
    strategy = NoopStrategy("s-1")
    assert strategy.state.orders == {}
    assert strategy.state.active_orders == {}
    assert strategy.state.trades == []


def test_on_simulation_start_resets_state() -> None:
    strategy = NoopStrategy("s-1")
    strategy.state.cash = 10.0
    strategy.on_simulation_start()
    assert strategy.state.cash == 0.0
    assert strategy.state.orders == {}


def test_on_order_update_adds_order() -> None:
    strategy = NoopStrategy("s-1")
    event = OrderUpdateEvent(
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        order_id="o-1",
        strategy_id="s-1",
        old_status=None,
        new_status=OrderStatus.ACTIVE,
        remaining_qty=2.0,
    )
    strategy.on_order_update(event)
    assert "o-1" in strategy.state.orders
    assert "o-1" in strategy.state.active_orders


def test_on_order_update_updates_existing_status() -> None:
    strategy = NoopStrategy("s-1")
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    strategy.on_order_update(
        OrderUpdateEvent(timestamp=ts, order_id="o-1", strategy_id="s-1", old_status=None, new_status=OrderStatus.ACTIVE, remaining_qty=2.0)
    )
    strategy.on_order_update(
        OrderUpdateEvent(timestamp=ts, order_id="o-1", strategy_id="s-1", old_status=OrderStatus.ACTIVE, new_status=OrderStatus.PARTIALLY_FILLED, remaining_qty=1.0)
    )
    assert strategy.state.orders["o-1"].status == OrderStatus.PARTIALLY_FILLED


def test_filled_order_removed_from_active_orders() -> None:
    strategy = NoopStrategy("s-1")
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    strategy.on_order_update(
        OrderUpdateEvent(timestamp=ts, order_id="o-1", strategy_id="s-1", old_status=None, new_status=OrderStatus.ACTIVE, remaining_qty=2.0)
    )
    strategy.on_order_update(
        OrderUpdateEvent(timestamp=ts, order_id="o-1", strategy_id="s-1", old_status=OrderStatus.ACTIVE, new_status=OrderStatus.FILLED, remaining_qty=0.0)
    )
    assert "o-1" not in strategy.state.active_orders


def test_canceled_order_removed_from_active_orders() -> None:
    strategy = NoopStrategy("s-1")
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    strategy.on_order_update(
        OrderUpdateEvent(timestamp=ts, order_id="o-1", strategy_id="s-1", old_status=None, new_status=OrderStatus.ACTIVE, remaining_qty=2.0)
    )
    strategy.on_order_update(
        OrderUpdateEvent(timestamp=ts, order_id="o-1", strategy_id="s-1", old_status=OrderStatus.ACTIVE, new_status=OrderStatus.CANCELED, remaining_qty=2.0)
    )
    assert "o-1" not in strategy.state.active_orders

