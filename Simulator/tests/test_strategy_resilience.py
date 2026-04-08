from __future__ import annotations

from src.enums import OrderStatus
from tests.helpers.contexts import make_empty_strategy_context
from tests.helpers.events import make_order_update_event, make_trade_event
from tests.helpers.snapshots import make_snapshot
from tests.helpers.strategies import CancelStaleOrderStrategy, NoOpStrategy


def test_strategy_does_not_fail_without_new_trades(base_time) -> None:
    strategy = NoOpStrategy("s-1")
    assert strategy.on_snapshot(make_empty_strategy_context(base_time, make_snapshot(base_time))) == []


def test_strategy_does_not_fail_without_order_updates(base_time) -> None:
    strategy = NoOpStrategy("s-1")
    strategy.on_simulation_start()
    assert strategy.on_snapshot(make_empty_strategy_context(base_time, make_snapshot(base_time))) == []


def test_strategy_does_not_fail_with_no_active_orders(base_time) -> None:
    strategy = CancelStaleOrderStrategy("s-1")
    actions = strategy.on_snapshot(make_empty_strategy_context(base_time, make_snapshot(base_time)))
    assert isinstance(actions, list)


def test_strategy_handles_multiple_events_on_one_step(base_time) -> None:
    strategy = NoOpStrategy("s-1")
    strategy.on_order_update(make_order_update_event(base_time, order_id="o-1", new_status=OrderStatus.ACTIVE))
    strategy.on_order_update(make_order_update_event(base_time, order_id="o-2", new_status=OrderStatus.ACTIVE))
    strategy.on_trade(make_trade_event(base_time, order_id="o-1", qty=1.0))
    strategy.on_trade(make_trade_event(base_time, order_id="o-1", trade_id="t-2", qty=2.0))
    assert len(strategy.state.orders) == 2
    assert len(strategy.state.trades) == 2


def test_strategy_handles_partial_fill_then_cancel(base_time) -> None:
    strategy = NoOpStrategy("s-1")
    strategy.on_order_update(make_order_update_event(base_time, order_id="o-1", new_status=OrderStatus.ACTIVE, remaining_qty=3.0))
    strategy.on_trade(make_trade_event(base_time, order_id="o-1", qty=1.0))
    strategy.on_order_update(make_order_update_event(base_time, order_id="o-1", old_status=OrderStatus.ACTIVE, new_status=OrderStatus.PARTIALLY_FILLED, remaining_qty=2.0))
    strategy.on_order_update(make_order_update_event(base_time, order_id="o-1", old_status=OrderStatus.PARTIALLY_FILLED, new_status=OrderStatus.CANCELED, remaining_qty=2.0))
    assert "o-1" not in strategy.state.active_orders


def test_strategy_handles_long_sequence_without_trades(base_time) -> None:
    strategy = NoOpStrategy("s-1")
    for _ in range(10):
        strategy.on_snapshot(make_empty_strategy_context(base_time, make_snapshot(base_time)))
    assert strategy.state.trades == []

