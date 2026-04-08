from __future__ import annotations

from src.actions import CancelOrderAction, ModifyOrderAction, PlaceOrderAction
from tests.helpers.contexts import make_empty_strategy_context
from tests.helpers.orders import make_active_order
from tests.helpers.snapshots import make_snapshot
from tests.helpers.strategies import AlwaysPlaceBuyLimitStrategy, CancelStaleOrderStrategy, RepriceStrategy, ThresholdStrategy


def test_strategy_returns_place_order_when_entry_condition_met(base_time) -> None:
    strategy = ThresholdStrategy("s-1", threshold=101.0, qty=1.0)
    context = make_empty_strategy_context(base_time, make_snapshot(base_time, best_ask=101.0, best_bid=100.0))
    actions = strategy.on_snapshot(context)
    assert len(actions) == 1
    assert isinstance(actions[0], PlaceOrderAction)


def test_strategy_returns_no_actions_when_entry_condition_not_met(base_time) -> None:
    strategy = ThresholdStrategy("s-1", threshold=100.0, qty=1.0)
    context = make_empty_strategy_context(base_time, make_snapshot(base_time, best_ask=101.0, best_bid=100.0))
    assert strategy.on_snapshot(context) == []


def test_strategy_does_not_duplicate_when_active_order_exists(base_time) -> None:
    strategy = AlwaysPlaceBuyLimitStrategy("s-1")
    strategy.state.active_orders["o-1"] = make_active_order(base_time)
    context = make_empty_strategy_context(base_time, make_snapshot(base_time))
    assert strategy.on_snapshot(context) == []


def test_cancel_stale_strategy_returns_cancel_action(base_time) -> None:
    strategy = CancelStaleOrderStrategy("s-1", stale_after_steps=1)
    strategy.state.orders["o-1"] = make_active_order(base_time)
    strategy.state.active_orders["o-1"] = make_active_order(base_time)
    strategy.step_index = 1
    actions = strategy.on_snapshot(make_empty_strategy_context(base_time, make_snapshot(base_time)))
    assert len(actions) == 1
    assert isinstance(actions[0], CancelOrderAction)


def test_reprice_strategy_returns_modify_action(base_time) -> None:
    strategy = RepriceStrategy("s-1", new_price=99.0)
    strategy.state.active_orders["o-1"] = make_active_order(base_time)
    actions = strategy.on_snapshot(make_empty_strategy_context(base_time, make_snapshot(base_time)))
    assert len(actions) == 1
    assert isinstance(actions[0], ModifyOrderAction)

