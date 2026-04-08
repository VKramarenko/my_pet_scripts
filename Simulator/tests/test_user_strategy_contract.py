from __future__ import annotations

from src.actions import PlaceOrderAction
from src.strategy.base import BaseStrategy
from tests.helpers.contexts import make_empty_strategy_context
from tests.helpers.snapshots import make_snapshot


class UserStrategyExample(BaseStrategy):
    def on_snapshot(self, context):
        if not self.has_active_orders():
            return [PlaceOrderAction(strategy_id=self.strategy_id, side=context.snapshot.best_bid() and context.snapshot.best_bid().price and __import__("src.enums").enums.Side.BUY, price=100.0, qty=1.0, order_type=__import__("src.enums").enums.OrderType.LIMIT)]
        return []


def test_user_strategy_returns_actions_instead_of_mutating_engine(base_time) -> None:
    strategy = UserStrategyExample("s-1")
    context = make_empty_strategy_context(base_time, make_snapshot(base_time))
    actions = strategy.on_snapshot(context)
    assert len(actions) == 1
    assert isinstance(actions[0], PlaceOrderAction)


def test_no_action_is_expressed_by_empty_list(base_time) -> None:
    strategy = UserStrategyExample("s-1")
    strategy.state.active_orders["o-1"] = object()  # type: ignore[assignment]
    context = make_empty_strategy_context(base_time, make_snapshot(base_time))
    assert strategy.on_snapshot(context) == []


def test_strategy_can_safely_use_local_state_fields(base_time) -> None:
    strategy = UserStrategyExample("s-1")
    strategy.state.position = 1.0
    strategy.state.cash = -100.0
    strategy.state.equity = 10.0
    context = make_empty_strategy_context(base_time, make_snapshot(base_time))
    assert isinstance(strategy.on_snapshot(context), list)

