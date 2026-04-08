from __future__ import annotations

from src.actions import PlaceOrderAction
from src.enums import OrderType, Side
from src.risk_limits import StrategyLimits, check_action_against_limits
from src.strategy.state import StrategyState


def make_action(side: Side = Side.BUY, qty: float = 2.0, price: float = 100.0) -> PlaceOrderAction:
    return PlaceOrderAction(strategy_id="s-1", side=side, price=price, qty=qty, order_type=OrderType.LIMIT)


def test_order_above_max_qty_is_rejected() -> None:
    result = check_action_against_limits(
        make_action(qty=5.0),
        strategy_state=StrategyState(),
        active_orders={},
        limits=StrategyLimits(max_order_qty=4.0),
        step_index=1,
    )
    assert result.allowed is False


def test_order_above_max_notional_per_order_is_rejected() -> None:
    result = check_action_against_limits(
        make_action(qty=2.0, price=100.0),
        strategy_state=StrategyState(),
        active_orders={},
        limits=StrategyLimits(max_notional_per_order=150.0),
        step_index=1,
    )
    assert result.allowed is False


def test_order_above_max_active_orders_is_rejected() -> None:
    state = StrategyState()
    active_orders = {"o-1": object()}  # type: ignore[dict-item]
    result = check_action_against_limits(
        make_action(),
        strategy_state=state,
        active_orders=active_orders,
        limits=StrategyLimits(max_active_orders=1),
        step_index=1,
    )
    assert result.allowed is False


def test_allow_short_false_rejects_short_opening_sell() -> None:
    result = check_action_against_limits(
        make_action(side=Side.SELL, qty=1.0),
        strategy_state=StrategyState(),
        active_orders={},
        limits=StrategyLimits(allow_short=False),
        step_index=1,
    )
    assert result.allowed is False


def test_max_position_abs_rejects_large_position() -> None:
    result = check_action_against_limits(
        make_action(qty=3.0),
        strategy_state=StrategyState(),
        active_orders={},
        limits=StrategyLimits(max_position_abs=2.0),
        step_index=1,
    )
    assert result.allowed is False


def test_cooldown_blocks_new_order_after_trade() -> None:
    state = StrategyState(last_trade_step_index=1)
    result = check_action_against_limits(
        make_action(),
        strategy_state=state,
        active_orders={},
        limits=StrategyLimits(cooldown_steps_after_trade=2),
        step_index=2,
    )
    assert result.allowed is False

