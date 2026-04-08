from __future__ import annotations

from dataclasses import dataclass

from src.actions import PlaceOrderAction
from src.enums import Side
from src.models import Order
from src.strategy.state import StrategyState


@dataclass(slots=True)
class StrategyLimits:
    max_position_abs: float | None = None
    max_active_orders: int | None = None
    max_order_qty: float | None = None
    max_notional_per_order: float | None = None
    max_notional_total: float | None = None
    allow_short: bool = True
    cooldown_steps_after_trade: int = 0


@dataclass(slots=True)
class LimitCheckResult:
    allowed: bool
    reason: str | None = None


def check_action_against_limits(
    action: PlaceOrderAction,
    *,
    strategy_state: StrategyState,
    active_orders: dict[str, Order],
    limits: StrategyLimits,
    step_index: int,
) -> LimitCheckResult:
    if limits.max_order_qty is not None and action.qty > limits.max_order_qty:
        return LimitCheckResult(False, "max_order_qty exceeded")

    order_notional = action.price * action.qty
    if limits.max_notional_per_order is not None and order_notional > limits.max_notional_per_order:
        return LimitCheckResult(False, "max_notional_per_order exceeded")

    if limits.max_active_orders is not None and len(active_orders) + 1 > limits.max_active_orders:
        return LimitCheckResult(False, "max_active_orders exceeded")

    if not limits.allow_short and action.side == Side.SELL and strategy_state.position < action.qty:
        return LimitCheckResult(False, "short selling is not allowed")

    projected_position = strategy_state.position + action.qty if action.side == Side.BUY else strategy_state.position - action.qty
    if limits.max_position_abs is not None and abs(projected_position) > limits.max_position_abs:
        return LimitCheckResult(False, "max_position_abs exceeded")

    existing_notional = abs(strategy_state.position * (strategy_state.last_mid_price or 0.0)) + sum(
        abs(order.price * order.remaining_qty) for order in active_orders.values()
    )
    if limits.max_notional_total is not None and existing_notional + order_notional > limits.max_notional_total:
        return LimitCheckResult(False, "max_notional_total exceeded")

    if (
        limits.cooldown_steps_after_trade > 0
        and strategy_state.last_trade_step_index is not None
        and step_index - strategy_state.last_trade_step_index <= limits.cooldown_steps_after_trade
    ):
        return LimitCheckResult(False, "cooldown is active")

    return LimitCheckResult(True)

