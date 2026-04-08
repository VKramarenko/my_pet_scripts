from __future__ import annotations

from datetime import datetime

from src.events import OrderUpdateEvent
from src.models import Order, Snapshot, Trade
from src.strategy.context import StrategyContext
from src.strategy.metrics import StrategyMetrics


def make_strategy_context(
    timestamp: datetime,
    snapshot: Snapshot,
    *,
    active_orders: dict[str, Order] | None = None,
    new_trades: list[Trade] | None = None,
    order_updates: list[OrderUpdateEvent] | None = None,
    position: float = 0.0,
    cash: float = 0.0,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    equity: float = 0.0,
    metrics: StrategyMetrics | None = None,
) -> StrategyContext:
    return StrategyContext(
        timestamp=timestamp,
        snapshot=snapshot,
        active_orders=dict(active_orders or {}),
        new_trades=list(new_trades or []),
        order_updates=list(order_updates or []),
        position=position,
        cash=cash,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        equity=equity,
        metrics=metrics or StrategyMetrics(),
    )


def make_empty_strategy_context(timestamp: datetime, snapshot: Snapshot) -> StrategyContext:
    return make_strategy_context(timestamp, snapshot)


def make_context_with_active_orders(timestamp: datetime, snapshot: Snapshot, active_orders: dict[str, Order]) -> StrategyContext:
    return make_strategy_context(timestamp, snapshot, active_orders=active_orders)


def make_context_with_trades(timestamp: datetime, snapshot: Snapshot, trades: list[Trade]) -> StrategyContext:
    return make_strategy_context(timestamp, snapshot, new_trades=trades)

