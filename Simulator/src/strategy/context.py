from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.events import OrderUpdateEvent
from src.models import Order, Snapshot, Trade
from src.strategy.metrics import StrategyMetrics


@dataclass(slots=True)
class StrategyContext:
    """Snapshot of strategy-visible state for the current simulation step."""

    timestamp: datetime
    snapshot: Snapshot
    active_orders: dict[str, Order]
    new_trades: list[Trade]
    order_updates: list[OrderUpdateEvent]
    position: float
    cash: float
    realized_pnl: float | None = None
    unrealized_pnl: float | None = None
    equity: float | None = None
    metrics: StrategyMetrics | None = None

