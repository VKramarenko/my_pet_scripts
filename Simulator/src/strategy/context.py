from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.events import BaseEvent, OrderUpdateEvent
from src.models import Order, Snapshot, Trade
from src.strategy.metrics import StrategyMetrics


@dataclass(slots=True)
class StrategyContext:
    """Snapshot of strategy-visible state for the current simulation step."""

    timestamp: datetime
    snapshot: Snapshot | None          # primary/trigger snapshot; None for non-snapshot events
    active_orders: dict[str, Order]
    new_trades: list[Trade]
    order_updates: list[OrderUpdateEvent]
    position: float                    # legacy: position for "default" instrument
    cash: float
    realized_pnl: float | None = None
    unrealized_pnl: float | None = None
    equity: float | None = None
    metrics: StrategyMetrics | None = None
    snapshots: dict[str, Snapshot] = field(default_factory=dict)  # all current snapshots by instrument_id
    positions: dict[str, float] = field(default_factory=dict)     # position per instrument_id
    triggering_event: BaseEvent | None = None                     # event that caused this tick
