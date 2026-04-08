from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.events import BaseEvent
from src.models import Order, Snapshot, Trade


@dataclass(slots=True)
class SimulationState:
    """Mutable simulation state owned by the engine."""

    current_snapshot: Snapshot | None = None
    active_orders: dict[str, Order] = field(default_factory=dict)
    completed_orders: dict[str, Order] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    event_log: list[BaseEvent] = field(default_factory=list)
    current_time: datetime | None = None
    order_sequence: int = 0
    trade_sequence: int = 0

    def register_order(self, order: Order) -> None:
        if order.is_done():
            self.completed_orders[order.order_id] = order
            self.active_orders.pop(order.order_id, None)
            return
        self.active_orders[order.order_id] = order

    def complete_order(self, order: Order) -> None:
        self.active_orders.pop(order.order_id, None)
        self.completed_orders[order.order_id] = order

    def append_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self.current_time = trade.timestamp

    def append_event(self, event: BaseEvent) -> None:
        self.event_log.append(event)
        self.current_time = event.timestamp

    def next_order_id(self) -> str:
        self.order_sequence += 1
        return f"ORD-{self.order_sequence:06d}"

    def next_trade_id(self) -> str:
        self.trade_sequence += 1
        return f"TRD-{self.trade_sequence:06d}"
