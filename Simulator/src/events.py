from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.enums import EventType, OrderStatus
from src.models import Snapshot, Trade


@dataclass(slots=True)
class BaseEvent:
    """Base event passed between engine and strategy boundary."""

    event_type: EventType
    timestamp: datetime


@dataclass(slots=True)
class MarketSnapshotEvent(BaseEvent):
    snapshot: Snapshot
    event_type: EventType = field(init=False, default=EventType.MARKET_SNAPSHOT)

    def __post_init__(self) -> None:
        self.timestamp = self.snapshot.timestamp


@dataclass(slots=True)
class OrderUpdateEvent(BaseEvent):
    order_id: str
    strategy_id: str
    old_status: OrderStatus | None
    new_status: OrderStatus
    filled_qty_delta: float = 0.0
    remaining_qty: float | None = None
    reason: str | None = None
    event_type: EventType = field(init=False, default=EventType.ORDER_UPDATE)

    def __post_init__(self) -> None:
        if self.filled_qty_delta < 0:
            raise ValueError("filled_qty_delta must be >= 0")
        if self.remaining_qty is not None and self.remaining_qty < 0:
            raise ValueError("remaining_qty must be >= 0")


@dataclass(slots=True)
class OwnTradeEvent(BaseEvent):
    trade: Trade
    event_type: EventType = field(init=False, default=EventType.OWN_TRADE)

    def __post_init__(self) -> None:
        self.timestamp = self.trade.timestamp

