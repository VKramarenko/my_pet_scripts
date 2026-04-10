from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.enums import LiquidityRole, OrderStatus, OrderType, Side


@dataclass(slots=True)
class Level:
    """Single book level."""

    price: float
    qty: float

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("price must be > 0")
        if self.qty < 0:
            raise ValueError("qty must be >= 0")


@dataclass(slots=True)
class Snapshot:
    """Full order book snapshot at a point in time."""

    timestamp: datetime
    asks: list[Level]
    bids: list[Level]
    instrument_id: str = "default"
    validate_crossed_book: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.asks != sorted(self.asks, key=lambda level: level.price):
            raise ValueError("asks must be sorted ascending by price")
        if self.bids != sorted(self.bids, key=lambda level: level.price, reverse=True):
            raise ValueError("bids must be sorted descending by price")
        if (
            self.validate_crossed_book
            and self.asks
            and self.bids
            and self.bids[0].price >= self.asks[0].price
        ):
            raise ValueError("best_bid must be < best_ask")

    def best_bid(self) -> Level | None:
        return self.bids[0] if self.bids else None

    def best_ask(self) -> Level | None:
        return self.asks[0] if self.asks else None

    def mid_price(self) -> float | None:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid.price + best_ask.price) / 2.0


@dataclass(slots=True)
class Order:
    """Strategy order stored and tracked by the simulation engine."""

    order_id: str
    strategy_id: str
    side: Side
    price: float
    qty: float
    remaining_qty: float
    order_type: OrderType
    status: OrderStatus
    created_at: datetime
    updated_at: datetime | None = None
    instrument_id: str = "default"

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("price must be > 0")
        if self.qty <= 0:
            raise ValueError("qty must be > 0")
        if self.remaining_qty < 0:
            raise ValueError("remaining_qty must be >= 0")
        if self.remaining_qty > self.qty:
            raise ValueError("remaining_qty must be <= qty")

    def is_active(self) -> bool:
        return self.status in {OrderStatus.NEW, OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}

    def is_done(self) -> bool:
        return self.status in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED}

    def filled_qty(self) -> float:
        return self.qty - self.remaining_qty

    def mark_updated(self, ts: datetime) -> None:
        self.updated_at = ts


@dataclass(slots=True)
class Trade:
    """Own trade generated for a strategy order."""

    trade_id: str
    order_id: str
    strategy_id: str
    timestamp: datetime
    side: Side
    price: float
    qty: float
    liquidity_role: LiquidityRole = LiquidityRole.UNKNOWN
    raw_price: float | None = None
    commission: float = 0.0
    notional: float | None = None
    instrument_id: str = "default"

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("price must be > 0")
        if self.qty <= 0:
            raise ValueError("qty must be > 0")
        if self.commission < 0:
            raise ValueError("commission must be >= 0")
        if self.raw_price is not None and self.raw_price <= 0:
            raise ValueError("raw_price must be > 0")
        if self.notional is None:
            self.notional = self.price * self.qty
