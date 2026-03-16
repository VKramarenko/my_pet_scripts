from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["LIMIT", "MARKET"]
OrderStatus = Literal["NEW", "ACKED", "PARTIAL", "FILLED", "CANCELED", "REJECTED"]
Liquidity = Literal["MAKER", "TAKER"]


@dataclass
class Order:
    id: str
    side: OrderSide
    type: OrderType
    price: float | None
    qty: float
    remaining: float
    tif: str
    created_ts: float
    status: OrderStatus = "NEW"
    effective_ts: float = 0.0


@dataclass(frozen=True)
class Ack:
    order_id: str
    ts: float


@dataclass(frozen=True)
class CancelAck:
    order_id: str
    ts: float


@dataclass(frozen=True)
class Reject:
    order_id: str | None
    ts: float
    reason: str


@dataclass(frozen=True)
class Fill:
    order_id: str
    ts: float
    price: float
    qty: float
    fee: float
    liquidity: Liquidity

