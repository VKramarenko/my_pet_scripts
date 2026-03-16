from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias


BookLevel = tuple[float, float]


@dataclass(frozen=True)
class MarketSnapshot:
    ts: float
    bids: Sequence[BookLevel]
    asks: Sequence[BookLevel]


@dataclass(frozen=True)
class MarketTrade:
    ts: float
    price: float
    size: float
    side: Literal["buyer_initiated", "seller_initiated"]


@dataclass(frozen=True)
class TimerEvent:
    ts: float


Event: TypeAlias = MarketSnapshot | MarketTrade | TimerEvent

