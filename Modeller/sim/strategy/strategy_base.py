from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from sim.core.events import MarketSnapshot
from sim.exchange.exchange_sim import PlaceOrderRequest
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2


@dataclass(frozen=True)
class CancelRequest:
    order_id: str


class StrategyBase(ABC):
    @abstractmethod
    def on_snapshot(
        self,
        ts: float,
        book: OrderBookL2,
        active_orders: Iterable[Order],
    ) -> list[PlaceOrderRequest | CancelRequest]:
        raise NotImplementedError

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        raise NotImplementedError

