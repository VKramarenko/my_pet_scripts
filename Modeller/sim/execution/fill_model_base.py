from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from sim.core.events import MarketSnapshot, MarketTrade
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2


class IFillModel(ABC):
    @abstractmethod
    def on_snapshot(
        self,
        book: OrderBookL2,
        snapshot: MarketSnapshot,
        active_orders: Iterable[Order],
        ts: float,
    ) -> list[Fill]:
        raise NotImplementedError

    @abstractmethod
    def on_trade(
        self,
        book: OrderBookL2,
        trade: MarketTrade,
        active_orders: Iterable[Order],
        ts: float,
    ) -> list[Fill]:
        raise NotImplementedError

