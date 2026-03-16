from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sim.exchange.exchange_sim import PlaceOrderRequest
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2
from sim.strategy.strategy_base import CancelRequest, StrategyBase


@dataclass
class BasicMMStrategy(StrategyBase):
    spread: float = 1.0
    quote_size: float = 1.0
    _counter: int = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def on_snapshot(
        self,
        ts: float,
        book: OrderBookL2,
        active_orders: Iterable[Order],
    ) -> list[PlaceOrderRequest | CancelRequest]:
        mid = book.mid()
        if mid is None:
            return []
        active = list(active_orders)
        orders: list[PlaceOrderRequest | CancelRequest] = []
        if len(active) > 2:
            for order in active[2:]:
                orders.append(CancelRequest(order.id))
        buy_exists = any(o.side == "BUY" for o in active)
        sell_exists = any(o.side == "SELL" for o in active)
        half = self.spread / 2.0
        if not buy_exists:
            orders.append(
                PlaceOrderRequest(
                    order_id=self._next_id("buy"),
                    side="BUY",
                    type="LIMIT",
                    price=mid - half,
                    qty=self.quote_size,
                )
            )
        if not sell_exists:
            orders.append(
                PlaceOrderRequest(
                    order_id=self._next_id("sell"),
                    side="SELL",
                    type="LIMIT",
                    price=mid + half,
                    qty=self.quote_size,
                )
            )
        return orders

    def on_fill(self, fill: Fill) -> None:
        return None

