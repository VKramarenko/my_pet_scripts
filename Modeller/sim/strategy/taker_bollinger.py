from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable

from sim.exchange.exchange_sim import PlaceOrderRequest
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2
from sim.strategy.strategy_base import CancelRequest, StrategyBase


@dataclass
class TakerBollingerStrategy(StrategyBase):
    window: int = 20
    std_mult: float = 2.0
    order_qty: float = 1.0
    cooldown: float = 0.0
    max_position: float = 1.0
    _counter: int = 0
    _position: float = 0.0
    _mid_history: deque[float] = field(default_factory=deque)
    _order_sides: dict[str, str] = field(default_factory=dict)
    _last_entry_ts: float | None = None

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _append_mid(self, mid: float) -> None:
        self._mid_history.append(mid)
        while len(self._mid_history) > self.window:
            self._mid_history.popleft()

    def _bands(self) -> tuple[float, float, float] | None:
        if self.window <= 1 or len(self._mid_history) < self.window:
            return None
        values = list(self._mid_history)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = sqrt(variance)
        return mean, mean + self.std_mult * std, mean - self.std_mult * std

    def on_snapshot(
        self,
        ts: float,
        book: OrderBookL2,
        active_orders: Iterable[Order],
    ) -> list[PlaceOrderRequest | CancelRequest]:
        _ = active_orders
        mid = book.mid()
        if mid is None:
            return []
        self._append_mid(mid)
        bands = self._bands()
        if bands is None:
            return []

        mean, upper, lower = bands
        entry_size = min(self.order_qty, self.max_position)
        target_position = self._position

        if mid > upper:
            target_position = entry_size
        elif mid < lower:
            target_position = -entry_size
        elif self._position > 0 and mid <= mean:
            target_position = -entry_size
        elif self._position < 0 and mid >= mean:
            target_position = entry_size

        target_position = max(-self.max_position, min(self.max_position, target_position))
        delta = target_position - self._position
        if abs(delta) <= 1e-12:
            return []

        is_new_entry = self._position == 0.0 and target_position != 0.0
        if is_new_entry and self._last_entry_ts is not None and ts - self._last_entry_ts < self.cooldown:
            return []

        side = "BUY" if delta > 0 else "SELL"
        qty = abs(delta)
        order_id = self._next_id("tk")
        self._order_sides[order_id] = side
        if is_new_entry:
            self._last_entry_ts = ts
        return [
            PlaceOrderRequest(
                order_id=order_id,
                side=side,
                type="MARKET",
                price=None,
                qty=qty,
            )
        ]

    def _clamp_position(self) -> None:
        self._position = max(-self.max_position, min(self.max_position, self._position))

    def on_fill(self, fill: Fill) -> None:
        side = self._order_sides.get(fill.order_id)
        if side == "BUY":
            self._position += fill.qty
        elif side == "SELL":
            self._position -= fill.qty
        self._clamp_position()

