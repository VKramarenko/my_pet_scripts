from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sim.core.events import MarketSnapshot, MarketTrade
from sim.execution.config import ExecutionConfig
from sim.execution.fill_model_base import IFillModel
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2


@dataclass
class OrderExecutionState:
    queue_ahead: float
    last_seen_level_size: float
    price_level: float
    side: str


class QueueFillModel(IFillModel):
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()
        self._state: dict[str, OrderExecutionState] = {}

    def _ensure_state(self, book: OrderBookL2, order: Order) -> OrderExecutionState | None:
        if order.type != "LIMIT" or order.price is None:
            return None
        if order.id in self._state:
            return self._state[order.id]
        level_size = book.level(order.side, order.price)
        state = OrderExecutionState(
            queue_ahead=max(level_size, 0.0),
            last_seen_level_size=level_size,
            price_level=order.price,
            side=order.side,
        )
        self._state[order.id] = state
        return state

    def _cleanup(self, active_orders: Iterable[Order]) -> None:
        active_ids = {o.id for o in active_orders}
        for order_id in list(self._state.keys()):
            if order_id not in active_ids:
                self._state.pop(order_id, None)

    def on_snapshot(
        self,
        book: OrderBookL2,
        snapshot: MarketSnapshot,
        active_orders: Iterable[Order],
        ts: float,
    ) -> list[Fill]:
        orders = list(active_orders)
        self._cleanup(orders)
        for order in orders:
            state = self._ensure_state(book, order)
            if state is None:
                continue
            new_size = book.level(order.side, state.price_level)
            delta = new_size - state.last_seen_level_size
            if delta > 0 and self.config.mode == "pessimistic":
                state.queue_ahead += delta
            elif delta < 0:
                state.queue_ahead += delta
            state.queue_ahead = max(state.queue_ahead, 0.0)
            state.last_seen_level_size = new_size
        return []

    def on_trade(
        self,
        book: OrderBookL2,
        trade: MarketTrade,
        active_orders: Iterable[Order],
        ts: float,
    ) -> list[Fill]:
        orders = list(active_orders)
        self._cleanup(orders)
        fills: list[Fill] = []
        for order in orders:
            if order.type != "LIMIT" or order.price is None or order.remaining <= 0:
                continue
            state = self._ensure_state(book, order)
            if state is None:
                continue
            if order.side == "BUY" and trade.side != "seller_initiated":
                continue
            if order.side == "SELL" and trade.side != "buyer_initiated":
                continue
            touches = trade.price == order.price
            if self.config.allow_trade_through:
                if order.side == "BUY":
                    touches = touches or trade.price <= order.price
                else:
                    touches = touches or trade.price >= order.price
            if not touches:
                continue
            state.queue_ahead -= trade.size
            if state.queue_ahead > 0:
                continue
            fillable = min(order.remaining, abs(state.queue_ahead))
            if self.config.max_fill_per_event is not None:
                fillable = min(fillable, self.config.max_fill_per_event)
            if fillable <= 0:
                continue
            state.queue_ahead = max(-(abs(state.queue_ahead) - fillable), 0.0)
            fills.append(Fill(order.id, ts, order.price, fillable, 0.0, "MAKER"))
        return fills

    def debug_state(self, order_id: str) -> OrderExecutionState | None:
        return self._state.get(order_id)

