from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sim.core.events import MarketSnapshot, MarketTrade
from sim.execution.fill_model_base import IFillModel
from sim.exchange.fees import FeeModel
from sim.exchange.orders import Ack, CancelAck, Fill, Order, OrderSide, Reject
from sim.market.orderbook_l2 import OrderBookL2


@dataclass
class PlaceOrderRequest:
    order_id: str
    side: OrderSide
    type: str
    price: float | None
    qty: float
    tif: str = "GTC"


class ExchangeSim:
    def __init__(
        self,
        book: OrderBookL2,
        fill_model: IFillModel,
        fee_model: FeeModel | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.book = book
        self.fill_model = fill_model
        self.fee_model = fee_model or FeeModel()
        self.latency_ms = latency_ms
        self._orders: dict[str, Order] = {}

    def submit_place(self, order_request: PlaceOrderRequest, ts: float) -> list[Ack | Reject]:
        if order_request.qty <= 0:
            return [Reject(order_request.order_id, ts, "qty must be > 0")]
        if order_request.type == "LIMIT" and order_request.price is None:
            return [Reject(order_request.order_id, ts, "limit order requires price")]
        if order_request.price is not None and order_request.price <= 0:
            return [Reject(order_request.order_id, ts, "price must be > 0")]
        if order_request.order_id in self._orders:
            return [Reject(order_request.order_id, ts, "duplicate order id")]
        effective_ts = ts + self.latency_ms / 1000.0
        order = Order(
            id=order_request.order_id,
            side=order_request.side,
            type=order_request.type,
            price=order_request.price,
            qty=order_request.qty,
            remaining=order_request.qty,
            tif=order_request.tif,
            created_ts=ts,
            status="ACKED",
            effective_ts=effective_ts,
        )
        self._orders[order.id] = order
        return [Ack(order.id, ts)]

    def submit_cancel(self, order_id: str, ts: float) -> list[CancelAck | Reject]:
        order = self._orders.get(order_id)
        if order is None or order.status in {"FILLED", "CANCELED", "REJECTED"}:
            return [Reject(order_id, ts, "unknown order id")]
        order.status = "CANCELED"
        return [CancelAck(order_id, ts)]

    def active_orders(self, ts: float | None = None) -> list[Order]:
        now = float("inf") if ts is None else ts
        return [
            order
            for order in self._orders.values()
            if order.status in {"ACKED", "PARTIAL"} and order.remaining > 0 and order.effective_ts <= now
        ]

    def _apply_fill(self, fill: Fill) -> Fill:
        order = self._orders[fill.order_id]
        qty = min(fill.qty, order.remaining)
        fee = self.fee_model.compute(fill.price, qty, fill.liquidity)
        order.remaining -= qty
        if order.remaining <= 0:
            order.remaining = 0.0
            order.status = "FILLED"
        else:
            order.status = "PARTIAL"
        return Fill(fill.order_id, fill.ts, fill.price, qty, fee, fill.liquidity)

    def on_market_event(self, event: MarketSnapshot | MarketTrade) -> list[Fill]:
        active = self.active_orders(event.ts)
        if isinstance(event, MarketSnapshot):
            raw_fills = self.fill_model.on_snapshot(self.book, event, active, event.ts)
        else:
            raw_fills = self.fill_model.on_trade(self.book, event, active, event.ts)
        return [self._apply_fill(fill) for fill in raw_fills]

    def order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def all_orders(self) -> Iterable[Order]:
        return self._orders.values()

