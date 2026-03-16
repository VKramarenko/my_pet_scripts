from __future__ import annotations

from typing import Iterable

from sim.core.events import MarketSnapshot, MarketTrade
from sim.execution.config import ExecutionConfig
from sim.execution.fill_model_base import IFillModel
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2


class TouchFillModel(IFillModel):
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def _fill_qty(self, order: Order, available: float | None) -> float:
        if available is None:
            qty = order.remaining
        else:
            qty = min(order.remaining, available)
        if self.config.max_fill_per_event is not None:
            qty = min(qty, self.config.max_fill_per_event)
        return max(qty, 0.0)

    def on_snapshot(
        self,
        book: OrderBookL2,
        snapshot: MarketSnapshot,
        active_orders: Iterable[Order],
        ts: float,
    ) -> list[Fill]:
        fills: list[Fill] = []
        best_bid = book.best_bid()
        best_ask = book.best_ask()
        for order in active_orders:
            if order.remaining <= 0:
                continue
            if order.type == "MARKET":
                if order.side == "BUY" and best_ask is not None:
                    price = best_ask * (1 + self.config.market_slippage_bps / 10_000.0)
                    qty = self._fill_qty(order, book.level("SELL", best_ask))
                    if qty > 0:
                        fills.append(Fill(order.id, ts, price, qty, 0.0, "TAKER"))
                elif order.side == "SELL" and best_bid is not None:
                    price = best_bid * (1 - self.config.market_slippage_bps / 10_000.0)
                    qty = self._fill_qty(order, book.level("BUY", best_bid))
                    if qty > 0:
                        fills.append(Fill(order.id, ts, price, qty, 0.0, "TAKER"))
                continue

            if order.type != "LIMIT" or order.price is None:
                continue

            if self.config.require_trade_for_fill:
                continue

            if order.side == "BUY" and best_ask is not None and best_ask <= order.price:
                price = order.price if self.config.fill_price == "order_price" else best_ask
                qty = self._fill_qty(order, book.level("SELL", best_ask))
                if qty > 0:
                    fills.append(Fill(order.id, ts, price, qty, 0.0, "MAKER"))
            elif order.side == "SELL" and best_bid is not None and best_bid >= order.price:
                price = order.price if self.config.fill_price == "order_price" else best_bid
                qty = self._fill_qty(order, book.level("BUY", best_bid))
                if qty > 0:
                    fills.append(Fill(order.id, ts, price, qty, 0.0, "MAKER"))
        return fills

    def on_trade(
        self,
        book: OrderBookL2,
        trade: MarketTrade,
        active_orders: Iterable[Order],
        ts: float,
    ) -> list[Fill]:
        if not self.config.require_trade_for_fill:
            return []
        fills: list[Fill] = []
        for order in active_orders:
            if order.type != "LIMIT" or order.price is None or order.remaining <= 0:
                continue
            if order.side == "BUY" and trade.price <= order.price:
                price = order.price if self.config.fill_price == "order_price" else trade.price
                qty = self._fill_qty(order, trade.size)
                if qty > 0:
                    fills.append(Fill(order.id, ts, price, qty, 0.0, "MAKER"))
            elif order.side == "SELL" and trade.price >= order.price:
                price = order.price if self.config.fill_price == "order_price" else trade.price
                qty = self._fill_qty(order, trade.size)
                if qty > 0:
                    fills.append(Fill(order.id, ts, price, qty, 0.0, "MAKER"))
        return fills

