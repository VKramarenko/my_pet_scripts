from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from sim.exchange.exchange_sim import PlaceOrderRequest
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2
from sim.strategy.strategy_base import CancelRequest, StrategyBase


@dataclass
class ObImbalanceStrategy(StrategyBase):
    """
    Стратегия на дисбалансе стакана (Orderbook Imbalance).

    Считает метрику: imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    для top-N уровней стакана. Значение в (-1, +1):
      +1 = только биды, давление вверх
      -1 = только аски, давление вниз

    Сглаживает сигнал скользящим средним и выставляет LIMIT-ордера:
      smooth_imb > threshold  → BUY LIMIT на best_bid
      smooth_imb < -threshold → SELL LIMIT на best_ask
    Выход из позиции (MARKET-ордером): когда |smooth_imb| < threshold / 2

    Параметры
    ---------
    depth       : количество уровней стакана с каждой стороны
    threshold   : порог для входа (например, 0.3)
    smoothing   : окно скользящего среднего для сигнала (1 = без сглаживания)
    order_qty   : размер одной заявки
    max_position: максимальная абсолютная позиция
    """

    depth: int = 5
    threshold: float = 0.3
    smoothing: int = 3
    order_qty: float = 1.0
    max_position: float = 1.0

    _counter: int = field(default=0, init=False, repr=False)
    _position: float = field(default=0.0, init=False, repr=False)
    _order_sides: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _imb_history: deque[float] = field(default_factory=deque, init=False, repr=False)
    _active_order_id: str | None = field(default=None, init=False, repr=False)

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _compute_imbalance(self, book: OrderBookL2) -> float | None:
        bids = book.top_n("BUY", self.depth)
        asks = book.top_n("SELL", self.depth)
        bid_vol = sum(s for _, s in bids)
        ask_vol = sum(s for _, s in asks)
        total = bid_vol + ask_vol
        if total == 0:
            return None
        return (bid_vol - ask_vol) / total

    def _smooth_imbalance(self, imb: float) -> float:
        self._imb_history.append(imb)
        while len(self._imb_history) > self.smoothing:
            self._imb_history.popleft()
        return sum(self._imb_history) / len(self._imb_history)

    def on_snapshot(
        self,
        ts: float,
        book: OrderBookL2,
        active_orders: Iterable[Order],
    ) -> list[PlaceOrderRequest | CancelRequest]:
        imb = self._compute_imbalance(book)
        if imb is None:
            return []

        smooth_imb = self._smooth_imbalance(imb)
        active = list(active_orders)
        actions: list[PlaceOrderRequest | CancelRequest] = []

        best_bid = book.best_bid()
        best_ask = book.best_ask()
        if best_bid is None or best_ask is None:
            return []

        exit_threshold = self.threshold / 2.0

        # --- Exit: imbalance normalised → close position with MARKET order ---
        if abs(smooth_imb) < exit_threshold and self._position != 0.0:
            # Cancel any pending limit orders
            for o in active:
                actions.append(CancelRequest(o.id))
            close_side = "SELL" if self._position > 0 else "BUY"
            oid = self._next_id("imb-close")
            self._order_sides[oid] = close_side
            actions.append(
                PlaceOrderRequest(
                    order_id=oid,
                    side=close_side,
                    type="MARKET",
                    price=None,
                    qty=abs(self._position),
                )
            )
            return actions

        # --- Cancel stale pending orders that contradict current signal ---
        for o in active:
            signal_side = self._order_sides.get(o.id)
            if signal_side == "BUY" and smooth_imb < 0:
                actions.append(CancelRequest(o.id))
            elif signal_side == "SELL" and smooth_imb > 0:
                actions.append(CancelRequest(o.id))

        # --- Entry: only if no position and no pending order on this side ---
        has_buy = any(o.side == "BUY" for o in active if o.id not in {a.order_id for a in actions if isinstance(a, CancelRequest)})
        has_sell = any(o.side == "SELL" for o in active if o.id not in {a.order_id for a in actions if isinstance(a, CancelRequest)})

        if smooth_imb > self.threshold and self._position < self.max_position and not has_buy:
            oid = self._next_id("imb-buy")
            self._order_sides[oid] = "BUY"
            actions.append(
                PlaceOrderRequest(
                    order_id=oid,
                    side="BUY",
                    type="LIMIT",
                    price=best_bid,
                    qty=self.order_qty,
                )
            )

        elif smooth_imb < -self.threshold and self._position > -self.max_position and not has_sell:
            oid = self._next_id("imb-sell")
            self._order_sides[oid] = "SELL"
            actions.append(
                PlaceOrderRequest(
                    order_id=oid,
                    side="SELL",
                    type="LIMIT",
                    price=best_ask,
                    qty=self.order_qty,
                )
            )

        return actions

    def on_fill(self, fill: Fill) -> None:
        side = self._order_sides.get(fill.order_id)
        if side == "BUY":
            self._position += fill.qty
        elif side == "SELL":
            self._position -= fill.qty
        self._position = max(-self.max_position, min(self.max_position, self._position))
