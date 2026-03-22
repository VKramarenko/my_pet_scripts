from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from sim.exchange.exchange_sim import PlaceOrderRequest
from sim.exchange.orders import Fill, Order
from sim.market.orderbook_l2 import OrderBookL2
from sim.strategy.strategy_base import CancelRequest, StrategyBase


@dataclass
class EmaCrossoverStrategy(StrategyBase):
    """
    Трендовая стратегия на пересечении двух EMA.

    Когда быстрая EMA пересекает медленную снизу вверх — открываем лонг
    (BUY LIMIT на best_bid). Когда сверху вниз — шорт (SELL LIMIT на best_ask).
    Используем LIMIT-ордера: становимся мейкером и платим меньше комиссии.

    Параметры
    ---------
    fast_window     : период быстрой EMA (e.g. 10)
    slow_window     : период медленной EMA (e.g. 30), должен быть > fast_window
    order_qty       : размер одной заявки
    max_position    : максимальная абсолютная позиция
    limit_offset    : отступ от best bid/ask (положительное значение → глубже в стакан)
    """

    fast_window: int = 10
    slow_window: int = 30
    order_qty: float = 1.0
    max_position: float = 1.0
    limit_offset: float = 0.0

    _counter: int = field(default=0, init=False, repr=False)
    _position: float = field(default=0.0, init=False, repr=False)
    _order_sides: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # EMA state
    _fast_ema: float | None = field(default=None, init=False, repr=False)
    _slow_ema: float | None = field(default=None, init=False, repr=False)
    _prev_fast: float | None = field(default=None, init=False, repr=False)
    _prev_slow: float | None = field(default=None, init=False, repr=False)

    # SMA warm-up accumulators
    _warmup_buf: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be less than slow_window")
        self._fast_alpha = 2.0 / (self.fast_window + 1)
        self._slow_alpha = 2.0 / (self.slow_window + 1)

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _update_ema(self, mid: float) -> None:
        self._prev_fast = self._fast_ema
        self._prev_slow = self._slow_ema

        if len(self._warmup_buf) < self.slow_window:
            self._warmup_buf.append(mid)
            n = len(self._warmup_buf)
            sma = sum(self._warmup_buf) / n
            # Fast EMA initialises after fast_window bars
            if n >= self.fast_window:
                self._fast_ema = sma if self._fast_ema is None else (
                    self._fast_alpha * mid + (1 - self._fast_alpha) * self._fast_ema
                )
            # Slow EMA initialises after slow_window bars (end of warmup)
            if n == self.slow_window:
                self._slow_ema = sma
        else:
            self._fast_ema = self._fast_alpha * mid + (1 - self._fast_alpha) * self._fast_ema  # type: ignore[operator]
            self._slow_ema = self._slow_alpha * mid + (1 - self._slow_alpha) * self._slow_ema  # type: ignore[operator]

    def _ema_ready(self) -> bool:
        return (
            self._fast_ema is not None
            and self._slow_ema is not None
            and self._prev_fast is not None
            and self._prev_slow is not None
        )

    def _crossed_up(self) -> bool:
        return (
            self._prev_fast < self._prev_slow  # type: ignore[operator]
            and self._fast_ema > self._slow_ema  # type: ignore[operator]
        )

    def _crossed_down(self) -> bool:
        return (
            self._prev_fast > self._prev_slow  # type: ignore[operator]
            and self._fast_ema < self._slow_ema  # type: ignore[operator]
        )

    def on_snapshot(
        self,
        ts: float,
        book: OrderBookL2,
        active_orders: Iterable[Order],
    ) -> list[PlaceOrderRequest | CancelRequest]:
        mid = book.mid()
        if mid is None:
            return []

        self._update_ema(mid)
        if not self._ema_ready():
            return []

        active = list(active_orders)
        actions: list[PlaceOrderRequest | CancelRequest] = []

        best_bid = book.best_bid()
        best_ask = book.best_ask()
        if best_bid is None or best_ask is None:
            return []

        # --- Entry signals ---
        if self._crossed_up() and self._position < self.max_position:
            # Cancel any existing SELL orders first
            for o in active:
                if o.side == "SELL":
                    actions.append(CancelRequest(o.id))
            buy_price = best_bid - self.limit_offset
            oid = self._next_id("ema-buy")
            self._order_sides[oid] = "BUY"
            actions.append(
                PlaceOrderRequest(
                    order_id=oid,
                    side="BUY",
                    type="LIMIT",
                    price=buy_price,
                    qty=self.order_qty,
                )
            )
            return actions

        if self._crossed_down() and self._position > -self.max_position:
            # Cancel any existing BUY orders first
            for o in active:
                if o.side == "BUY":
                    actions.append(CancelRequest(o.id))
            sell_price = best_ask + self.limit_offset
            oid = self._next_id("ema-sell")
            self._order_sides[oid] = "SELL"
            actions.append(
                PlaceOrderRequest(
                    order_id=oid,
                    side="SELL",
                    type="LIMIT",
                    price=sell_price,
                    qty=self.order_qty,
                )
            )
            return actions

        # --- Cancel stale orders if signal is gone ---
        if self._fast_ema <= self._slow_ema:
            for o in active:
                if o.side == "BUY":
                    actions.append(CancelRequest(o.id))
        if self._fast_ema >= self._slow_ema:
            for o in active:
                if o.side == "SELL":
                    actions.append(CancelRequest(o.id))

        return actions

    def on_fill(self, fill: Fill) -> None:
        side = self._order_sides.get(fill.order_id)
        if side == "BUY":
            self._position += fill.qty
        elif side == "SELL":
            self._position -= fill.qty
        self._position = max(-self.max_position, min(self.max_position, self._position))
