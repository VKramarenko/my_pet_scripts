from __future__ import annotations

from collections import deque

from src.actions import BaseAction, PlaceOrderAction
from src.enums import OrderType, Side
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext


def compute_rsi(prices: list[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(prices) < period + 1:
        return None

    window = prices[-(period + 1) :]
    gains = 0.0
    losses = 0.0
    for previous, current in zip(window, window[1:]):
        change = current - previous
        if change > 0:
            gains += change
        elif change < 0:
            losses += abs(change)

    average_gain = gains / period
    average_loss = losses / period
    if average_loss == 0:
        return 100.0
    rs = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + rs))


class RSIMeanReversionStrategy(BaseStrategy):
    """Buys on oversold RSI and exits on overbought RSI."""

    def __init__(self, strategy_id: str, config: dict | None = None) -> None:
        super().__init__(strategy_id, config)
        self._mid_prices: list[float] = []
        self.last_rsi: float | None = None

    def on_simulation_start(self) -> None:
        super().on_simulation_start()
        self._mid_prices = []
        self.last_rsi = None

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        mid_price = context.snapshot.mid_price()
        if mid_price is None:
            return []

        self._mid_prices.append(mid_price)
        period = int(self.config.get("rsi_period", 14))
        oversold = float(self.config.get("oversold", 30.0))
        overbought = float(self.config.get("overbought", 70.0))
        order_qty = float(self.config.get("qty", 1.0))
        order_type_name = str(self.config.get("order_type", "LIMIT")).upper()
        order_type = OrderType[order_type_name]

        rsi = compute_rsi(self._mid_prices, period)
        self.last_rsi = rsi
        if rsi is None or self.has_active_orders():
            return []

        best_ask = context.snapshot.best_ask()
        best_bid = context.snapshot.best_bid()
        if context.position <= 0 and rsi <= oversold and best_ask is not None:
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.BUY,
                    price=best_ask.price,
                    qty=order_qty,
                    order_type=order_type,
                )
            ]

        if context.position > 0 and rsi >= overbought and best_bid is not None:
            sell_qty = min(order_qty, context.position)
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.SELL,
                    price=best_bid.price,
                    qty=sell_qty,
                    order_type=order_type,
                )
            ]
        return []


class MovingAverageCrossStrategy(BaseStrategy):
    """Buys on bullish MA crossover and sells on bearish crossover."""

    def __init__(self, strategy_id: str, config: dict | None = None) -> None:
        super().__init__(strategy_id, config)
        self._mid_prices: deque[float] = deque()
        self.last_signal: str | None = None

    def on_simulation_start(self) -> None:
        super().on_simulation_start()
        self._mid_prices.clear()
        self.last_signal = None

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        mid_price = context.snapshot.mid_price()
        if mid_price is None:
            return []

        short_window = int(self.config.get("short_window", 3))
        long_window = int(self.config.get("long_window", 5))
        qty = float(self.config.get("qty", 1.0))
        if short_window <= 0 or long_window <= 0 or short_window >= long_window:
            raise ValueError("Require 0 < short_window < long_window")

        self._mid_prices.append(mid_price)
        while len(self._mid_prices) > long_window:
            self._mid_prices.popleft()

        if len(self._mid_prices) < long_window or self.has_active_orders():
            return []

        prices = list(self._mid_prices)
        short_ma = sum(prices[-short_window:]) / short_window
        long_ma = sum(prices[-long_window:]) / long_window
        best_ask = context.snapshot.best_ask()
        best_bid = context.snapshot.best_bid()

        if short_ma > long_ma and context.position <= 0 and self.last_signal != "buy" and best_ask is not None:
            self.last_signal = "buy"
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.BUY,
                    price=best_ask.price,
                    qty=qty,
                    order_type=OrderType.LIMIT,
                )
            ]

        if short_ma < long_ma and context.position > 0 and self.last_signal != "sell" and best_bid is not None:
            self.last_signal = "sell"
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.SELL,
                    price=best_bid.price,
                    qty=min(qty, context.position),
                    order_type=OrderType.LIMIT,
                )
            ]

        return []
