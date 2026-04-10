from __future__ import annotations

from src.actions import BaseAction, PlaceOrderAction
from src.enums import OrderType, Side
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.examples.indicators import compute_rsi


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
