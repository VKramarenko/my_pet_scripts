from __future__ import annotations

from datetime import timedelta

from src.actions import BaseAction, CancelOrderAction, PlaceOrderAction
from src.enums import OrderType, Side
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.examples.indicators import compute_rsi


class RSILimitOrderTemplateStrategy(BaseStrategy):
    """Educational RSI strategy showing place/wait/cancel order management."""

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
        rsi = self._compute_signal()
        self.last_rsi = rsi

        # Step 1. Manage already resting orders before considering new entries.
        cancel_actions = self._cancel_stale_orders(context)
        if cancel_actions:
            return cancel_actions

        # Step 2. Do not stack a new limit order while another one is still resting.
        if rsi is None or context.active_orders:
            return []

        # Step 3. Convert the RSI signal into a passive limit order near the top of book.
        signal = self._entry_signal(context, rsi)
        if signal is None:
            return []
        side, price, qty = signal
        return [
            PlaceOrderAction(
                strategy_id=self.strategy_id,
                side=side,
                price=price,
                qty=qty,
                order_type=OrderType.LIMIT,
            )
        ]

    def _compute_signal(self) -> float | None:
        period = int(self.config.get("rsi_period", 14))
        return compute_rsi(self._mid_prices, period)

    def _cancel_stale_orders(self, context: StrategyContext) -> list[BaseAction]:
        ttl = timedelta(seconds=float(self.config.get("order_ttl_seconds", 5.0)))
        stale_order_ids = []
        for order in context.active_orders.values():
            age = context.timestamp - order.created_at
            if age >= ttl:
                stale_order_ids.append(order.order_id)
        return [
            CancelOrderAction(strategy_id=self.strategy_id, order_id=order_id)
            for order_id in stale_order_ids
        ]

    def _entry_signal(self, context: StrategyContext, rsi: float) -> tuple[Side, float, float] | None:
        oversold = float(self.config.get("oversold", 30.0))
        overbought = float(self.config.get("overbought", 70.0))
        order_qty = float(self.config.get("qty", 1.0))

        best_bid = context.snapshot.best_bid()
        best_ask = context.snapshot.best_ask()

        if context.position <= 0 and rsi <= oversold and best_bid is not None:
            return Side.BUY, best_bid.price, order_qty

        if context.position > 0 and rsi >= overbought and best_ask is not None:
            return Side.SELL, best_ask.price, min(order_qty, context.position)

        return None
