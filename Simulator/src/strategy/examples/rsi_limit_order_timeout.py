from __future__ import annotations

from datetime import timedelta

from src.actions import BaseAction, CancelOrderAction, PlaceOrderAction
from src.enums import OrderType, Side
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.examples.indicators import compute_rsi


class RSILimitOrderTimeoutStrategy(BaseStrategy):
    """Places passive RSI-driven limit orders and cancels them after a TTL."""

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
        order_ttl_seconds = float(self.config.get("order_ttl_seconds", 5.0))

        rsi = compute_rsi(self._mid_prices, period)
        self.last_rsi = rsi

        cancel_actions = self._build_cancel_actions(context, order_ttl_seconds)
        if cancel_actions:
            return cancel_actions

        if rsi is None or context.active_orders:
            return []

        best_ask = context.snapshot.best_ask()
        best_bid = context.snapshot.best_bid()
        if context.position <= 0 and rsi <= oversold and best_bid is not None:
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.BUY,
                    price=best_bid.price,
                    qty=order_qty,
                    order_type=OrderType.LIMIT,
                )
            ]

        if context.position > 0 and rsi >= overbought and best_ask is not None:
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.SELL,
                    price=best_ask.price,
                    qty=min(order_qty, context.position),
                    order_type=OrderType.LIMIT,
                )
            ]

        return []

    def _build_cancel_actions(self, context: StrategyContext, order_ttl_seconds: float) -> list[BaseAction]:
        ttl = timedelta(seconds=order_ttl_seconds)
        stale_order_ids = [
            order.order_id
            for order in context.active_orders.values()
            if context.timestamp - order.created_at >= ttl
        ]
        return [
            CancelOrderAction(strategy_id=self.strategy_id, order_id=order_id)
            for order_id in stale_order_ids
        ]
