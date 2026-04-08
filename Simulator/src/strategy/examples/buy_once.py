from __future__ import annotations

from src.actions import BaseAction, PlaceOrderAction
from src.enums import OrderType, Side
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext


class PassiveBuyOnceStrategy(BaseStrategy):
    """Places one passive buy order once, then waits for fills/events."""

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        target_price = float(self.config.get("price", 0.0))
        target_qty = float(self.config.get("qty", 0.0))
        if self.has_active_orders() or self.state.position != 0:
            return []
        if self.state.orders:
            return []
        return [
            PlaceOrderAction(
                strategy_id=self.strategy_id,
                side=Side.BUY,
                price=target_price,
                qty=target_qty,
                order_type=OrderType.LIMIT,
            )
        ]
