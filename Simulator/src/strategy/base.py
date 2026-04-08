from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime

from src.actions import BaseAction
from src.events import OrderUpdateEvent, OwnTradeEvent
from src.models import Order
from src.strategy.context import StrategyContext
from src.strategy.state import StrategyState


class BaseStrategy(ABC):
    """Base class for concrete strategies with default local bookkeeping."""

    def __init__(self, strategy_id: str, config: dict | None = None) -> None:
        self.strategy_id = strategy_id
        self.config = config or {}
        self.state = StrategyState()

    def on_simulation_start(self) -> None:
        self.reset_state()

    def on_order_update(self, event: OrderUpdateEvent) -> None:
        self.state.apply_order_update(event)

    def on_trade(self, event: OwnTradeEvent) -> None:
        self.state.apply_trade(event)

    @abstractmethod
    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        raise NotImplementedError

    def on_simulation_end(self) -> None:
        return None

    def get_order(self, order_id: str) -> Order | None:
        order = self.state.orders.get(order_id)
        return replace(order) if order is not None else None

    def has_active_orders(self) -> bool:
        return bool(self.state.active_orders)

    def get_active_orders(self) -> dict[str, Order]:
        return {order_id: replace(order) for order_id, order in self.state.active_orders.items()}

    def record_equity_point(self, timestamp: datetime, equity: float | None) -> None:
        self.state.metrics.record_equity_point(timestamp, equity)

    def reset_state(self) -> None:
        self.state.reset()

