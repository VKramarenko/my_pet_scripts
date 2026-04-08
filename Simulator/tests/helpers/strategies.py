from __future__ import annotations

from collections import deque

from src.actions import BaseAction, CancelOrderAction, ModifyOrderAction, PlaceOrderAction
from src.enums import OrderType, Side
from src.events import OrderUpdateEvent, OwnTradeEvent
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy


class NoOpStrategy(BaseStrategy):
    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        return []


class RecordingStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, config: dict | None = None) -> None:
        super().__init__(strategy_id, config)
        self.started = 0
        self.ended = 0
        self.contexts: list[StrategyContext] = []
        self.order_updates: list[OrderUpdateEvent] = []
        self.trades: list[OwnTradeEvent] = []

    def on_simulation_start(self) -> None:
        super().on_simulation_start()
        self.started += 1

    def on_order_update(self, event: OrderUpdateEvent) -> None:
        super().on_order_update(event)
        self.order_updates.append(event)

    def on_trade(self, event: OwnTradeEvent) -> None:
        super().on_trade(event)
        self.trades.append(event)

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        self.contexts.append(context)
        return []

    def on_simulation_end(self) -> None:
        self.ended += 1


class ActionReturningStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, actions_by_step: list[list[BaseAction]], config: dict | None = None) -> None:
        super().__init__(strategy_id, config)
        self._actions_by_step = deque(actions_by_step)
        self.contexts: list[StrategyContext] = []

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        self.contexts.append(context)
        return self._actions_by_step.popleft() if self._actions_by_step else []


class AlwaysPlaceBuyLimitStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, price: float = 100.0, qty: float = 1.0) -> None:
        super().__init__(strategy_id)
        self.price = price
        self.qty = qty

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        if self.has_active_orders():
            return []
        return [
            PlaceOrderAction(
                strategy_id=self.strategy_id,
                side=Side.BUY,
                price=self.price,
                qty=self.qty,
                order_type=OrderType.LIMIT,
            )
        ]


class CancelStaleOrderStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, stale_after_steps: int = 2, price: float = 100.0, qty: float = 1.0) -> None:
        super().__init__(strategy_id)
        self.stale_after_steps = stale_after_steps
        self.price = price
        self.qty = qty
        self.step_index = 0

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        self.step_index += 1
        if not self.has_active_orders() and not self.state.orders:
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.BUY,
                    price=self.price,
                    qty=self.qty,
                    order_type=OrderType.LIMIT,
                )
            ]
        if self.has_active_orders() and self.step_index > self.stale_after_steps:
            order_id = next(iter(self.state.active_orders))
            return [CancelOrderAction(strategy_id=self.strategy_id, order_id=order_id)]
        return []


class RepriceStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, new_price: float) -> None:
        super().__init__(strategy_id)
        self.new_price = new_price

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        if self.has_active_orders():
            order_id = next(iter(self.state.active_orders))
            return [ModifyOrderAction(strategy_id=self.strategy_id, order_id=order_id, new_price=self.new_price)]
        return []


class ThresholdStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, threshold: float, qty: float = 1.0) -> None:
        super().__init__(strategy_id)
        self.threshold = threshold
        self.qty = qty

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        best_ask = context.snapshot.best_ask()
        if best_ask is None or self.has_active_orders():
            return []
        if best_ask.price <= self.threshold:
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.BUY,
                    price=best_ask.price,
                    qty=self.qty,
                    order_type=OrderType.LIMIT,
                )
            ]
        return []


__all__ = [
    "ActionReturningStrategy",
    "AlwaysPlaceBuyLimitStrategy",
    "CancelStaleOrderStrategy",
    "NoOpStrategy",
    "PassiveBuyOnceStrategy",
    "RecordingStrategy",
    "RepriceStrategy",
    "ThresholdStrategy",
]

