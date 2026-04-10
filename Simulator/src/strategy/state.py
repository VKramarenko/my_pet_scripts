from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from src.enums import OrderStatus, Side
from src.events import OrderUpdateEvent, OwnTradeEvent
from src.models import Order, Trade
from src.strategy.metrics import StrategyMetrics


@dataclass(slots=True)
class StrategyState:
    """Local strategy-side cache of orders, trades and basic accounting."""

    orders: dict[str, Order] = field(default_factory=dict)
    active_orders: dict[str, Order] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    position: float = 0.0
    cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    avg_entry_price: float | None = None
    last_mid_price: float | None = None
    last_trade_step_index: int | None = None
    metrics: StrategyMetrics = field(default_factory=StrategyMetrics)
    positions: dict[str, float] = field(default_factory=dict)        # position per instrument_id
    last_mid_prices: dict[str, float] = field(default_factory=dict)  # last mid price per instrument_id

    def reset(self) -> None:
        self.orders.clear()
        self.active_orders.clear()
        self.trades.clear()
        self.position = 0.0
        self.cash = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.equity = 0.0
        self.avg_entry_price = None
        self.last_mid_price = None
        self.last_trade_step_index = None
        self.metrics = StrategyMetrics()
        self.positions.clear()
        self.last_mid_prices.clear()

    def apply_order_update(self, event: OrderUpdateEvent) -> None:
        existing = self.orders.get(event.order_id)
        if existing is None:
            order = Order(
                order_id=event.order_id,
                strategy_id=event.strategy_id,
                side=Side.BUY,
                price=0.0000001,
                qty=(event.remaining_qty or 0.0) + event.filled_qty_delta,
                remaining_qty=event.remaining_qty or 0.0,
                order_type=existing.order_type if existing is not None else None,  # type: ignore[attr-defined]
                status=event.new_status,
                created_at=event.timestamp,
                updated_at=event.timestamp,
            )
            # Side and order_type are unknown from OrderUpdateEvent alone at this stage.
            # Keep a synthetic but valid order container for local bookkeeping.
            order.side = Side.BUY
            from src.enums import OrderType

            order.order_type = OrderType.LIMIT
        else:
            order = replace(existing)
            order.status = event.new_status
            order.remaining_qty = event.remaining_qty if event.remaining_qty is not None else order.remaining_qty
            order.mark_updated(event.timestamp)

        self.orders[event.order_id] = order
        if event.new_status in {OrderStatus.NEW, OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}:
            self.active_orders[event.order_id] = order
        else:
            self.active_orders.pop(event.order_id, None)

    def apply_trade(self, event: OwnTradeEvent) -> None:
        from src.accounting import apply_trade_to_strategy_state

        apply_trade_to_strategy_state(self, replace(event.trade))
