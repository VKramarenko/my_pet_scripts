from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from datetime import datetime
from typing import Any, Iterable

from src.actions import BaseAction, CancelOrderAction, ModifyOrderAction, PlaceOrderAction
from src.accounting import mark_strategy_to_market
from src.commission_models import CommissionModel, NoCommission
from src.config import SimulationConfig
from src.enums import OrderStatus, OrderType
from src.events import BaseEvent, MarketSnapshotEvent, OrderUpdateEvent, OwnTradeEvent
from src.matching import execute_fok_against_snapshot, execute_limit_against_snapshot, sort_active_orders, try_fill_resting_order
from src.models import Order, Snapshot, Trade
from src.order_manager import resolve_order_status_after_fill
from src.risk_limits import StrategyLimits, check_action_against_limits
from src.slippage_models import NoSlippage, SlippageModel
from src.state import SimulationState
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext


@dataclass(slots=True)
class SimulationEngine:
    """Core event-driven simulation engine for snapshot-based execution."""

    config: SimulationConfig = field(default_factory=SimulationConfig)
    state: SimulationState = field(default_factory=SimulationState)
    strategy: BaseStrategy | None = None
    commission_model: CommissionModel = field(default_factory=NoCommission)
    slippage_model: SlippageModel = field(default_factory=NoSlippage)
    strategy_limits: StrategyLimits = field(default_factory=StrategyLimits)
    _strategy_started: bool = False
    _step_index: int = 0

    def process_market_event(self, event: BaseEvent) -> None:
        if isinstance(event, MarketSnapshotEvent):
            self.process_snapshot(event.snapshot)
            return
        self.state.append_event(event)

    def process_snapshot(
        self,
        snapshot: Snapshot,
        strategy_actions: list[BaseAction] | None = None,
    ) -> list[BaseEvent]:
        if self.strategy is not None and not self._strategy_started:
            self.strategy.on_simulation_start()
            self._strategy_started = True

        self._step_index += 1
        self.state.current_snapshot = snapshot
        self.state.current_time = snapshot.timestamp
        market_event = MarketSnapshotEvent(timestamp=snapshot.timestamp, snapshot=snapshot)
        self.state.append_event(market_event)
        step_events: list[BaseEvent] = [market_event]

        resting_events = self._process_resting_orders(snapshot)
        step_events.extend(resting_events)

        self._deliver_events_to_strategy(resting_events)

        actions: list[BaseAction]
        if self.strategy is not None:
            context = self._build_strategy_context(snapshot, resting_events)
            actions = self.strategy.on_snapshot(context)
        else:
            context = self._build_strategy_context(snapshot=snapshot, step_events=resting_events)
            _ = context
            actions = strategy_actions or []

        action_events = self._process_strategy_actions(actions, snapshot)
        step_events.extend(action_events)
        self._deliver_events_to_strategy(action_events)
        self._update_accounting(snapshot)
        return step_events

    def _process_resting_orders(self, snapshot: Snapshot) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        active_orders = sort_active_orders(list(self.state.active_orders.values()))
        for order in active_orders:
            trades, new_remaining_qty = try_fill_resting_order(order, snapshot)
            if not trades:
                continue
            events.extend(self._apply_trades(order, trades, new_remaining_qty, snapshot.timestamp))
        return events

    def _build_strategy_context(
        self,
        snapshot: Snapshot | None = None,
        step_events: list[BaseEvent] | None = None,
    ) -> StrategyContext | dict[str, Any]:
        snapshot = snapshot or self.state.current_snapshot
        step_events = step_events or []
        if self.strategy is None:
            return {
                "current_snapshot": self.state.current_snapshot,
                "active_orders": dict(self.state.active_orders),
            }

        if snapshot is None:
            raise ValueError("snapshot is required to build strategy context")

        order_updates = [
            event
            for event in step_events
            if isinstance(event, OrderUpdateEvent) and event.strategy_id == self.strategy.strategy_id
        ]
        new_trades = [
            replace(event.trade)
            for event in step_events
            if isinstance(event, OwnTradeEvent) and event.trade.strategy_id == self.strategy.strategy_id
        ]
        active_orders = {
            order_id: replace(order)
            for order_id, order in self.state.active_orders.items()
            if order.strategy_id == self.strategy.strategy_id
        }
        return StrategyContext(
            timestamp=snapshot.timestamp,
            snapshot=snapshot,
            active_orders=active_orders,
            new_trades=new_trades,
            order_updates=order_updates,
            position=self.strategy.state.position,
            cash=self.strategy.state.cash,
            realized_pnl=self.strategy.state.realized_pnl,
            unrealized_pnl=self.strategy.state.unrealized_pnl,
            equity=self.strategy.state.equity,
            metrics=self.strategy.state.metrics,
        )

    def _process_strategy_actions(self, actions: list[BaseAction], snapshot: Snapshot) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        for action in actions:
            if isinstance(action, PlaceOrderAction):
                events.extend(self._place_order_from_action(action, snapshot))
            elif isinstance(action, CancelOrderAction):
                events.extend(self._cancel_order(action.order_id, snapshot.timestamp))
            elif isinstance(action, ModifyOrderAction):
                events.extend(self._modify_order(action, snapshot))
        return events

    def _deliver_events_to_strategy(self, events: list[BaseEvent]) -> None:
        if self.strategy is None:
            return

        order_updates = [
            event
            for event in events
            if isinstance(event, OrderUpdateEvent) and event.strategy_id == self.strategy.strategy_id
        ]
        own_trades = [
            event
            for event in events
            if isinstance(event, OwnTradeEvent) and event.trade.strategy_id == self.strategy.strategy_id
        ]

        for event in order_updates:
            self.strategy.on_order_update(event)
        for event in own_trades:
            self.strategy.on_trade(event)

    def run(self, snapshots: Iterable[Snapshot]) -> SimulationState:
        if self.strategy is not None and not self._strategy_started:
            self.strategy.on_simulation_start()
            self._strategy_started = True

        for snapshot in snapshots:
            self.process_snapshot(snapshot)

        if self.strategy is not None and self._strategy_started:
            self.strategy.on_simulation_end()
        return self.state

    def _place_order_from_action(self, action: PlaceOrderAction, snapshot: Snapshot) -> list[BaseEvent]:
        if self.strategy is not None and action.strategy_id == self.strategy.strategy_id:
            self.strategy.state.metrics.submitted_orders += 1
            limit_result = check_action_against_limits(
                action,
                strategy_state=self.strategy.state,
                active_orders={
                    order_id: order
                    for order_id, order in self.state.active_orders.items()
                    if order.strategy_id == action.strategy_id
                },
                limits=self.strategy_limits,
                step_index=self._step_index,
            )
            if not limit_result.allowed:
                rejected_order = Order(
                    order_id=self.state.next_order_id(),
                    strategy_id=action.strategy_id,
                    side=action.side,
                    price=action.price,
                    qty=action.qty,
                    remaining_qty=action.qty,
                    order_type=action.order_type,
                    status=OrderStatus.REJECTED,
                    created_at=snapshot.timestamp,
                    updated_at=snapshot.timestamp,
                )
                self.state.complete_order(rejected_order)
                rejected_event = self._order_update_event(
                    rejected_order,
                    old_status=OrderStatus.NEW,
                    new_status=OrderStatus.REJECTED,
                    filled_qty_delta=0.0,
                    timestamp=snapshot.timestamp,
                    reason=limit_result.reason,
                )
                self.state.append_event(rejected_event)
                return [rejected_event]

        order = Order(
            order_id=self.state.next_order_id(),
            strategy_id=action.strategy_id,
            side=action.side,
            price=action.price,
            qty=action.qty,
            remaining_qty=action.qty,
            order_type=action.order_type,
            status=OrderStatus.NEW,
            created_at=snapshot.timestamp,
            updated_at=snapshot.timestamp,
        )

        if order.order_type == OrderType.FOK:
            trades, new_remaining_qty = execute_fok_against_snapshot(order, snapshot)
            if not trades:
                order.status = OrderStatus.CANCELED
                order.remaining_qty = order.qty
                self.state.complete_order(order)
                canceled_event = self._order_update_event(
                    order,
                    old_status=OrderStatus.NEW,
                    new_status=OrderStatus.CANCELED,
                    filled_qty_delta=0.0,
                    timestamp=snapshot.timestamp,
                )
                self.state.append_event(canceled_event)
                return [canceled_event]
            return self._apply_trades(order, trades, new_remaining_qty, snapshot.timestamp, old_status=OrderStatus.NEW)

        trades, new_remaining_qty = execute_limit_against_snapshot(order, snapshot)
        return self._apply_trades(order, trades, new_remaining_qty, snapshot.timestamp, old_status=OrderStatus.NEW)

    def _cancel_order(self, order_id: str, timestamp: datetime) -> list[BaseEvent]:
        order = self.state.active_orders.get(order_id)
        if order is None:
            return []

        old_status = order.status
        order.status = OrderStatus.CANCELED
        order.mark_updated(timestamp)
        self.state.complete_order(order)
        event = self._order_update_event(
            order,
            old_status=old_status,
            new_status=OrderStatus.CANCELED,
            filled_qty_delta=0.0,
            timestamp=timestamp,
        )
        self.state.append_event(event)
        return [event]

    def _modify_order(self, action: ModifyOrderAction, snapshot: Snapshot) -> list[BaseEvent]:
        existing_order = self.state.active_orders.get(action.order_id)
        if existing_order is None:
            return []

        events = self._cancel_order(existing_order.order_id, snapshot.timestamp)
        replacement_action = PlaceOrderAction(
            strategy_id=existing_order.strategy_id,
            side=existing_order.side,
            price=action.new_price if action.new_price is not None else existing_order.price,
            qty=action.new_qty if action.new_qty is not None else existing_order.remaining_qty,
            order_type=existing_order.order_type,
            client_order_id=None,
        )
        events.extend(self._place_order_from_action(replacement_action, snapshot))
        return events

    def _apply_trades(
        self,
        order: Order,
        trades: list[Trade],
        new_remaining_qty: float,
        timestamp: datetime,
        *,
        old_status: OrderStatus | None = None,
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        starting_status = old_status if old_status is not None else order.status

        for trade in trades:
            exec_price = self.slippage_model.apply(trade.side, trade.price, trade.qty)
            persisted_trade = Trade(
                trade_id=self.state.next_trade_id(),
                order_id=order.order_id,
                strategy_id=trade.strategy_id,
                timestamp=timestamp,
                side=trade.side,
                price=exec_price,
                qty=trade.qty,
                liquidity_role=trade.liquidity_role,
                raw_price=trade.price,
                commission=0.0,
                notional=exec_price * trade.qty,
            )
            persisted_trade.commission = self.commission_model.compute(persisted_trade)
            self.state.append_trade(persisted_trade)
            event = OwnTradeEvent(timestamp=timestamp, trade=persisted_trade)
            self.state.append_event(event)
            events.append(event)

        order.remaining_qty = new_remaining_qty
        order.mark_updated(timestamp)
        new_status = resolve_order_status_after_fill(order, new_remaining_qty)
        order.status = new_status

        if new_status in {OrderStatus.ACTIVE, OrderStatus.PARTIALLY_FILLED}:
            self.state.register_order(order)
        else:
            self.state.complete_order(order)

        filled_qty_delta = sum(trade.qty for trade in trades)
        status_changed = starting_status != new_status
        if status_changed or filled_qty_delta > 0:
            update_event = self._order_update_event(
                order,
                old_status=starting_status,
                new_status=new_status,
                filled_qty_delta=filled_qty_delta,
                timestamp=timestamp,
            )
            self.state.append_event(update_event)
            events.append(update_event)
            if new_status == OrderStatus.FILLED and self.strategy is not None and order.strategy_id == self.strategy.strategy_id:
                self.strategy.state.metrics.filled_orders += 1
            if new_status == OrderStatus.CANCELED and self.strategy is not None and order.strategy_id == self.strategy.strategy_id:
                self.strategy.state.metrics.canceled_orders += 1
        return events

    def _order_update_event(
        self,
        order: Order,
        *,
        old_status: OrderStatus | None,
        new_status: OrderStatus,
        filled_qty_delta: float,
        timestamp: datetime,
        reason: str | None = None,
    ) -> OrderUpdateEvent:
        return OrderUpdateEvent(
            timestamp=timestamp,
            order_id=order.order_id,
            strategy_id=order.strategy_id,
            old_status=old_status,
            new_status=new_status,
            filled_qty_delta=filled_qty_delta,
            remaining_qty=order.remaining_qty,
            reason=reason,
        )

    def _update_accounting(self, snapshot: Snapshot) -> None:
        if self.strategy is None:
            return
        mark_strategy_to_market(
            self.strategy.state,
            snapshot,
            missing_policy=self.config.mark_to_mid_missing_policy,
        )
