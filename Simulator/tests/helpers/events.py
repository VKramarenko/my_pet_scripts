from __future__ import annotations

from datetime import datetime

from src.enums import LiquidityRole, OrderStatus, Side
from src.events import MarketSnapshotEvent, OrderUpdateEvent, OwnTradeEvent
from src.models import Snapshot, Trade


def make_order_update_event(
    timestamp: datetime,
    *,
    order_id: str = "o-1",
    strategy_id: str = "s-1",
    old_status: OrderStatus | None = None,
    new_status: OrderStatus = OrderStatus.ACTIVE,
    filled_qty_delta: float = 0.0,
    remaining_qty: float | None = 1.0,
    reason: str | None = None,
) -> OrderUpdateEvent:
    return OrderUpdateEvent(
        timestamp=timestamp,
        order_id=order_id,
        strategy_id=strategy_id,
        old_status=old_status,
        new_status=new_status,
        filled_qty_delta=filled_qty_delta,
        remaining_qty=remaining_qty,
        reason=reason,
    )


def make_trade_event(
    timestamp: datetime,
    *,
    trade_id: str = "t-1",
    order_id: str = "o-1",
    strategy_id: str = "s-1",
    side: Side = Side.BUY,
    price: float = 100.0,
    qty: float = 1.0,
    liquidity_role: LiquidityRole = LiquidityRole.TAKER,
    commission: float = 0.0,
) -> OwnTradeEvent:
    trade = Trade(
        trade_id=trade_id,
        order_id=order_id,
        strategy_id=strategy_id,
        timestamp=timestamp,
        side=side,
        price=price,
        qty=qty,
        liquidity_role=liquidity_role,
        commission=commission,
    )
    return OwnTradeEvent(timestamp=timestamp, trade=trade)


def make_snapshot_event(snapshot: Snapshot) -> MarketSnapshotEvent:
    return MarketSnapshotEvent(timestamp=snapshot.timestamp, snapshot=snapshot)

