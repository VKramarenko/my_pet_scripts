from __future__ import annotations

from datetime import datetime

from src.enums import OrderStatus, OrderType, Side
from src.models import Order


def make_limit_order(
    timestamp: datetime,
    *,
    order_id: str = "o-1",
    strategy_id: str = "s-1",
    side: Side = Side.BUY,
    price: float = 100.0,
    qty: float = 1.0,
    remaining_qty: float | None = None,
    status: OrderStatus = OrderStatus.NEW,
) -> Order:
    return Order(
        order_id=order_id,
        strategy_id=strategy_id,
        side=side,
        price=price,
        qty=qty,
        remaining_qty=qty if remaining_qty is None else remaining_qty,
        order_type=OrderType.LIMIT,
        status=status,
        created_at=timestamp,
        updated_at=timestamp,
    )


def make_fok_order(
    timestamp: datetime,
    *,
    order_id: str = "o-1",
    strategy_id: str = "s-1",
    side: Side = Side.BUY,
    price: float = 100.0,
    qty: float = 1.0,
) -> Order:
    return Order(
        order_id=order_id,
        strategy_id=strategy_id,
        side=side,
        price=price,
        qty=qty,
        remaining_qty=qty,
        order_type=OrderType.FOK,
        status=OrderStatus.NEW,
        created_at=timestamp,
        updated_at=timestamp,
    )


def make_active_order(
    timestamp: datetime,
    *,
    order_id: str = "o-1",
    strategy_id: str = "s-1",
    side: Side = Side.BUY,
    price: float = 100.0,
    qty: float = 1.0,
    remaining_qty: float | None = None,
) -> Order:
    return make_limit_order(
        timestamp,
        order_id=order_id,
        strategy_id=strategy_id,
        side=side,
        price=price,
        qty=qty,
        remaining_qty=qty if remaining_qty is None else remaining_qty,
        status=OrderStatus.ACTIVE,
    )

