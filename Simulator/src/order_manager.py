from __future__ import annotations

from src.enums import OrderStatus
from src.models import Order


def resolve_order_status_after_fill(order: Order, new_remaining_qty: float) -> OrderStatus:
    if new_remaining_qty <= 0:
        return OrderStatus.FILLED
    if new_remaining_qty < order.qty:
        return OrderStatus.PARTIALLY_FILLED
    return OrderStatus.ACTIVE
