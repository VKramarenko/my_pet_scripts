from __future__ import annotations

from dataclasses import replace

from src.enums import LiquidityRole, OrderType, Side
from src.models import Level, Order, Snapshot, Trade


def _iter_matching_levels(order: Order, snapshot: Snapshot) -> list[Level]:
    if order.side == Side.BUY:
        return [level for level in snapshot.asks if level.price <= order.price]
    return [level for level in snapshot.bids if level.price >= order.price]


def sort_active_orders(orders: list[Order]) -> list[Order]:
    return sorted(
        orders,
        key=lambda order: (order.created_at, order.order_id),
    )


def can_fully_fill(order: Order, snapshot: Snapshot) -> bool:
    total_qty = sum(level.qty for level in _iter_matching_levels(order, snapshot))
    return total_qty >= order.remaining_qty


def _build_trades(order: Order, levels: list[Level], trade_id_prefix: str = "PREVIEW") -> tuple[list[Trade], float]:
    remaining_qty = order.remaining_qty
    trades: list[Trade] = []

    for index, level in enumerate(levels, start=1):
        if remaining_qty <= 0:
            break
        if level.qty <= 0:
            continue

        fill_qty = min(remaining_qty, level.qty)
        trades.append(
            Trade(
                trade_id=f"{trade_id_prefix}-{index:06d}",
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                timestamp=order.updated_at or order.created_at,
                side=order.side,
                price=level.price,
                qty=fill_qty,
                liquidity_role=LiquidityRole.TAKER,
            )
        )
        remaining_qty -= fill_qty

    return trades, remaining_qty


def execute_fok_against_snapshot(order: Order, snapshot: Snapshot) -> tuple[list[Trade], float]:
    if order.order_type != OrderType.FOK:
        raise ValueError("order must be FOK")
    if not can_fully_fill(order, snapshot):
        return [], order.remaining_qty
    return _build_trades(order, _iter_matching_levels(order, snapshot))


def execute_limit_against_snapshot(order: Order, snapshot: Snapshot) -> tuple[list[Trade], float]:
    if order.order_type != OrderType.LIMIT:
        raise ValueError("order must be LIMIT")
    return _build_trades(order, _iter_matching_levels(order, snapshot))


def try_fill_resting_order(order: Order, snapshot: Snapshot) -> tuple[list[Trade], float]:
    resting_order = replace(order, updated_at=snapshot.timestamp)
    return execute_limit_against_snapshot(resting_order, snapshot)
