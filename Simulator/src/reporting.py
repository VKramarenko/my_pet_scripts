from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.enums import OrderStatus, Side
from src.events import OrderUpdateEvent
from src.models import Order, Trade

if TYPE_CHECKING:
    from src.engine import SimulationEngine


@dataclass(frozen=True, slots=True)
class ExecutionFillRecord:
    trade_id: str
    order_id: str
    strategy_id: str
    timestamp: str
    side: str
    qty: float
    exec_price: float
    raw_price: float | None
    commission: float
    notional: float
    liquidity_role: str


@dataclass(frozen=True, slots=True)
class ExecutionOrderRecord:
    order_id: str
    strategy_id: str
    side: str
    order_type: str
    submitted_at: str
    submitted_price: float
    submitted_qty: float
    final_status: str
    final_updated_at: str
    filled_qty_total: float
    remaining_qty: float
    avg_fill_price: float | None
    first_fill_at: str | None
    last_fill_at: str | None
    trade_count: int
    commission_total: float
    slippage_cost_total: float
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    strategy_id: str | None
    generated_orders: int
    fills_total: int
    filled_orders: int
    canceled_orders: int
    rejected_orders: int
    active_orders: int
    orders: list[ExecutionOrderRecord]
    fills: list[ExecutionFillRecord]


def _iter_strategy_orders(engine: SimulationEngine, strategy_id: str | None) -> list[Order]:
    all_orders = [
        *engine.state.completed_orders.values(),
        *engine.state.active_orders.values(),
    ]
    filtered_orders = [
        order
        for order in all_orders
        if strategy_id is None or order.strategy_id == strategy_id
    ]
    return sorted(filtered_orders, key=lambda order: (order.created_at, order.order_id))


def _iter_strategy_trades(engine: SimulationEngine, strategy_id: str | None) -> list[Trade]:
    filtered_trades = [
        trade
        for trade in engine.state.trades
        if strategy_id is None or trade.strategy_id == strategy_id
    ]
    return sorted(filtered_trades, key=lambda trade: (trade.timestamp, trade.trade_id))


def _collect_rejection_reasons(engine: SimulationEngine) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for event in engine.state.event_log:
        if not isinstance(event, OrderUpdateEvent):
            continue
        if event.reason is None:
            continue
        reasons[event.order_id] = event.reason
    return reasons


def _compute_signed_slippage_cost(trade: Trade) -> float:
    if trade.raw_price is None:
        return 0.0
    if trade.side == Side.BUY:
        return (trade.price - trade.raw_price) * trade.qty
    return (trade.raw_price - trade.price) * trade.qty


def build_execution_report(
    engine: SimulationEngine,
    *,
    strategy_id: str | None = None,
) -> ExecutionReport:
    orders = _iter_strategy_orders(engine, strategy_id)
    trades = _iter_strategy_trades(engine, strategy_id)
    rejection_reasons = _collect_rejection_reasons(engine)

    trades_by_order_id: dict[str, list[Trade]] = {}
    for trade in trades:
        trades_by_order_id.setdefault(trade.order_id, []).append(trade)

    order_records: list[ExecutionOrderRecord] = []
    for order in orders:
        order_trades = trades_by_order_id.get(order.order_id, [])
        filled_qty_total = sum(trade.qty for trade in order_trades)
        total_notional = sum(trade.price * trade.qty for trade in order_trades)
        avg_fill_price = (
            total_notional / filled_qty_total if filled_qty_total > 0 else None
        )
        first_fill_at = order_trades[0].timestamp.isoformat() if order_trades else None
        last_fill_at = order_trades[-1].timestamp.isoformat() if order_trades else None
        commission_total = sum(trade.commission for trade in order_trades)
        slippage_cost_total = sum(_compute_signed_slippage_cost(trade) for trade in order_trades)
        final_updated_at = order.updated_at or order.created_at

        order_records.append(
            ExecutionOrderRecord(
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                side=order.side.value,
                order_type=order.order_type.value,
                submitted_at=order.created_at.isoformat(),
                submitted_price=order.price,
                submitted_qty=order.qty,
                final_status=order.status.value,
                final_updated_at=final_updated_at.isoformat(),
                filled_qty_total=filled_qty_total,
                remaining_qty=order.remaining_qty,
                avg_fill_price=avg_fill_price,
                first_fill_at=first_fill_at,
                last_fill_at=last_fill_at,
                trade_count=len(order_trades),
                commission_total=commission_total,
                slippage_cost_total=slippage_cost_total,
                rejection_reason=rejection_reasons.get(order.order_id),
            )
        )

    fill_records = [
        ExecutionFillRecord(
            trade_id=trade.trade_id,
            order_id=trade.order_id,
            strategy_id=trade.strategy_id,
            timestamp=trade.timestamp.isoformat(),
            side=trade.side.value,
            qty=trade.qty,
            exec_price=trade.price,
            raw_price=trade.raw_price,
            commission=trade.commission,
            notional=trade.notional or (trade.price * trade.qty),
            liquidity_role=trade.liquidity_role.value,
        )
        for trade in trades
    ]

    return ExecutionReport(
        strategy_id=strategy_id,
        generated_orders=len(order_records),
        fills_total=len(fill_records),
        filled_orders=sum(1 for order in order_records if order.final_status == OrderStatus.FILLED.value),
        canceled_orders=sum(1 for order in order_records if order.final_status == OrderStatus.CANCELED.value),
        rejected_orders=sum(1 for order in order_records if order.final_status == OrderStatus.REJECTED.value),
        active_orders=sum(
            1
            for order in order_records
            if order.final_status in {OrderStatus.ACTIVE.value, OrderStatus.PARTIALLY_FILLED.value}
        ),
        orders=order_records,
        fills=fill_records,
    )


def format_execution_report(report: ExecutionReport) -> str:
    lines = [
        "execution_report:",
        f"execution_orders_total={report.generated_orders}",
        f"execution_fills_total={report.fills_total}",
        f"execution_filled_orders={report.filled_orders}",
        f"execution_canceled_orders={report.canceled_orders}",
        f"execution_rejected_orders={report.rejected_orders}",
        f"execution_active_orders={report.active_orders}",
    ]
    for order in report.orders:
        lines.append(
            " | ".join(
                [
                    order.order_id,
                    order.side,
                    order.order_type,
                    f"submitted_at={order.submitted_at}",
                    f"submitted_price={order.submitted_price:.6f}",
                    f"submitted_qty={order.submitted_qty:.6f}",
                    f"status={order.final_status}",
                    f"filled_qty={order.filled_qty_total:.6f}",
                    f"avg_fill_price={order.avg_fill_price:.6f}" if order.avg_fill_price is not None else "avg_fill_price=None",
                    f"commission_total={order.commission_total:.6f}",
                    f"slippage_cost_total={order.slippage_cost_total:.6f}",
                ]
            )
        )
    return "\n".join(lines)


def write_execution_report_json(report: ExecutionReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_id": report.strategy_id,
        "generated_orders": report.generated_orders,
        "fills_total": report.fills_total,
        "filled_orders": report.filled_orders,
        "canceled_orders": report.canceled_orders,
        "rejected_orders": report.rejected_orders,
        "active_orders": report.active_orders,
        "orders": [asdict(order) for order in report.orders],
        "fills": [asdict(fill) for fill in report.fills],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_execution_orders_csv(report: ExecutionReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(report.orders[0]).keys()) if report.orders else list(asdict(ExecutionOrderRecord(
            order_id="",
            strategy_id="",
            side="",
            order_type="",
            submitted_at="",
            submitted_price=0.0,
            submitted_qty=0.0,
            final_status="",
            final_updated_at="",
            filled_qty_total=0.0,
            remaining_qty=0.0,
            avg_fill_price=None,
            first_fill_at=None,
            last_fill_at=None,
            trade_count=0,
            commission_total=0.0,
            slippage_cost_total=0.0,
            rejection_reason=None,
        )).keys()))
        writer.writeheader()
        for order in report.orders:
            writer.writerow(asdict(order))
    return output_path


def write_execution_fills_csv(report: ExecutionReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(report.fills[0]).keys()) if report.fills else list(asdict(ExecutionFillRecord(
            trade_id="",
            order_id="",
            strategy_id="",
            timestamp="",
            side="",
            qty=0.0,
            exec_price=0.0,
            raw_price=None,
            commission=0.0,
            notional=0.0,
            liquidity_role="",
        )).keys()))
        writer.writeheader()
        for fill in report.fills:
            writer.writerow(asdict(fill))
    return output_path
