from __future__ import annotations

import json
from datetime import datetime

from src.actions import PlaceOrderAction
from src.engine import SimulationEngine
from src.enums import OrderType, Side
from src.reporting import (
    build_execution_report,
    format_execution_report,
    write_execution_fills_csv,
    write_execution_orders_csv,
    write_execution_report_json,
)
from tests.helpers.snapshots import make_crossable_snapshot_for_buy
from tests.helpers.strategies import ActionReturningStrategy


def test_build_execution_report_collects_order_and_fill_details(tmp_path) -> None:
    strategy = ActionReturningStrategy(
        "strategy-1",
        actions_by_step=[
            [
                PlaceOrderAction(
                    strategy_id="strategy-1",
                    side=Side.BUY,
                    price=100.0,
                    qty=2.0,
                    order_type=OrderType.LIMIT,
                )
            ]
        ],
    )
    engine = SimulationEngine(strategy=strategy)

    engine.process_snapshot(make_crossable_snapshot_for_buy(datetime(2024, 1, 1, 10, 0, 0)))

    report = build_execution_report(engine, strategy_id="strategy-1")

    assert report.generated_orders == 1
    assert report.fills_total == 1
    assert report.filled_orders == 1
    assert report.orders[0].order_id == "ORD-000001"
    assert report.orders[0].final_status == "FILLED"
    assert report.orders[0].filled_qty_total == 2.0
    assert report.orders[0].avg_fill_price == 100.0
    assert report.orders[0].first_fill_at == "2024-01-01T10:00:00"
    assert report.fills[0].trade_id == "TRD-000001"
    assert report.fills[0].exec_price == 100.0

    json_path = tmp_path / "execution_report.json"
    orders_csv_path = tmp_path / "orders.csv"
    fills_csv_path = tmp_path / "fills.csv"

    write_execution_report_json(report, json_path)
    write_execution_orders_csv(report, orders_csv_path)
    write_execution_fills_csv(report, fills_csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["generated_orders"] == 1
    assert payload["orders"][0]["order_id"] == "ORD-000001"
    assert "order_id" in orders_csv_path.read_text(encoding="utf-8")
    assert "trade_id" in fills_csv_path.read_text(encoding="utf-8")


def test_format_execution_report_includes_order_log_line() -> None:
    strategy = ActionReturningStrategy(
        "strategy-1",
        actions_by_step=[
            [
                PlaceOrderAction(
                    strategy_id="strategy-1",
                    side=Side.BUY,
                    price=100.0,
                    qty=1.0,
                    order_type=OrderType.LIMIT,
                )
            ]
        ],
    )
    engine = SimulationEngine(strategy=strategy)
    engine.process_snapshot(make_crossable_snapshot_for_buy(datetime(2024, 1, 1, 10, 0, 0)))

    rendered = format_execution_report(build_execution_report(engine, strategy_id="strategy-1"))

    assert "execution_report:" in rendered
    assert "ORD-000001 | BUY | LIMIT" in rendered
