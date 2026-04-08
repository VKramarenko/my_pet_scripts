from __future__ import annotations

import json
from datetime import datetime

import pytest

from src.actions import PlaceOrderAction
from src.engine import SimulationEngine
from src.enums import OrderType, Side
from src.reporting import build_execution_report, write_execution_report_json
from tests.helpers.snapshots import make_crossable_snapshot_for_buy
from tests.helpers.strategies import ActionReturningStrategy
from view_execution_dash import (
    build_cumulative_position_series,
    build_execution_figure,
    build_price_series,
    load_execution_report,
)


def test_build_price_series_reads_book_prices(tmp_path) -> None:
    csv_path = tmp_path / "book.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,101,100,1,1\n"
        "2024-01-01T10:00:01,102,99,1,1\n",
        encoding="utf-8",
    )

    series = build_price_series(csv_path)

    assert len(series) == 2
    assert series[0]["best_bid"] == 100.0
    assert series[0]["best_ask"] == 101.0
    assert series[0]["mid_price"] == 100.5


def test_dash_helpers_build_position_series_and_figure(tmp_path) -> None:
    pytest.importorskip("plotly")

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

    report_path = tmp_path / "execution_report.json"
    write_execution_report_json(build_execution_report(engine, strategy_id="strategy-1"), report_path)
    report = load_execution_report(report_path)

    position_points = build_cumulative_position_series(report)
    assert position_points == [{"timestamp": "2024-01-01T10:00:00", "position": 1.0}]

    price_series = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "best_bid": 99.0,
            "best_ask": 100.0,
            "mid_price": 99.5,
        }
    ]
    figure = build_execution_figure(price_series, report)

    assert len(figure.data) >= 5
    assert figure.layout.title.text == "Execution Dashboard"


def test_load_execution_report_reads_json_payload(tmp_path) -> None:
    report_path = tmp_path / "execution_report.json"
    report_path.write_text(json.dumps({"orders": [], "fills": []}), encoding="utf-8")

    payload = load_execution_report(report_path)

    assert payload == {"orders": [], "fills": []}
