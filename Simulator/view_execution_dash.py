from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_simulation import infer_csv_depth
from src.data_loader import load_snapshots_csv
from src.validation import CSVSnapshotLoaderConfig


def load_execution_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_price_series(csv_path: str | Path, depth: int | None = None) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    effective_depth = depth if depth is not None else infer_csv_depth(csv_path)
    snapshots = load_snapshots_csv(csv_path, CSVSnapshotLoaderConfig(depth=effective_depth))
    return [
        {
            "timestamp": snapshot.timestamp.isoformat(),
            "best_bid": snapshot.best_bid().price if snapshot.best_bid() is not None else None,
            "best_ask": snapshot.best_ask().price if snapshot.best_ask() is not None else None,
            "mid_price": snapshot.mid_price(),
        }
        for snapshot in snapshots
    ]


def build_cumulative_position_series(report: dict[str, Any]) -> list[dict[str, Any]]:
    cumulative_position = 0.0
    points: list[dict[str, Any]] = []
    fills = sorted(report.get("fills", []), key=lambda fill: (fill["timestamp"], fill["trade_id"]))
    for fill in fills:
        signed_qty = fill["qty"] if fill["side"] == "BUY" else -fill["qty"]
        cumulative_position += signed_qty
        points.append(
            {
                "timestamp": fill["timestamp"],
                "position": cumulative_position,
            }
        )
    return points


def build_execution_figure(price_series: list[dict[str, Any]], report: dict[str, Any]):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for visualization. Install it with: python -m pip install plotly dash"
        ) from exc

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Book Prices And Execution", "Position From Fills"),
    )

    timestamps = [point["timestamp"] for point in price_series]
    best_bids = [point["best_bid"] for point in price_series]
    best_asks = [point["best_ask"] for point in price_series]
    mid_prices = [point["mid_price"] for point in price_series]

    figure.add_trace(
        go.Scatter(x=timestamps, y=best_bids, name="Best Bid", mode="lines", line={"color": "#1f77b4"}),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=timestamps, y=best_asks, name="Best Ask", mode="lines", line={"color": "#d62728"}),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=mid_prices,
            name="Mid Price",
            mode="lines",
            line={"color": "#2ca02c", "dash": "dash"},
        ),
        row=1,
        col=1,
    )

    orders = report.get("orders", [])
    order_x = [order["submitted_at"] for order in orders]
    order_y = [order["submitted_price"] for order in orders]
    order_symbols = ["triangle-up" if order["side"] == "BUY" else "triangle-down" for order in orders]
    order_colors = [
        {
            "FILLED": "#2ca02c",
            "PARTIALLY_FILLED": "#bcbd22",
            "ACTIVE": "#1f77b4",
            "CANCELED": "#7f7f7f",
            "REJECTED": "#d62728",
        }.get(order["final_status"], "#9467bd")
        for order in orders
    ]
    order_text = [
        (
            f"{order['order_id']} | {order['side']} {order['submitted_qty']} @ {order['submitted_price']}<br>"
            f"status={order['final_status']} | filled={order['filled_qty_total']}"
        )
        for order in orders
    ]
    figure.add_trace(
        go.Scatter(
            x=order_x,
            y=order_y,
            name="Submitted Orders",
            mode="markers",
            text=order_text,
            hoverinfo="text",
            marker={
                "size": 11,
                "symbol": order_symbols,
                "color": order_colors,
                "line": {"width": 1, "color": "#111111"},
            },
        ),
        row=1,
        col=1,
    )

    fills = report.get("fills", [])
    figure.add_trace(
        go.Scatter(
            x=[fill["timestamp"] for fill in fills],
            y=[fill["exec_price"] for fill in fills],
            name="Executed Fills",
            mode="markers",
            text=[
                (
                    f"{fill['trade_id']} | {fill['side']} {fill['qty']} @ {fill['exec_price']}<br>"
                    f"order={fill['order_id']} | commission={fill['commission']}"
                )
                for fill in fills
            ],
            hoverinfo="text",
            marker={
                "size": 10,
                "symbol": "x",
                "color": "#ff7f0e",
                "line": {"width": 1, "color": "#111111"},
            },
        ),
        row=1,
        col=1,
    )

    position_points = build_cumulative_position_series(report)
    figure.add_trace(
        go.Scatter(
            x=[point["timestamp"] for point in position_points],
            y=[point["position"] for point in position_points],
            name="Position",
            mode="lines+markers",
            line={"color": "#8c564b"},
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        template="plotly_white",
        height=860,
        title="Execution Dashboard",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Position", row=2, col=1)
    figure.update_xaxes(title_text="Timestamp", row=2, col=1)
    return figure


def create_dash_app(price_series: list[dict[str, Any]], report: dict[str, Any]):
    try:
        from dash import Dash, dash_table, dcc, html
    except ImportError as exc:
        raise RuntimeError(
            "Dash is required for the dashboard UI. Install it with: python -m pip install dash plotly"
        ) from exc

    figure = build_execution_figure(price_series, report)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Simulation Execution Dashboard"),
            html.P(
                "Prices come from the source snapshot CSV, while executions come from the saved strategy execution report."
            ),
            dcc.Graph(figure=figure),
            html.H3("Order Summary"),
            dash_table.DataTable(
                data=report.get("orders", []),
                columns=[{"name": key, "id": key} for key in (report.get("orders", [{}])[0].keys() if report.get("orders") else [])],
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
            ),
            html.H3("Fill Details"),
            dash_table.DataTable(
                data=report.get("fills", []),
                columns=[{"name": key, "id": key} for key in (report.get("fills", [{}])[0].keys() if report.get("fills") else [])],
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
            ),
        ],
        style={"maxWidth": "1400px", "margin": "0 auto", "padding": "24px"},
    )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize simulation execution against source order book prices.")
    parser.add_argument("--csv", required=True, help="Path to source snapshots CSV file.")
    parser.add_argument("--execution-report-json", required=True, help="Path to execution report JSON created by run_simulation.py.")
    parser.add_argument("--depth", type=int, default=None, help="Optional explicit CSV depth. If omitted, inferred from CSV header.")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port.")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = load_execution_report(args.execution_report_json)
    price_series = build_price_series(args.csv, depth=args.depth)
    app = create_dash_app(price_series, report)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
