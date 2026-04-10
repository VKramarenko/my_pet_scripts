from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_simulation import infer_csv_depth, peek_instrument_id
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


def _filter_by_instrument(records: list[dict[str, Any]], instrument_id: str) -> list[dict[str, Any]]:
    """Filter records by instrument_id. Falls back to all records for old reports without the field."""
    if not records or "instrument_id" not in records[0]:
        return records
    return [r for r in records if r.get("instrument_id") == instrument_id]


def build_cumulative_position_series(report_or_fills: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accepts either a full report dict (legacy) or a fills list directly."""
    fills = report_or_fills.get("fills", []) if isinstance(report_or_fills, dict) else report_or_fills
    cumulative_position = 0.0
    points: list[dict[str, Any]] = []
    sorted_fills = sorted(fills, key=lambda f: (f["timestamp"], f["trade_id"]))
    for fill in sorted_fills:
        signed_qty = fill["qty"] if fill["side"] == "BUY" else -fill["qty"]
        cumulative_position += signed_qty
        points.append({"timestamp": fill["timestamp"], "position": cumulative_position})
    return points


def build_instrument_figure(
    instrument_id: str,
    price_series: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
):
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
        subplot_titles=(
            f"{instrument_id} — Book Prices and Execution",
            f"{instrument_id} — Position from Fills",
        ),
    )

    timestamps = [p["timestamp"] for p in price_series]
    figure.add_trace(
        go.Scatter(x=timestamps, y=[p["best_bid"] for p in price_series],
                   name="Best Bid", mode="lines", line={"color": "#1f77b4"}),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(x=timestamps, y=[p["best_ask"] for p in price_series],
                   name="Best Ask", mode="lines", line={"color": "#d62728"}),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(x=timestamps, y=[p["mid_price"] for p in price_series],
                   name="Mid Price", mode="lines",
                   line={"color": "#2ca02c", "dash": "dash"}),
        row=1, col=1,
    )

    if orders:
        order_colors = [
            {"FILLED": "#2ca02c", "PARTIALLY_FILLED": "#bcbd22",
             "ACTIVE": "#1f77b4", "CANCELED": "#7f7f7f",
             "REJECTED": "#d62728"}.get(o["final_status"], "#9467bd")
            for o in orders
        ]
        figure.add_trace(
            go.Scatter(
                x=[o["submitted_at"] for o in orders],
                y=[o["submitted_price"] for o in orders],
                name="Submitted Orders",
                mode="markers",
                text=[
                    f"{o['order_id']} | {o['side']} {o['submitted_qty']} @ {o['submitted_price']}<br>"
                    f"status={o['final_status']} | filled={o['filled_qty_total']}"
                    for o in orders
                ],
                hoverinfo="text",
                marker={
                    "size": 11,
                    "symbol": ["triangle-up" if o["side"] == "BUY" else "triangle-down" for o in orders],
                    "color": order_colors,
                    "line": {"width": 1, "color": "#111111"},
                },
            ),
            row=1, col=1,
        )

    if fills:
        figure.add_trace(
            go.Scatter(
                x=[f["timestamp"] for f in fills],
                y=[f["exec_price"] for f in fills],
                name="Executed Fills",
                mode="markers",
                text=[
                    f"{f['trade_id']} | {f['side']} {f['qty']} @ {f['exec_price']}<br>"
                    f"order={f['order_id']} | commission={f['commission']}"
                    for f in fills
                ],
                hoverinfo="text",
                marker={
                    "size": 10,
                    "symbol": "x",
                    "color": "#ff7f0e",
                    "line": {"width": 1, "color": "#111111"},
                },
            ),
            row=1, col=1,
        )

    position_points = build_cumulative_position_series(fills)
    if position_points:
        figure.add_trace(
            go.Scatter(
                x=[p["timestamp"] for p in position_points],
                y=[p["position"] for p in position_points],
                name="Position",
                mode="lines+markers",
                line={"color": "#8c564b"},
            ),
            row=2, col=1,
        )

    figure.update_layout(
        template="plotly_white",
        height=720,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0.0},
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Position", row=2, col=1)
    figure.update_xaxes(title_text="Timestamp", row=2, col=1)
    return figure


def build_execution_figure(price_series: list[dict[str, Any]], report: dict[str, Any]):
    """Backward-compat wrapper: builds a single-instrument figure from a full report dict."""
    orders = report.get("orders", [])
    fills = report.get("fills", [])
    figure = build_instrument_figure("", price_series, orders, fills)
    figure.update_layout(title="Execution Dashboard")
    return figure


def create_dash_app(
    instruments: list[tuple[str, list[dict[str, Any]]]],
    report: dict[str, Any],
):
    """
    instruments: list of (instrument_id, price_series) — one entry per CSV.
    report: loaded execution report JSON dict.
    """
    try:
        from dash import Dash, dash_table, dcc, html
    except ImportError as exc:
        raise RuntimeError(
            "Dash is required for the dashboard UI. Install it with: python -m pip install dash plotly"
        ) from exc

    all_orders: list[dict[str, Any]] = report.get("orders", [])
    all_fills: list[dict[str, Any]] = report.get("fills", [])

    instrument_panels = []
    for instrument_id, price_series in instruments:
        inst_orders = _filter_by_instrument(all_orders, instrument_id)
        inst_fills = _filter_by_instrument(all_fills, instrument_id)
        fig = build_instrument_figure(instrument_id, price_series, inst_orders, inst_fills)
        instrument_panels.append(
            html.Div([
                html.H3(f"Instrument: {instrument_id}",
                        style={"marginTop": "32px", "marginBottom": "8px"}),
                dcc.Graph(figure=fig),
            ])
        )

    order_columns = [{"name": k, "id": k} for k in (all_orders[0].keys() if all_orders else [])]
    fill_columns = [{"name": k, "id": k} for k in (all_fills[0].keys() if all_fills else [])]

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Simulation Execution Dashboard"),
            html.P(
                "Prices come from the source snapshot CSVs. "
                "Each instrument is shown in its own chart. "
                "Executions come from the saved strategy execution report."
            ),
            *instrument_panels,
            html.H3("Order Summary", style={"marginTop": "40px"}),
            dash_table.DataTable(
                data=all_orders,
                columns=order_columns,
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
            ),
            html.H3("Fill Details"),
            dash_table.DataTable(
                data=all_fills,
                columns=fill_columns,
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
    parser = argparse.ArgumentParser(
        description="Visualize simulation execution against source order book prices."
    )
    # Single-book (backward compat)
    parser.add_argument("--csv", help="Path to source snapshots CSV (single-book mode).")
    # Multi-book
    parser.add_argument(
        "--trading-csv",
        action="append",
        dest="trading_csvs",
        metavar="PATH",
        help="Trading instrument CSV (repeatable). One chart per entry.",
    )
    parser.add_argument(
        "--execution-report-json",
        required=True,
        help="Path to execution report JSON created by run_simulation.py.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Optional explicit CSV depth. If omitted, inferred from CSV header.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port.")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_paths: list[Path] = []
    if args.trading_csvs:
        csv_paths.extend(Path(p) for p in args.trading_csvs)
    if args.csv:
        csv_paths.append(Path(args.csv))

    if not csv_paths:
        parser.error("Provide --csv or at least one --trading-csv.")

    report = load_execution_report(args.execution_report_json)

    instruments: list[tuple[str, list[dict[str, Any]]]] = []
    for csv_path in csv_paths:
        price_series = build_price_series(csv_path, depth=args.depth)
        instrument_id = peek_instrument_id(csv_path)
        instruments.append((instrument_id, price_series))

    app = create_dash_app(instruments, report)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
