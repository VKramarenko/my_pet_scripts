from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sim.portfolio.metrics import MetricsCollector


def _to_equity_points(metrics: MetricsCollector) -> list[dict[str, float]]:
    return [{"ts": ts, "equity": equity} for ts, equity in metrics.equity_curve]


def _build_stats(metrics: MetricsCollector) -> dict[str, float | int | None]:
    points = _to_equity_points(metrics)
    if not points:
        return {
            "num_points": 0,
            "num_fills": metrics.num_fills,
            "start_equity": None,
            "last_equity": None,
            "pnl_abs": None,
            "pnl_pct": None,
        }
    start_equity = points[0]["equity"]
    last_equity = points[-1]["equity"]
    pnl_abs = last_equity - start_equity
    pnl_pct = None if start_equity == 0 else pnl_abs / abs(start_equity) * 100.0
    return {
        "num_points": len(points),
        "num_fills": metrics.num_fills,
        "start_equity": start_equity,
        "last_equity": last_equity,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
    }


def _build_equity_figure(metrics: MetricsCollector):
    import plotly.graph_objects as go

    points = _to_equity_points(metrics)
    x_axis = [_ts_to_datetime(point["ts"]) for point in points]
    y_axis = [point["equity"] for point in points]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=x_axis, y=y_axis, mode="lines", name="Equity", line={"width": 2})
    )
    figure.update_layout(
        title="Backtest Equity Curve",
        xaxis_title="Datetime (UTC)",
        yaxis_title="Equity",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        xaxis={
            "tickangle": -45,
            "tickformat": "%Y-%m-%d %H:%M:%S",
            "type": "date",
        },
    )
    return figure


def _build_stats_panel(stats: dict[str, float | int | None]) -> list[str]:
    pnl_pct = stats["pnl_pct"]
    pnl_pct_text = "n/a" if pnl_pct is None else f"{pnl_pct:.4f}%"
    return [
        f"Equity points: {stats['num_points']}",
        f"Fills: {stats['num_fills']}",
        "Start equity: "
        + ("n/a" if stats["start_equity"] is None else f"{stats['start_equity']:.6f}"),
        "Last equity: "
        + ("n/a" if stats["last_equity"] is None else f"{stats['last_equity']:.6f}"),
        "PnL: " + ("n/a" if stats["pnl_abs"] is None else f"{stats['pnl_abs']:.6f}"),
        f"PnL %: {pnl_pct_text}",
    ]


def _ts_to_datetime(ts: float):
    from datetime import datetime

    return datetime.utcfromtimestamp(ts)


def _build_tob_trades_figure(metrics: MetricsCollector):
    import plotly.graph_objects as go

    figure = go.Figure()

    if metrics.tob_curve:
        dt_tob = [_ts_to_datetime(t[0]) for t in metrics.tob_curve]
        mid_tob = [t[1] for t in metrics.tob_curve]
        figure.add_trace(
            go.Scatter(
                x=dt_tob,
                y=mid_tob,
                mode="lines",
                name="TOB (mid)",
                line={"width": 2, "color": "#1f77b4"},
            )
        )

    if metrics.market_trades:
        buy_trades = [t for t in metrics.market_trades if t["side"] == "buyer_initiated"]
        sell_trades = [t for t in metrics.market_trades if t["side"] == "seller_initiated"]

        if buy_trades:
            dt_buy = [_ts_to_datetime(t["ts"]) for t in buy_trades]
            figure.add_trace(
                go.Scatter(
                    x=dt_buy,
                    y=[t["price"] for t in buy_trades],
                    mode="markers",
                    name="BUY",
                    marker={"size": 10, "color": "#2ca02c", "symbol": "triangle-up"},
                    customdata=[t["size"] for t in buy_trades],
                    hovertemplate="%{x}<br>price: %{y}<br>size: %{customdata}<extra></extra>",
                )
            )
        if sell_trades:
            dt_sell = [_ts_to_datetime(t["ts"]) for t in sell_trades]
            figure.add_trace(
                go.Scatter(
                    x=dt_sell,
                    y=[t["price"] for t in sell_trades],
                    mode="markers",
                    name="SELL",
                    marker={"size": 10, "color": "#d62728", "symbol": "triangle-down"},
                    customdata=[t["size"] for t in sell_trades],
                    hovertemplate="%{x}<br>price: %{y}<br>size: %{customdata}<extra></extra>",
                )
            )

    if metrics.fill_events:
        strat_buy = [f for f in metrics.fill_events if f["side"] == "BUY"]
        strat_sell = [f for f in metrics.fill_events if f["side"] == "SELL"]
        if strat_buy:
            dt_buy = [_ts_to_datetime(f["ts"]) for f in strat_buy]
            figure.add_trace(
                go.Scatter(
                    x=dt_buy,
                    y=[f["price"] for f in strat_buy],
                    mode="markers",
                    name="Strategy BUY",
                    marker={"size": 12, "color": "#0066cc", "symbol": "diamond"},
                    customdata=[f["qty"] for f in strat_buy],
                    hovertemplate="%{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>",
                )
            )
        if strat_sell:
            dt_sell = [_ts_to_datetime(f["ts"]) for f in strat_sell]
            figure.add_trace(
                go.Scatter(
                    x=dt_sell,
                    y=[f["price"] for f in strat_sell],
                    mode="markers",
                    name="Strategy SELL",
                    marker={"size": 12, "color": "#cc6600", "symbol": "diamond"},
                    customdata=[f["qty"] for f in strat_sell],
                    hovertemplate="%{x}<br>price: %{y}<br>qty: %{customdata}<extra></extra>",
                )
            )

    figure.update_layout(
        title="Top of Book, Market Trades & Strategy Fills",
        xaxis_title="Datetime (UTC)",
        yaxis_title="Price",
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        showlegend=True,
        xaxis={
            "tickangle": -45,
            "tickformat": "%Y-%m-%d %H:%M:%S",
            "type": "date",
        },
    )
    return figure


def _to_fills_rows(metrics: MetricsCollector) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for event in metrics.fill_events:
        rows.append(
            {
                "ts": _ts_to_datetime(float(event["ts"])).strftime("%Y-%m-%d %H:%M:%S"),
                "order_id": str(event["order_id"]),
                "side": str(event["side"]),
                "price": float(event["price"]),
                "qty": float(event["qty"]),
                "fee": float(event["fee"]),
                "liquidity": str(event["liquidity"]),
                "notional": float(event["notional"]),
            }
        )
    return rows


def create_dash_app(
    initial_metrics: MetricsCollector,
    run_backtest: Callable[[dict[str, Any]], MetricsCollector],
    initial_form: dict[str, Any],
):
    """
    Build a Dash application for backtest metrics.

    Dash imports stay local so the simulator can run without UI dependencies.
    """
    try:
        from dash import Dash, Input, Output, State, dash_table, dcc, html
    except ImportError as exc:
        raise RuntimeError(
            "Dash visualization requires optional dependencies. "
            "Install them with: pip install dash plotly"
        ) from exc

    stats = _build_stats(initial_metrics)
    fills_rows = _to_fills_rows(initial_metrics)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Strategy Simulation Dashboard"),
            html.Div(
                [
                    html.H4("Run configuration"),
                    dcc.Input(
                        id="l2-input",
                        value=initial_form["l2"],
                        type="text",
                        placeholder="Path to L2 file",
                        style={"width": "100%", "marginBottom": "8px"},
                    ),
                    dcc.Input(
                        id="trades-input",
                        value=initial_form["trades"],
                        type="text",
                        placeholder="Path to trades file (optional)",
                        style={"width": "100%", "marginBottom": "8px"},
                    ),
                    dcc.Dropdown(
                        id="loader-input",
                        options=[
                            {"label": "default", "value": "default"},
                            {"label": "bybit", "value": "bybit"},
                            {"label": "test_data", "value": "test_data"},
                            {"label": "wide", "value": "wide"},
                        ],
                        value=initial_form["loader"],
                        clearable=False,
                        style={"marginBottom": "8px"},
                    ),
                    dcc.Input(
                        id="symbol-input",
                        value=initial_form.get("symbol") or "",
                        type="text",
                        placeholder="Symbol filter (wide loader, optional)",
                        style={"width": "100%", "marginBottom": "8px"},
                    ),
                    dcc.Dropdown(
                        id="strategy-input",
                        options=[
                            {"label": "mm", "value": "mm"},
                            {"label": "taker", "value": "taker"},
                        ],
                        value=initial_form["strategy"],
                        clearable=False,
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("MM spread"),
                                    dcc.Input(
                                        id="mm-spread-input",
                                        value=initial_form["mm_spread"],
                                        type="number",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("MM quote size"),
                                    dcc.Input(
                                        id="mm-quote-size-input",
                                        value=initial_form["mm_quote_size"],
                                        type="number",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Taker window"),
                                    dcc.Input(
                                        id="taker-window-input",
                                        value=initial_form["taker_window"],
                                        type="number",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Taker std mult"),
                                    dcc.Input(
                                        id="taker-std-mult-input",
                                        value=initial_form["taker_std_mult"],
                                        type="number",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Taker qty"),
                                    dcc.Input(
                                        id="taker-qty-input",
                                        value=initial_form["taker_qty"],
                                        type="number",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Taker cooldown"),
                                    dcc.Input(
                                        id="taker-cooldown-input",
                                        value=initial_form["taker_cooldown"],
                                        type="number",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Taker max position"),
                                    dcc.Input(
                                        id="taker-max-position-input",
                                        value=initial_form["taker_max_position"],
                                        type="number",
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "8px", "marginBottom": "8px"},
                    ),
                    html.Button("Run", id="run-button", n_clicks=0),
                    html.Div(
                        id="run-status",
                        children="Ready",
                        style={"marginTop": "8px", "fontStyle": "italic"},
                    ),
                ],
                style={"marginBottom": "16px"},
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Equity",
                        children=[dcc.Graph(id="equity-graph", figure=_build_equity_figure(initial_metrics))],
                    ),
                    dcc.Tab(
                        label="TOB & Trades",
                        children=[
                            dcc.Graph(
                                id="tob-trades-graph",
                                figure=_build_tob_trades_figure(initial_metrics),
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Stats",
                        children=[
                            html.Ul(
                                [html.Li(row) for row in _build_stats_panel(stats)],
                                id="stats-list",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Fills",
                        children=[
                            dash_table.DataTable(
                                id="fills-table",
                                columns=[
                                    {"name": "datetime (UTC)", "id": "ts"},
                                    {"name": "order_id", "id": "order_id"},
                                    {"name": "side", "id": "side"},
                                    {"name": "price", "id": "price"},
                                    {"name": "qty", "id": "qty"},
                                    {"name": "fee", "id": "fee"},
                                    {"name": "liquidity", "id": "liquidity"},
                                    {"name": "notional", "id": "notional"},
                                ],
                                data=fills_rows,
                                page_size=15,
                                style_table={"overflowX": "auto"},
                            )
                        ],
                    ),
                ]
            ),
        ],
        style={"maxWidth": "1000px", "margin": "0 auto", "padding": "16px"},
    )

    @app.callback(
        Output("run-status", "children"),
        Output("equity-graph", "figure"),
        Output("stats-list", "children"),
        Output("fills-table", "data"),
        Output("tob-trades-graph", "figure"),
        Input("run-button", "n_clicks"),
        State("l2-input", "value"),
        State("trades-input", "value"),
        State("loader-input", "value"),
        State("symbol-input", "value"),
        State("strategy-input", "value"),
        State("mm-spread-input", "value"),
        State("mm-quote-size-input", "value"),
        State("taker-window-input", "value"),
        State("taker-std-mult-input", "value"),
        State("taker-qty-input", "value"),
        State("taker-cooldown-input", "value"),
        State("taker-max-position-input", "value"),
        prevent_initial_call=True,
    )
    def _run_callback(
        _n_clicks: int,
        l2: str,
        trades: str | None,
        loader: str,
        symbol_filter: str | None,
        strategy: str,
        mm_spread: float,
        mm_quote_size: float,
        taker_window: int,
        taker_std_mult: float,
        taker_qty: float,
        taker_cooldown: float,
        taker_max_position: float,
    ):
        form = {
            "l2": l2 or "",
            "trades": trades or None,
            "loader": loader,
            "symbol": symbol_filter or None,
            "strategy": strategy,
            "mm_spread": float(mm_spread),
            "mm_quote_size": float(mm_quote_size),
            "taker_window": int(taker_window),
            "taker_std_mult": float(taker_std_mult),
            "taker_qty": float(taker_qty),
            "taker_cooldown": float(taker_cooldown),
            "taker_max_position": float(taker_max_position),
        }
        try:
            metrics = run_backtest(form)
        except Exception as exc:  # pragma: no cover - UI error path
            return (
                f"Run failed: {exc}",
                _build_equity_figure(initial_metrics),
                [html.Li(row) for row in _build_stats_panel(stats)],
                fills_rows,
                _build_tob_trades_figure(initial_metrics),
            )
        run_stats = _build_stats(metrics)
        return (
            "Run completed",
            _build_equity_figure(metrics),
            [html.Li(row) for row in _build_stats_panel(run_stats)],
            _to_fills_rows(metrics),
            _build_tob_trades_figure(metrics),
        )

    return app
