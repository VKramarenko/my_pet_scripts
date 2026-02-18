"""Dash UI for the ToyOption calibration tool.

Run from repo root:   python -m ToyOption.app
Run directly:         python ToyOption/app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_root = _pkg_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import base64
import json
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, no_update
import plotly.graph_objects as go
import numpy as np

from ToyOption.data import CanonicalQuoteSet
from ToyOption.service import ModelService
from ToyOption.emulator import Trade, ReactionConfig, COMBINATION_MODELS
from ToyOption.black76 import implied_vols_from_prices

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
svc = ModelService()

EXAMPLE_DATA = {
    "F": 100.0,
    "T": 0.25,
    "calls": [
        {"K": 90, "price": 12.5},
        {"K": 95, "price": 8.3},
        {"K": 100, "price": 5.1},
        {"K": 105, "price": 2.8},
    ],
    "puts": [
        {"K": 90, "price": 2.4},
        {"K": 95, "price": 3.2},
        {"K": 100, "price": 5.0},
        {"K": 105, "price": 7.7},
    ],
}

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _make_param_controls(model):
    """Generate slider + numeric input for each model parameter."""
    children = []
    for i, name in enumerate(model.param_names):
        lo, hi = model.bounds()[i]
        default = float(model.default_params()[i])
        children.append(html.Div([
            html.Label(name, style={"fontWeight": "bold", "marginRight": "8px"}),
            dcc.Input(
                id={"type": "param-input", "index": i},
                type="number",
                value=round(default, 4),
                step=0.01,
                style={"width": "80px", "marginRight": "8px"},
            ),
            dcc.Slider(
                id={"type": "param-slider", "index": i},
                min=lo,
                max=hi,
                value=default,
                step=(hi - lo) / 200,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"marginBottom": "12px"}))
    return children


def build_layout():
    model = svc.model
    return html.Div([
        html.H2("ToyOption Calibration Tool",
                 style={"textAlign": "center", "marginBottom": "4px"}),
        html.P("Research tool — direct price model (no Black-Scholes)",
               style={"textAlign": "center", "color": "#666", "marginTop": 0}),

        html.Div([
            # ---- LEFT: Data Input ----
            html.Div([
                html.H4("Data Input"),
                html.Div([
                    html.Label("Forward (F)"),
                    dcc.Input(id="input-F", type="number", value=100.0,
                              style={"width": "100%"}),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Label("Time to expiry (T, years)"),
                    dcc.Input(id="input-T", type="number", value=0.25,
                              step=0.01, style={"width": "100%"}),
                ], style={"marginBottom": "12px"}),

                html.H5("Calls"),
                dash_table.DataTable(
                    id="table-calls",
                    columns=[
                        {"name": "K", "id": "K", "type": "numeric", "editable": True},
                        {"name": "Price", "id": "price", "type": "numeric", "editable": True},
                    ],
                    data=[{"K": r["K"], "price": r["price"]} for r in EXAMPLE_DATA["calls"]],
                    editable=True,
                    row_deletable=True,
                    style_table={"marginBottom": "8px"},
                    style_cell={"textAlign": "center", "padding": "4px"},
                ),
                html.Button("+ Add call", id="btn-add-call", n_clicks=0,
                            style={"marginBottom": "12px", "fontSize": "12px"}),

                html.H5("Puts"),
                dash_table.DataTable(
                    id="table-puts",
                    columns=[
                        {"name": "K", "id": "K", "type": "numeric", "editable": True},
                        {"name": "Price", "id": "price", "type": "numeric", "editable": True},
                    ],
                    data=[{"K": r["K"], "price": r["price"]} for r in EXAMPLE_DATA["puts"]],
                    editable=True,
                    row_deletable=True,
                    style_table={"marginBottom": "8px"},
                    style_cell={"textAlign": "center", "padding": "4px"},
                ),
                html.Button("+ Add put", id="btn-add-put", n_clicks=0,
                            style={"marginBottom": "12px", "fontSize": "12px"}),

                # --- CSV Upload ---
                dcc.Upload(
                    id="upload-csv",
                    children=html.Button("Upload CSV",
                                         style={"width": "100%", "marginTop": "8px",
                                                "cursor": "pointer"}),
                    accept=".csv",
                ),
                html.Div(id="upload-status",
                         style={"fontSize": "12px", "color": "#666",
                                "marginTop": "4px"}),

                html.Button("Load Example", id="btn-example", n_clicks=0,
                            style={"width": "100%", "marginTop": "8px"}),

                html.Div([
                    dcc.Input(id="input-filename", type="text",
                              value="toy_option_data.csv",
                              placeholder="filename.csv",
                              style={"width": "calc(100% - 8px)", "marginBottom": "4px",
                                     "fontSize": "12px"}),
                    html.Button("Save to CSV", id="btn-save-csv", n_clicks=0,
                                style={"width": "100%", "cursor": "pointer"}),
                ], style={"marginTop": "8px"}),
                dcc.Download(id="download-csv"),
            ], style={"width": "22%", "padding": "12px", "verticalAlign": "top",
                       "display": "inline-block", "borderRight": "1px solid #ddd"}),

            # ---- MIDDLE: Model & Calibration ----
            html.Div([
                html.H4("Model & Calibration"),
                html.Div([
                    html.Label("Model"),
                    dcc.Dropdown(
                        id="dropdown-model",
                        options=[{"label": n, "value": n} for n in svc.available_models()],
                        value=model.name,
                        clearable=False,
                    ),
                ], style={"marginBottom": "16px"}),

                html.Div(id="param-controls", children=_make_param_controls(model)),

                html.Div([
                    html.Button("Calibrate", id="btn-calibrate", n_clicks=0,
                                style={"marginRight": "8px", "fontWeight": "bold",
                                        "backgroundColor": "#4CAF50", "color": "white",
                                        "border": "none", "padding": "8px 20px",
                                        "cursor": "pointer"}),
                    html.Button("Reset", id="btn-reset", n_clicks=0,
                                style={"padding": "8px 20px", "cursor": "pointer"}),
                ], style={"marginTop": "16px", "marginBottom": "12px"}),

                html.Div(id="metrics-text",
                         style={"whiteSpace": "pre-wrap", "fontFamily": "monospace",
                                "fontSize": "13px", "backgroundColor": "#f8f8f8",
                                "padding": "8px", "borderRadius": "4px"}),

                # ---- Reaction Emulator ----
                html.Hr(style={"marginTop": "20px"}),
                html.H4("Reaction Emulator"),
                html.Div([
                    html.Label("Combination Model"),
                    dcc.Dropdown(
                        id="dropdown-reaction-mode",
                        options=[{"label": n, "value": n} for n in COMBINATION_MODELS],
                        value="Linear",
                        clearable=False,
                    ),
                ], style={"marginBottom": "8px"}),
                html.Div([
                    html.Label("shift_atm"),
                    dcc.Input(id="input-shift-atm", type="number", value=1.0,
                              step=0.1, style={"width": "100%"}),
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Label("shift_wing"),
                    dcc.Input(id="input-shift-wing", type="number", value=1.0,
                              step=0.1, style={"width": "100%"}),
                ], style={"marginBottom": "6px"}),
                html.Div([
                    html.Label("volume_ref"),
                    dcc.Input(id="input-volume-ref", type="number", value=100.0,
                              step=10, style={"width": "100%"}),
                ], style={"marginBottom": "10px"}),

                html.H5("Trades"),
                dash_table.DataTable(
                    id="table-trades",
                    columns=[
                        {"name": "Side", "id": "side", "presentation": "dropdown", "editable": True},
                        {"name": "Strike", "id": "strike", "type": "numeric", "editable": True},
                        {"name": "Volume", "id": "volume", "type": "numeric", "editable": True},
                        {"name": "Direction", "id": "direction", "presentation": "dropdown", "editable": True},
                    ],
                    data=[],
                    editable=True,
                    row_deletable=True,
                    dropdown={
                        "side": {"options": [
                            {"label": "call", "value": "call"},
                            {"label": "put", "value": "put"},
                        ]},
                        "direction": {"options": [
                            {"label": "buy", "value": "buy"},
                            {"label": "sell", "value": "sell"},
                        ]},
                    },
                    style_table={"marginBottom": "8px"},
                    style_cell={"textAlign": "center", "padding": "4px", "fontSize": "12px"},
                ),
                html.Button("+ Add trade", id="btn-add-trade", n_clicks=0,
                            style={"fontSize": "12px", "marginRight": "6px"}),
                html.Div([
                    html.Button("Apply Trades", id="btn-apply-trades", n_clicks=0,
                                style={"marginRight": "8px", "fontWeight": "bold",
                                        "backgroundColor": "#2196F3", "color": "white",
                                        "border": "none", "padding": "8px 16px",
                                        "cursor": "pointer"}),
                    html.Button("Reset Emulator", id="btn-reset-emulator", n_clicks=0,
                                style={"padding": "8px 16px", "cursor": "pointer"}),
                ], style={"marginTop": "8px", "marginBottom": "8px"}),
                html.Div(id="emulator-status",
                         style={"whiteSpace": "pre-wrap", "fontFamily": "monospace",
                                "fontSize": "12px", "color": "#666",
                                "marginTop": "4px"}),
            ], style={"width": "26%", "padding": "12px", "verticalAlign": "top",
                       "display": "inline-block", "borderRight": "1px solid #ddd"}),

            # ---- RIGHT: Plots ----
            html.Div([
                dcc.Graph(id="graph-prices", style={"height": "420px"}),
                dcc.Graph(id="graph-residuals", style={"height": "250px"}),
                dcc.Graph(id="graph-iv", style={"height": "350px"}),
                html.Div(id="noarb-text",
                         style={"whiteSpace": "pre-wrap", "fontFamily": "monospace",
                                "fontSize": "12px", "padding": "8px",
                                "backgroundColor": "#f8f8f8", "borderRadius": "4px"}),
            ], style={"width": "50%", "padding": "12px", "verticalAlign": "top",
                       "display": "inline-block"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),

        # Hidden stores for triggering updates
        dcc.Store(id="store-params-json"),
        dcc.Store(id="store-base-params-json"),
    ])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = build_layout

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# ---- Switch model ----
@callback(
    Output("param-controls", "children"),
    Output("store-params-json", "data", allow_duplicate=True),
    Output("metrics-text", "children", allow_duplicate=True),
    Input("dropdown-model", "value"),
    prevent_initial_call=True,
)
def switch_model(model_name):
    svc.set_model(model_name)
    controls = _make_param_controls(svc.model)
    params_json = json.dumps([round(float(v), 6) for v in svc.params])
    return controls, params_json, ""


# Add row buttons
@callback(Output("table-calls", "data", allow_duplicate=True),
          Input("btn-add-call", "n_clicks"),
          State("table-calls", "data"),
          prevent_initial_call=True)
def add_call_row(n, rows):
    rows.append({"K": "", "price": ""})
    return rows

@callback(Output("table-puts", "data", allow_duplicate=True),
          Input("btn-add-put", "n_clicks"),
          State("table-puts", "data"),
          prevent_initial_call=True)
def add_put_row(n, rows):
    rows.append({"K": "", "price": ""})
    return rows

# ---- Upload CSV ----
@callback(
    Output("input-F", "value", allow_duplicate=True),
    Output("input-T", "value", allow_duplicate=True),
    Output("table-calls", "data", allow_duplicate=True),
    Output("table-puts", "data", allow_duplicate=True),
    Output("upload-status", "children"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def upload_csv(contents, filename):
    if contents is None:
        return no_update, no_update, no_update, no_update, ""
    try:
        # Dash uploads are base64-encoded
        _, content_b64 = contents.split(",", 1)
        decoded = base64.b64decode(content_b64).decode("utf-8")
        # Write to a temp file so from_csv can parse it
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                          delete=False, encoding="utf-8")
        tmp.write(decoded)
        tmp.close()
        try:
            qs = CanonicalQuoteSet.from_csv(tmp.name)
        finally:
            os.unlink(tmp.name)

        call_rows = [{"K": k, "price": p} for k, p, _ in qs.calls]
        put_rows = [{"K": k, "price": p} for k, p, _ in qs.puts]
        return qs.F, qs.T, call_rows, put_rows, f"Loaded {filename}"
    except Exception as e:
        return no_update, no_update, no_update, no_update, f"Error: {e}"


# Load example
@callback(
    Output("input-F", "value"),
    Output("input-T", "value"),
    Output("table-calls", "data"),
    Output("table-puts", "data"),
    Input("btn-example", "n_clicks"),
    prevent_initial_call=True,
)
def load_example(n):
    return (
        EXAMPLE_DATA["F"],
        EXAMPLE_DATA["T"],
        [{"K": r["K"], "price": r["price"]} for r in EXAMPLE_DATA["calls"]],
        [{"K": r["K"], "price": r["price"]} for r in EXAMPLE_DATA["puts"]],
    )


# ---- Save to CSV ----
@callback(
    Output("download-csv", "data"),
    Input("btn-save-csv", "n_clicks"),
    State("input-F", "value"),
    State("input-T", "value"),
    State("table-calls", "data"),
    State("table-puts", "data"),
    State("input-filename", "value"),
    prevent_initial_call=True,
)
def save_csv(n, F, T, call_rows, put_rows, filename):
    lines = [f"F,{F}", f"T,{T}", "type,K,price,weight"]
    for r in call_rows:
        try:
            lines.append(f"call,{float(r['K'])},{float(r['price'])},1.0")
        except (ValueError, TypeError, KeyError):
            pass
    for r in put_rows:
        try:
            lines.append(f"put,{float(r['K'])},{float(r['price'])},1.0")
        except (ValueError, TypeError, KeyError):
            pass
    content = "\n".join(lines) + "\n"
    fname = (filename or "toy_option_data").strip()
    if not fname.endswith(".csv"):
        fname += ".csv"
    return dict(content=content, filename=fname)


def _read_params_from_inputs(param_values):
    """Convert list of input values to numpy array."""
    return np.array([float(v) for v in param_values])


def _build_quote_set(F, T, call_rows, put_rows):
    """Build CanonicalQuoteSet from UI state."""
    calls = []
    for r in call_rows:
        try:
            calls.append((float(r["K"]), float(r["price"])))
        except (ValueError, TypeError, KeyError):
            pass
    puts = []
    for r in put_rows:
        try:
            puts.append((float(r["K"]), float(r["price"])))
        except (ValueError, TypeError, KeyError):
            pass
    if not calls and not puts:
        return None
    return CanonicalQuoteSet.from_manual(float(F), float(T), calls, puts)


# ---- Calibrate button ----
@callback(
    Output("store-params-json", "data"),
    Output("metrics-text", "children"),
    Input("btn-calibrate", "n_clicks"),
    State("input-F", "value"),
    State("input-T", "value"),
    State("table-calls", "data"),
    State("table-puts", "data"),
    State({"type": "param-input", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def calibrate(n, F, T, call_rows, put_rows, param_values):
    qs = _build_quote_set(F, T, call_rows, put_rows)
    if qs is None:
        return no_update, "No valid data points"
    svc.set_data(qs)
    svc.set_params(_read_params_from_inputs(param_values))
    try:
        result = svc.calibrate()
    except Exception as e:
        return no_update, f"Calibration failed: {e}"
    params_list = [round(float(v), 6) for v in result.params]
    m = result.metrics
    msg = (
        f"Status: {'OK' if result.success else 'FAILED'}\n"
        f"RMSE: {m['rmse']:.4f}  MAE: {m['mae']:.4f}  Max: {m['max_error']:.4f}\n"
        f"Params: {dict(zip(svc.model.param_names, params_list))}"
    )
    return json.dumps(params_list), msg


# ---- Reset button ----
@callback(
    Output("store-params-json", "data", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_params(n):
    svc.params = svc.model.default_params()
    return json.dumps([round(float(v), 6) for v in svc.params])


# ---- Add trade row ----
@callback(
    Output("table-trades", "data", allow_duplicate=True),
    Input("btn-add-trade", "n_clicks"),
    State("table-trades", "data"),
    prevent_initial_call=True,
)
def add_trade_row(n, rows):
    rows.append({"side": "call", "strike": 100, "volume": 100, "direction": "buy"})
    return rows


# ---- Apply trades ----
@callback(
    Output("store-params-json", "data", allow_duplicate=True),
    Output("store-base-params-json", "data"),
    Output("emulator-status", "children"),
    Input("btn-apply-trades", "n_clicks"),
    State("input-F", "value"),
    State("input-T", "value"),
    State("table-calls", "data"),
    State("table-puts", "data"),
    State("table-trades", "data"),
    State("dropdown-reaction-mode", "value"),
    State("input-shift-atm", "value"),
    State("input-shift-wing", "value"),
    State("input-volume-ref", "value"),
    State({"type": "param-input", "index": dash.ALL}, "value"),
    prevent_initial_call=True,
)
def apply_trades(n, F, T, call_rows, put_rows, trade_rows,
                 combo_model, shift_atm, shift_wing, volume_ref, param_values):
    qs = _build_quote_set(F, T, call_rows, put_rows)
    if qs is None:
        return no_update, no_update, "No valid data points"

    # Parse trades
    trades = []
    for r in trade_rows:
        try:
            t = Trade(
                side=r["side"],
                strike=float(r["strike"]),
                volume=float(r["volume"]),
                direction=r["direction"],
            )
            trades.append(t)
        except (ValueError, TypeError, KeyError):
            pass

    if not trades:
        return no_update, no_update, "No valid trades to apply"

    # Setup
    svc.set_data(qs)
    svc.set_params(_read_params_from_inputs(param_values))

    config = ReactionConfig(
        shift_atm=float(shift_atm if shift_atm is not None else 1.0),
        shift_wing=float(shift_wing or 1.0),
        volume_ref=float(volume_ref or 100.0),
        combination_model=combo_model,
    )
    svc.init_emulator(config)
    base_json = json.dumps([round(float(v), 6) for v in svc.base_params])

    # Apply each trade cumulatively
    log_lines = []
    for i, t in enumerate(trades, 1):
        try:
            svc.apply_trade(t)
            log_lines.append(f"#{i} {t.direction} {t.side} K={t.strike} v={t.volume} -> OK")
        except Exception as e:
            log_lines.append(f"#{i} {t.direction} {t.side} K={t.strike} v={t.volume} -> ERR: {e}")

    params_json = json.dumps([round(float(v), 6) for v in svc.params])
    status = f"Combination: {combo_model}\n" + "\n".join(log_lines)
    return params_json, base_json, status


# ---- Reset emulator ----
@callback(
    Output("store-params-json", "data", allow_duplicate=True),
    Output("store-base-params-json", "data", allow_duplicate=True),
    Output("emulator-status", "children", allow_duplicate=True),
    Input("btn-reset-emulator", "n_clicks"),
    prevent_initial_call=True,
)
def reset_emulator(n):
    svc.reset_emulator()
    params_json = json.dumps([round(float(v), 6) for v in svc.params])
    return params_json, None, "Emulator reset to base params"


# ---- Sync store -> sliders/inputs ----
@callback(
    Output({"type": "param-input", "index": dash.ALL}, "value"),
    Output({"type": "param-slider", "index": dash.ALL}, "value"),
    Input("store-params-json", "data"),
    prevent_initial_call=True,
)
def sync_params_to_controls(params_json):
    if params_json is None:
        return no_update, no_update
    vals = json.loads(params_json)
    rounded = [round(v, 4) for v in vals]
    return rounded, vals


# ---- Main plot update (on any input/slider change) ----
@callback(
    Output("graph-prices", "figure"),
    Output("graph-residuals", "figure"),
    Output("graph-iv", "figure"),
    Output("noarb-text", "children"),
    Input({"type": "param-input", "index": dash.ALL}, "value"),
    Input({"type": "param-slider", "index": dash.ALL}, "value"),
    Input("input-F", "value"),
    Input("input-T", "value"),
    Input("table-calls", "data"),
    Input("table-puts", "data"),
    State("store-base-params-json", "data"),
)
def update_plots(param_inputs, param_sliders, F, T, call_rows, put_rows, base_params_json):
    # Determine which triggered — prefer sliders for smooth interaction
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger_src = "input"
    else:
        trigger_src = "slider" if "param-slider" in ctx.triggered[0]["prop_id"] else "input"

    param_values = param_sliders if trigger_src == "slider" else param_inputs

    try:
        params = np.array([float(v) for v in param_values])
    except (TypeError, ValueError):
        return no_update, no_update, no_update, no_update

    qs = _build_quote_set(F, T, call_rows, put_rows)
    if qs is None:
        return (_empty_fig("Prices"), _empty_fig("Residuals"),
                _empty_fig("Implied Volatility"), "No data")

    svc.set_data(qs)
    svc.set_params(params)
    payload = svc.evaluate_for_plots()

    K_grid = payload["K_grid"]
    fwd = payload["F"]
    T_val = float(T)
    mc = payload["market_calls"]
    mp = payload["market_puts"]

    # ---- Combined price chart (calls + puts) ----
    fig_prices = go.Figure()
    # Model curves
    fig_prices.add_trace(go.Scatter(
        x=K_grid, y=payload["call_curve"], mode="lines",
        name="Call (model)", line={"color": "#1f77b4", "width": 2},
    ))
    fig_prices.add_trace(go.Scatter(
        x=K_grid, y=payload["put_curve"], mode="lines",
        name="Put (model)", line={"color": "#ff7f0e", "width": 2},
    ))
    # Market points
    if len(mc["K"]):
        fig_prices.add_trace(go.Scatter(
            x=mc["K"], y=mc["P"], mode="markers",
            name="Call (market)",
            marker={"color": "#1f77b4", "size": 10, "symbol": "circle",
                    "line": {"color": "white", "width": 1.5}},
        ))
    if len(mp["K"]):
        fig_prices.add_trace(go.Scatter(
            x=mp["K"], y=mp["P"], mode="markers",
            name="Put (market)",
            marker={"color": "#ff7f0e", "size": 10, "symbol": "diamond",
                    "line": {"color": "white", "width": 1.5}},
        ))
    # Model prices at user strikes (to see how model fits the given points)
    if len(mc["K"]):
        model_call_at_K = svc.model.vectorized_price("call", mc["K"], fwd, T_val, svc.params)
        fig_prices.add_trace(go.Scatter(
            x=mc["K"], y=model_call_at_K, mode="markers",
            name="Call (model@strikes)",
            marker={"color": "#1f77b4", "size": 8, "symbol": "x",
                    "line": {"width": 2}},
        ))
    if len(mp["K"]):
        model_put_at_K = svc.model.vectorized_price("put", mp["K"], fwd, T_val, svc.params)
        fig_prices.add_trace(go.Scatter(
            x=mp["K"], y=model_put_at_K, mode="markers",
            name="Put (model@strikes)",
            marker={"color": "#ff7f0e", "size": 8, "symbol": "x",
                    "line": {"width": 2}},
        ))
    # Base curves (pre-trade) as dashed lines
    if base_params_json is not None:
        base_curves = svc.evaluate_base_curves()
        if base_curves is not None:
            fig_prices.add_trace(go.Scatter(
                x=base_curves["K_grid"], y=base_curves["call_curve"], mode="lines",
                name="Call (base)", line={"color": "#1f77b4", "width": 1, "dash": "dash"},
                opacity=0.5,
            ))
            fig_prices.add_trace(go.Scatter(
                x=base_curves["K_grid"], y=base_curves["put_curve"], mode="lines",
                name="Put (base)", line={"color": "#ff7f0e", "width": 1, "dash": "dash"},
                opacity=0.5,
            ))

    fig_prices.add_vline(x=fwd, line_dash="dash", line_color="gray",
                         annotation_text="F")
    fig_prices.update_layout(
        title="Option prices (Call & Put)",
        xaxis_title="Strike",
        yaxis_title="Price",
        margin=dict(t=40, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    )

    # ---- Residuals chart ----
    fig_res = go.Figure()
    res_c = payload["res_call"]
    res_p = payload["res_put"]
    if len(res_c):
        fig_res.add_trace(go.Bar(
            x=[f"C {k:.0f}" for k in mc["K"]], y=res_c, name="Call err",
            marker_color="#1f77b4",
        ))
    if len(res_p):
        fig_res.add_trace(go.Bar(
            x=[f"P {k:.0f}" for k in mp["K"]], y=res_p, name="Put err",
            marker_color="#ff7f0e",
        ))
    fig_res.update_layout(title="Residuals (model - market)",
                          yaxis_title="Error", margin=dict(t=40, b=30),
                          barmode="group")

    # ---- Implied Volatility chart ----
    fig_iv = go.Figure()
    if T_val > 0:
        # Model IV curve (from model prices on K_grid)
        # Use only OTM options for cleaner IV: calls for K > F, puts for K < F
        call_mask = K_grid >= fwd
        put_mask = K_grid <= fwd
        if np.any(call_mask):
            K_call_grid = K_grid[call_mask]
            iv_call_curve = implied_vols_from_prices(
                payload["call_curve"][call_mask], K_call_grid, fwd, T_val, "call"
            )
            valid = ~np.isnan(iv_call_curve)
            if np.any(valid):
                fig_iv.add_trace(go.Scatter(
                    x=K_call_grid[valid], y=iv_call_curve[valid] * 100, mode="lines",
                    name="Call IV (model)", line={"color": "#1f77b4", "width": 2},
                ))
        if np.any(put_mask):
            K_put_grid = K_grid[put_mask]
            iv_put_curve = implied_vols_from_prices(
                payload["put_curve"][put_mask], K_put_grid, fwd, T_val, "put"
            )
            valid = ~np.isnan(iv_put_curve)
            if np.any(valid):
                fig_iv.add_trace(go.Scatter(
                    x=K_put_grid[valid], y=iv_put_curve[valid] * 100, mode="lines",
                    name="Put IV (model)", line={"color": "#ff7f0e", "width": 2},
                ))
        # Market IV at user strikes
        if len(mc["K"]):
            iv_call_mkt = implied_vols_from_prices(mc["P"], mc["K"], fwd, T_val, "call")
            valid = ~np.isnan(iv_call_mkt)
            if np.any(valid):
                fig_iv.add_trace(go.Scatter(
                    x=mc["K"][valid], y=iv_call_mkt[valid] * 100, mode="markers",
                    name="Call IV (market)",
                    marker={"color": "#1f77b4", "size": 10, "symbol": "circle",
                            "line": {"color": "white", "width": 1.5}},
                ))
        if len(mp["K"]):
            iv_put_mkt = implied_vols_from_prices(mp["P"], mp["K"], fwd, T_val, "put")
            valid = ~np.isnan(iv_put_mkt)
            if np.any(valid):
                fig_iv.add_trace(go.Scatter(
                    x=mp["K"][valid], y=iv_put_mkt[valid] * 100, mode="markers",
                    name="Put IV (market)",
                    marker={"color": "#ff7f0e", "size": 10, "symbol": "diamond",
                            "line": {"color": "white", "width": 1.5}},
                ))
        # Model IV at user strikes
        if len(mc["K"]):
            model_call_prices = svc.model.vectorized_price("call", mc["K"], fwd, T_val, svc.params)
            iv_call_model = implied_vols_from_prices(model_call_prices, mc["K"], fwd, T_val, "call")
            valid = ~np.isnan(iv_call_model)
            if np.any(valid):
                fig_iv.add_trace(go.Scatter(
                    x=mc["K"][valid], y=iv_call_model[valid] * 100, mode="markers",
                    name="Call IV (model@strikes)",
                    marker={"color": "#1f77b4", "size": 8, "symbol": "x",
                            "line": {"width": 2}},
                ))
        if len(mp["K"]):
            model_put_prices = svc.model.vectorized_price("put", mp["K"], fwd, T_val, svc.params)
            iv_put_model = implied_vols_from_prices(model_put_prices, mp["K"], fwd, T_val, "put")
            valid = ~np.isnan(iv_put_model)
            if np.any(valid):
                fig_iv.add_trace(go.Scatter(
                    x=mp["K"][valid], y=iv_put_model[valid] * 100, mode="markers",
                    name="Put IV (model@strikes)",
                    marker={"color": "#ff7f0e", "size": 8, "symbol": "x",
                            "line": {"width": 2}},
                ))
        # Base IV curves (pre-trade) as dashed lines
        if base_params_json is not None:
            base_curves = svc.evaluate_base_curves()
            if base_curves is not None:
                bK = base_curves["K_grid"]
                bc_mask = bK >= fwd
                bp_mask = bK <= fwd
                if np.any(bc_mask):
                    bK_call = bK[bc_mask]
                    iv_base_call = implied_vols_from_prices(
                        base_curves["call_curve"][bc_mask], bK_call, fwd, T_val, "call"
                    )
                    valid = ~np.isnan(iv_base_call)
                    if np.any(valid):
                        fig_iv.add_trace(go.Scatter(
                            x=bK_call[valid], y=iv_base_call[valid] * 100, mode="lines",
                            name="Call IV (base)",
                            line={"color": "#1f77b4", "width": 1, "dash": "dash"},
                            opacity=0.5,
                        ))
                if np.any(bp_mask):
                    bK_put = bK[bp_mask]
                    iv_base_put = implied_vols_from_prices(
                        base_curves["put_curve"][bp_mask], bK_put, fwd, T_val, "put"
                    )
                    valid = ~np.isnan(iv_base_put)
                    if np.any(valid):
                        fig_iv.add_trace(go.Scatter(
                            x=bK_put[valid], y=iv_base_put[valid] * 100, mode="lines",
                            name="Put IV (base)",
                            line={"color": "#ff7f0e", "width": 1, "dash": "dash"},
                            opacity=0.5,
                        ))

    fig_iv.add_vline(x=fwd, line_dash="dash", line_color="gray",
                     annotation_text="F")
    fig_iv.update_layout(
        title="Implied Volatility (Black-76)",
        xaxis_title="Strike",
        yaxis_title="IV (%)",
        margin=dict(t=40, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    )

    # ---- No-arb text ----
    noarb_lines = []
    for chk in payload.get("noarb", []):
        icon = "OK" if chk["ok"] else "FAIL"
        noarb_lines.append(f"[{icon}] {chk['name']}: {chk['detail']}")
    m = payload.get("metrics", {})
    if m:
        noarb_lines.append(
            f"\nRMSE={m.get('rmse', 0):.4f}  MAE={m.get('mae', 0):.4f}  "
            f"MaxErr={m.get('max_error', 0):.4f}"
        )
    noarb_text = "\n".join(noarb_lines) if noarb_lines else ""

    return fig_prices, fig_res, fig_iv, noarb_text


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, margin=dict(t=40, b=30))
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
