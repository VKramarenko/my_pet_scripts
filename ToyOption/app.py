"""Dash UI for the ToyOption calibration tool.

Run:  python -m ToyOption.app
"""

from __future__ import annotations
import json
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, no_update
import plotly.graph_objects as go
import numpy as np

from .data import CanonicalQuoteSet
from .service import ModelService

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

                html.Button("Load Example", id="btn-example", n_clicks=0,
                            style={"width": "100%", "marginTop": "8px"}),
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
            ], style={"width": "26%", "padding": "12px", "verticalAlign": "top",
                       "display": "inline-block", "borderRight": "1px solid #ddd"}),

            # ---- RIGHT: Plots ----
            html.Div([
                dcc.Graph(id="graph-calls", style={"height": "300px"}),
                dcc.Graph(id="graph-puts", style={"height": "300px"}),
                dcc.Graph(id="graph-residuals", style={"height": "250px"}),
                html.Div(id="noarb-text",
                         style={"whiteSpace": "pre-wrap", "fontFamily": "monospace",
                                "fontSize": "12px", "padding": "8px",
                                "backgroundColor": "#f8f8f8", "borderRadius": "4px"}),
            ], style={"width": "50%", "padding": "12px", "verticalAlign": "top",
                       "display": "inline-block"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),

        # Hidden store for triggering updates
        dcc.Store(id="store-params-json"),
    ])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__)
app.layout = build_layout

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

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
    Output("graph-calls", "figure"),
    Output("graph-puts", "figure"),
    Output("graph-residuals", "figure"),
    Output("noarb-text", "children"),
    Input({"type": "param-input", "index": dash.ALL}, "value"),
    Input({"type": "param-slider", "index": dash.ALL}, "value"),
    Input("input-F", "value"),
    Input("input-T", "value"),
    Input("table-calls", "data"),
    Input("table-puts", "data"),
)
def update_plots(param_inputs, param_sliders, F, T, call_rows, put_rows):
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
        return _empty_fig("Calls"), _empty_fig("Puts"), _empty_fig("Residuals"), "No data"

    svc.set_data(qs)
    svc.set_params(params)
    payload = svc.evaluate_for_plots()

    K_grid = payload["K_grid"]
    fwd = payload["F"]

    # ---- Call chart ----
    fig_call = go.Figure()
    fig_call.add_trace(go.Scatter(
        x=K_grid, y=payload["call_curve"], mode="lines", name="Model",
        line={"color": "#1f77b4", "width": 2},
    ))
    mc = payload["market_calls"]
    if len(mc["K"]):
        fig_call.add_trace(go.Scatter(
            x=mc["K"], y=mc["P"], mode="markers", name="Market",
            marker={"color": "red", "size": 9, "symbol": "circle"},
        ))
    fig_call.add_vline(x=fwd, line_dash="dash", line_color="gray",
                       annotation_text="F")
    fig_call.update_layout(title="Call prices", xaxis_title="Strike",
                           yaxis_title="Price", margin=dict(t=40, b=30),
                           legend=dict(x=0.8, y=1))

    # ---- Put chart ----
    fig_put = go.Figure()
    fig_put.add_trace(go.Scatter(
        x=K_grid, y=payload["put_curve"], mode="lines", name="Model",
        line={"color": "#ff7f0e", "width": 2},
    ))
    mp = payload["market_puts"]
    if len(mp["K"]):
        fig_put.add_trace(go.Scatter(
            x=mp["K"], y=mp["P"], mode="markers", name="Market",
            marker={"color": "red", "size": 9, "symbol": "circle"},
        ))
    fig_put.add_vline(x=fwd, line_dash="dash", line_color="gray",
                      annotation_text="F")
    fig_put.update_layout(title="Put prices", xaxis_title="Strike",
                          yaxis_title="Price", margin=dict(t=40, b=30),
                          legend=dict(x=0.8, y=1))

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

    return fig_call, fig_put, fig_res, noarb_text


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, margin=dict(t=40, b=30))
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
