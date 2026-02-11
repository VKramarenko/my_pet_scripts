"""
Интерактивный дашборд для демонстрации пакета ts_analysis.

Запуск:
    python dashboard.py

Откроется на http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output

from ts_analysis import (
    trend_diagnostics,
    mean_reversion_summary,
    regime_diagnostics,
    johansen_test,
)
from examples import (
    generate_random_walk,
    generate_rw_with_drift,
    generate_deterministic_trend_rw,
    generate_quadratic_trend_rw,
    generate_ornstein_uhlenbeck,
    generate_ar1,
    generate_cointegrated_pair,
    generate_cointegrated_triple,
)

# ── Стили ────────────────────────────────────────────────────────────

COLORS = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#c9d1d9",
    "muted": "#8b949e",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "yellow": "#d29922",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "12px",
}

METRIC_STYLE = {
    "display": "inline-block",
    "backgroundColor": COLORS["bg"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "6px",
    "padding": "8px 14px",
    "margin": "4px",
    "textAlign": "center",
}

PLOT_TEMPLATE = "plotly_dark"


# ── Генерация данных ─────────────────────────────────────────────────

def build_all_data(n: int, seed: int):
    """Генерирует все синтетические ряды с заданным seed."""
    rng = np.random.default_rng(seed)

    trend_series = {
        "RW с дрифтом (μ=0.05)": generate_rw_with_drift(n, rng, drift=0.05),
        "Детерм. тренд + RW": generate_deterministic_trend_rw(n, rng, slope=0.1),
        "Квадратич. тренд + RW": generate_quadratic_trend_rw(n, rng),
        "Чистый RW": generate_random_walk(n, rng),
    }

    mr_series = {
        "Ornstein-Uhlenbeck (θ=0.15)": generate_ornstein_uhlenbeck(n, rng, theta=0.15),
        "AR(1) φ=0.5": generate_ar1(n, rng, phi=0.5),
        "AR(1) φ=0.9": generate_ar1(n, rng, phi=0.9),
        "Чистый RW": generate_random_walk(n, rng),
    }

    rng_coint = np.random.default_rng(seed + 1)
    cx, cy = generate_cointegrated_pair(n, rng_coint, beta=1.5, spread_std=0.5)
    cx1, cx2, cx3 = generate_cointegrated_triple(n, rng_coint)
    rng_indep = np.random.default_rng(seed + 2)
    indep_a = generate_random_walk(n, rng_indep)
    indep_b = generate_random_walk(n, rng_indep)

    return trend_series, mr_series, (cx, cy), (cx1, cx2, cx3), (indep_a, indep_b)


# ── Вспомогательные функции для карточек метрик ──────────────────────

def metric_badge(label: str, value: str, color: str = COLORS["accent"]):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": COLORS["muted"]}),
        html.Div(value, style={"fontSize": "15px", "fontWeight": "bold", "color": color}),
    ], style=METRIC_STYLE)


def assessment_color(text: str) -> str:
    t = text.lower()
    if "high" in t or "full" in t:
        return COLORS["green"]
    if "moderate" in t or "cointegration" in t:
        return COLORS["yellow"]
    if "unlikely" in t or "no " in t:
        return COLORS["red"]
    return COLORS["text"]


# ── Построение графиков и карточек ───────────────────────────────────

def build_trend_tab(trend_series: dict) -> list:
    """Вкладка Тренд: графики + результаты тестов."""
    children = []

    colors = ["#58a6ff", "#f0883e", "#a371f7", "#8b949e"]
    fig = go.Figure()
    for (name, y), c in zip(trend_series.items(), colors):
        fig.add_trace(go.Scatter(y=y, name=name, line=dict(width=1.5, color=c)))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Ряды с трендом",
        height=380,
        margin=dict(l=50, r=20, t=40, b=30),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.15),
    )
    children.append(html.Div(dcc.Graph(figure=fig), style=CARD_STYLE))

    for name, y in trend_series.items():
        r = trend_diagnostics(y)
        interp = r["interpretation_adf_kpss"]
        children.append(html.Div([
            html.H4(name, style={"color": COLORS["accent"], "marginBottom": "8px"}),
            html.Div([
                metric_badge("ADF p-val", f"{r['adf_ct']['pvalue']:.4f}"),
                metric_badge("KPSS p-val", f"{r['kpss_ct']['pvalue']:.4f}"),
                metric_badge("Mann-Kendall", r["mann_kendall"]["trend"]),
                metric_badge("Spearman ρ", f"{r['spearman']['rho']:.3f}"),
                metric_badge("Cox-Stuart", r["cox_stuart"]["trend"]),
                metric_badge("Slope (HAC)", f"{r['linear_trend_hac']['slope']:.4f}"),
            ]),
            html.P(f"Интерпретация: {interp}",
                   style={"marginTop": "8px", "color": COLORS["muted"], "fontSize": "13px"}),
        ], style=CARD_STYLE))

    return children


def build_mr_tab(mr_series: dict) -> list:
    """Вкладка Mean Reversion: графики + score-карточки."""
    children = []

    colors = ["#3fb950", "#58a6ff", "#d29922", "#8b949e"]
    fig = go.Figure()
    for (name, y), c in zip(mr_series.items(), colors):
        fig.add_trace(go.Scatter(y=y, name=name, line=dict(width=1.5, color=c)))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Ряды с возвращением к среднему",
        height=380,
        margin=dict(l=50, r=20, t=40, b=30),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.15),
    )
    children.append(html.Div(dcc.Graph(figure=fig), style=CARD_STYLE))

    for name, y in mr_series.items():
        r = mean_reversion_summary(y)
        score_pct = f"{r['final_score']:.0%}"
        ac = assessment_color(r["assessment"])

        score_details = r["scores"]
        children.append(html.Div([
            html.Div([
                html.H4(name, style={"color": COLORS["accent"], "marginBottom": "4px",
                                     "display": "inline-block"}),
                html.Span(f"  Score: {score_pct}",
                          style={"fontSize": "18px", "fontWeight": "bold",
                                 "color": ac, "marginLeft": "12px"}),
            ]),
            html.Div([
                metric_badge("ADF", "✓" if score_details["adf"] else "✗",
                             COLORS["green"] if score_details["adf"] else COLORS["red"]),
                metric_badge("KPSS", "✓" if score_details["kpss"] else "✗",
                             COLORS["green"] if score_details["kpss"] else COLORS["red"]),
                metric_badge("Hurst", f"{r['hurst']['hurst_value']:.3f}",
                             COLORS["green"] if score_details["hurst"] else COLORS["red"]),
                metric_badge("ACF", "✓" if score_details["acf"] else "✗",
                             COLORS["green"] if score_details["acf"] else COLORS["red"]),
                metric_badge("Half-life", f"{r['half_life']['half_life']:.1f}",
                             COLORS["green"] if score_details["half_life"] else COLORS["red"]),
            ]),
            html.P(f"{r['assessment']} — {r['recommendation']}",
                   style={"marginTop": "8px", "color": COLORS["muted"], "fontSize": "13px"}),
        ], style=CARD_STYLE))

    return children


def build_regime_tab(trend_series: dict, mr_series: dict) -> list:
    """Вкладка Режимная диагностика: сравнение тренд vs MR."""
    children = []

    # Берём по одному характерному ряду из каждой категории + пограничный
    rng = np.random.default_rng(42)
    n = 500
    regime_series = {
        "RW с дрифтом (тренд)": generate_rw_with_drift(n, rng, drift=0.05),
        "OU (mean reversion)": generate_ornstein_uhlenbeck(n, rng, theta=0.15),
        "AR(1) φ=0.9 (пограничный)": generate_ar1(n, rng, phi=0.9),
    }

    colors = ["#f85149", "#3fb950", "#d29922"]
    fig = go.Figure()
    for (name, y), c in zip(regime_series.items(), colors):
        fig.add_trace(go.Scatter(y=y, name=name, line=dict(width=1.5, color=c)))
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Режимная диагностика: тренд vs mean reversion",
        height=380,
        margin=dict(l=50, r=20, t=40, b=30),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.15),
    )
    children.append(html.Div(dcc.Graph(figure=fig), style=CARD_STYLE))

    for name, y in regime_series.items():
        r = regime_diagnostics(y)
        vrs = r["variance_ratio"]

        # Маленький график VR по горизонтам
        vr_fig = go.Figure()
        ks = [v["k"] for v in vrs]
        vr_vals = [v["vr"] for v in vrs]
        vr_fig.add_trace(go.Bar(x=[f"k={k}" for k in ks], y=vr_vals,
                                marker_color=[COLORS["green"] if v < 1 else COLORS["red"]
                                              for v in vr_vals]))
        vr_fig.add_hline(y=1, line_dash="dash", line_color=COLORS["muted"])
        vr_fig.update_layout(
            template=PLOT_TEMPLATE,
            height=180, margin=dict(l=40, r=10, t=10, b=30),
            paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["bg"],
            yaxis_title="VR",
        )

        children.append(html.Div([
            html.H4(name, style={"color": COLORS["accent"], "marginBottom": "8px"}),
            html.Div([
                metric_badge("AR(1) φ", f"{r['ar1']['phi']:.4f}"),
                metric_badge("Hurst H", f"{r['hurst_rs']['H']:.3f}"),
                metric_badge("ZA break", str(r["zivot_andrews_ct"]["break_index"])),
                metric_badge("ZA p-val", f"{r['zivot_andrews_ct']['pvalue']:.4f}"),
            ]),
            html.Div([
                html.Div("Variance Ratio по горизонтам:",
                         style={"color": COLORS["muted"], "fontSize": "12px",
                                "marginTop": "8px"}),
                dcc.Graph(figure=vr_fig, config={"displayModeBar": False}),
            ]),
            html.Div([
                html.Span("Hints: ", style={"color": COLORS["muted"], "fontSize": "12px"}),
                html.Span(" | ".join(r["hints"]),
                          style={"color": COLORS["text"], "fontSize": "12px"}),
            ], style={"marginTop": "4px"}),
        ], style=CARD_STYLE))

    return children


def build_coint_tab(pair, triple, indep):
    """Вкладка Коинтеграция: графики пар/троек + результаты Йохансена."""
    children = []
    cx, cy = pair
    cx1, cx2, cx3 = triple
    indep_a, indep_b = indep

    # ── Коинтегрированная пара ──
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.6, 0.4],
                         subplot_titles=["Коинтегрированная пара (Y ≈ 1.5·X)", "Спред (Y - 1.5·X)"])
    fig1.add_trace(go.Scatter(y=cx, name="X", line=dict(width=1.5, color="#58a6ff")), row=1, col=1)
    fig1.add_trace(go.Scatter(y=cy, name="Y", line=dict(width=1.5, color="#f0883e")), row=1, col=1)
    spread = cy - 1.5 * cx
    fig1.add_trace(go.Scatter(y=spread, name="Spread", line=dict(width=1, color="#a371f7")),
                   row=2, col=1)
    fig1.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], row=2, col=1)
    fig1.update_layout(
        template=PLOT_TEMPLATE, height=420,
        margin=dict(l=50, r=20, t=40, b=30),
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["bg"],
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )

    pair_df = pd.DataFrame({"X": cx, "Y": cy})
    r1 = johansen_test(pair_df, det_order=0, k_ar_diff=1)
    ac1 = assessment_color(r1["conclusion"]["assessment"])

    children.append(html.Div([
        dcc.Graph(figure=fig1),
        html.Div([
            metric_badge("Коинтегрирован", "Да" if r1["conclusion"]["is_cointegrated"] else "Нет",
                         COLORS["green"] if r1["conclusion"]["is_cointegrated"] else COLORS["red"]),
            metric_badge("Векторов", str(r1["conclusion"]["recommended"])),
            metric_badge("Assessment", r1["conclusion"]["assessment"], ac1),
        ]),
        html.P(f"Eigenvalues: {np.round(r1['eigenvalues'], 4)}",
               style={"color": COLORS["muted"], "fontSize": "12px", "marginTop": "8px"}),
    ], style=CARD_STYLE))

    # ── Тройка ──
    fig2 = go.Figure()
    for name, y, c in [("X1", cx1, "#58a6ff"), ("X2", cx2, "#3fb950"), ("X3", cx3, "#f0883e")]:
        fig2.add_trace(go.Scatter(y=y, name=name, line=dict(width=1.5, color=c)))
    fig2.update_layout(
        template=PLOT_TEMPLATE, title="Тройка с общим фактором",
        height=320, margin=dict(l=50, r=20, t=40, b=30),
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.15),
    )

    triple_df = pd.DataFrame({"X1": cx1, "X2": cx2, "X3": cx3})
    r2 = johansen_test(triple_df, det_order=0, k_ar_diff=1)
    ac2 = assessment_color(r2["conclusion"]["assessment"])

    children.append(html.Div([
        dcc.Graph(figure=fig2),
        html.Div([
            metric_badge("Коинтегрирован", "Да" if r2["conclusion"]["is_cointegrated"] else "Нет",
                         COLORS["green"] if r2["conclusion"]["is_cointegrated"] else COLORS["red"]),
            metric_badge("Векторов", str(r2["conclusion"]["recommended"])),
            metric_badge("Assessment", r2["conclusion"]["assessment"], ac2),
        ]),
        html.P(f"Eigenvalues: {np.round(r2['eigenvalues'], 4)}",
               style={"color": COLORS["muted"], "fontSize": "12px", "marginTop": "8px"}),
    ], style=CARD_STYLE))

    # ── Независимые RW ──
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=indep_a, name="A", line=dict(width=1.5, color="#58a6ff")))
    fig3.add_trace(go.Scatter(y=indep_b, name="B", line=dict(width=1.5, color="#f85149")))
    fig3.update_layout(
        template=PLOT_TEMPLATE, title="Независимые random walks (контроль)",
        height=320, margin=dict(l=50, r=20, t=40, b=30),
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["bg"],
        legend=dict(orientation="h", y=-0.15),
    )

    indep_df = pd.DataFrame({"A": indep_a, "B": indep_b})
    r3 = johansen_test(indep_df, det_order=0, k_ar_diff=1)
    ac3 = assessment_color(r3["conclusion"]["assessment"])

    children.append(html.Div([
        dcc.Graph(figure=fig3),
        html.Div([
            metric_badge("Коинтегрирован", "Да" if r3["conclusion"]["is_cointegrated"] else "Нет",
                         COLORS["green"] if r3["conclusion"]["is_cointegrated"] else COLORS["red"]),
            metric_badge("Assessment", r3["conclusion"]["assessment"], ac3),
        ]),
    ], style=CARD_STYLE))

    return children


# ── Приложение Dash ──────────────────────────────────────────────────

app = Dash(__name__)
app.title = "ts_analysis — демонстрация"

app.layout = html.Div([
    html.H1("ts_analysis", style={"color": COLORS["accent"], "marginBottom": "2px"}),
    html.P("Диагностика временных рядов: тренд, mean reversion, коинтеграция",
           style={"color": COLORS["muted"], "marginTop": "0"}),

    html.Div([
        html.Div([
            html.Label("Длина ряда (n):", style={"color": COLORS["text"], "marginRight": "8px"}),
            dcc.Input(id="input-n", type="number", value=500, min=100, max=5000, step=100,
                      style={"width": "80px", "backgroundColor": COLORS["bg"],
                             "color": COLORS["text"], "border": f"1px solid {COLORS['border']}",
                             "borderRadius": "4px", "padding": "4px 8px"}),
        ], style={"display": "inline-block", "marginRight": "24px"}),
        html.Div([
            html.Label("Seed:", style={"color": COLORS["text"], "marginRight": "8px"}),
            dcc.Input(id="input-seed", type="number", value=42, min=0, max=9999, step=1,
                      style={"width": "80px", "backgroundColor": COLORS["bg"],
                             "color": COLORS["text"], "border": f"1px solid {COLORS['border']}",
                             "borderRadius": "4px", "padding": "4px 8px"}),
        ], style={"display": "inline-block"}),
    ], style={"marginBottom": "16px"}),

    dcc.Tabs(id="tabs", value="trend", children=[
        dcc.Tab(label="Тренд", value="trend",
                style={"backgroundColor": COLORS["bg"], "color": COLORS["muted"],
                        "border": f"1px solid {COLORS['border']}"},
                selected_style={"backgroundColor": COLORS["card"], "color": COLORS["accent"],
                                "borderTop": f"2px solid {COLORS['accent']}"}),
        dcc.Tab(label="Mean Reversion", value="mr",
                style={"backgroundColor": COLORS["bg"], "color": COLORS["muted"],
                        "border": f"1px solid {COLORS['border']}"},
                selected_style={"backgroundColor": COLORS["card"], "color": COLORS["accent"],
                                "borderTop": f"2px solid {COLORS['accent']}"}),
        dcc.Tab(label="Режимы", value="regime",
                style={"backgroundColor": COLORS["bg"], "color": COLORS["muted"],
                        "border": f"1px solid {COLORS['border']}"},
                selected_style={"backgroundColor": COLORS["card"], "color": COLORS["accent"],
                                "borderTop": f"2px solid {COLORS['accent']}"}),
        dcc.Tab(label="Коинтеграция", value="coint",
                style={"backgroundColor": COLORS["bg"], "color": COLORS["muted"],
                        "border": f"1px solid {COLORS['border']}"},
                selected_style={"backgroundColor": COLORS["card"], "color": COLORS["accent"],
                                "borderTop": f"2px solid {COLORS['accent']}"}),
    ]),
    html.Div(id="tab-content", style={"marginTop": "12px"}),

], style={
    "backgroundColor": COLORS["bg"],
    "color": COLORS["text"],
    "fontFamily": "'Segoe UI', system-ui, sans-serif",
    "padding": "24px",
    "minHeight": "100vh",
})


@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("input-n", "value"),
    Input("input-seed", "value"),
)
def update_tab(tab, n, seed):
    n = max(100, min(int(n or 500), 5000))
    seed = int(seed or 42)

    trend_series, mr_series, pair, triple, indep = build_all_data(n, seed)

    if tab == "trend":
        return build_trend_tab(trend_series)
    elif tab == "mr":
        return build_mr_tab(mr_series)
    elif tab == "regime":
        return build_regime_tab(trend_series, mr_series)
    elif tab == "coint":
        return build_coint_tab(pair, triple, indep)
    return []


if __name__ == "__main__":
    app.run(debug=True)
