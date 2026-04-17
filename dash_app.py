"""Dash-версия дашборда экономического симулятора РФ.

MVP-монолит UI поверх существующего Service/Engine слоя.
"""

from __future__ import annotations

from typing import Any

import dash
from dash import Dash, Input, Output, State, callback, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.interpreter import interpret_monte_carlo_bin, interpret_sobol_factor
from src.model import predict_scenario, predict_stock_scenario
from src.service import (
    get_moex_ticker_universe_service,
    get_processed_dataset_service,
    get_stock_history_service,
    run_monte_carlo_service,
    run_sobol_service,
)


REGIME_OPTIONS = [
    {"label": "Весь период (5 лет)", "value": "all"},
    {"label": "До февраля 2022", "value": "pre_2022"},
    {"label": "С февраля 2022", "value": "post_2022"},
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_dataset() -> pd.DataFrame:
    return get_processed_dataset_service().copy()


def _build_historical_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["imoex_close"], name="IMOEX", mode="lines"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["brent_usd"], name="Brent", mode="lines", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["usd_rub"], name="USD/RUB", mode="lines", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["key_rate"], name="Ключевая ставка", mode="lines", yaxis="y2"))
    fig.update_layout(
        template="plotly_white",
        title="Историческая динамика: IMOEX и макрофакторы",
        xaxis_title="Дата",
        yaxis=dict(title="IMOEX"),
        yaxis2=dict(title="Макрофакторы", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def _build_corr_fig(df: pd.DataFrame) -> go.Figure:
    cols = ["imoex_close", "brent_usd", "usd_rub", "key_rate", "inflation"]
    corr = df[cols].corr(numeric_only=True)
    label_map = {
        "imoex_close": "IMOEX",
        "brent_usd": "Brent",
        "usd_rub": "USD/RUB",
        "key_rate": "Ключевая ставка",
        "inflation": "Инфляция",
    }
    corr.index = [label_map[c] for c in corr.index]
    corr.columns = [label_map[c] for c in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
    fig.update_layout(template="plotly_white", title="Корреляционная матрица")
    return fig


def _ticker_options() -> list[dict[str, str]]:
    try:
        uni = get_moex_ticker_universe_service().copy()
        if "ticker" in uni.columns:
            uni["ticker"] = uni["ticker"].astype(str).str.upper().str.strip()
            uni = uni[uni["ticker"] != ""].drop_duplicates("ticker")
            out = [{"label": t, "value": t} for t in sorted(uni["ticker"].tolist())]
            if out:
                return out
    except Exception:
        pass
    reserve = ["LKOH", "ROSN", "TATN", "SBER", "GAZP", "NVTK", "GMKN", "MOEX", "VTBR", "T", "YDEX"]
    return [{"label": t, "value": t} for t in reserve]


def _asset_current(df: pd.DataFrame, asset_type: str, ticker: str) -> float:
    if asset_type == "stock":
        s = get_stock_history_service(ticker)
        col = f"{ticker.lower()}_close"
        if col in s.columns:
            return float(pd.to_numeric(s[col], errors="coerce").dropna().iloc[-1])
    return float(pd.to_numeric(df["imoex_close"], errors="coerce").dropna().iloc[-1])


def build_layout(df: pd.DataFrame) -> html.Div:
    default = df.iloc[-1]
    return html.Div(
        [
            html.H2("Экономический симулятор сценариев (Dash)"),
            dcc.Tabs(
                id="tabs",
                value="tab-history",
                children=[
                    dcc.Tab(
                        label="📊 Исторические данные",
                        value="tab-history",
                        children=[
                            dcc.Graph(id="hist-chart", figure=_build_historical_chart(df)),
                            dcc.Graph(id="corr-chart", figure=_build_corr_fig(df)),
                            html.Div("Пояснение: корреляция показывает линейную связь факторов от -1 до +1."),
                        ],
                    ),
                    dcc.Tab(
                        label="🎯 Сценарный анализ",
                        value="tab-scenario",
                        children=[
                            html.Div(
                                [
                                    html.Label("Целевой актив"),
                                    dcc.RadioItems(
                                        id="asset-type",
                                        options=[
                                            {"label": "IMOEX", "value": "imoex"},
                                            {"label": "Акция MOEX", "value": "stock"},
                                        ],
                                        value="imoex",
                                        inline=True,
                                    ),
                                    html.Label("Тикер (для акции)"),
                                    dcc.Dropdown(id="ticker", options=_ticker_options(), value="LKOH", clearable=False),
                                    html.Label("Режим модели"),
                                    dcc.Dropdown(id="regime", options=REGIME_OPTIONS, value="all", clearable=False),
                                    html.Label("Brent"),
                                    dcc.Slider(id="oil", min=30, max=130, step=5, value=float(default["brent_usd"])),
                                    html.Label("Ключевая ставка"),
                                    dcc.Slider(id="key-rate", min=5, max=30, step=0.5, value=float(default["key_rate"])),
                                    html.Label("USD/RUB"),
                                    dcc.Slider(id="usd-rub", min=60, max=150, step=1, value=float(default["usd_rub"])),
                                    html.Label("Инфляция"),
                                    dcc.Slider(id="inflation", min=2, max=20, step=0.5, value=float(default["inflation"])),
                                ]
                            ),
                            html.Br(),
                            html.Button("Рассчитать сценарий", id="run-scenario", n_clicks=0),
                            html.Div(id="scenario-output", style={"marginTop": "16px"}),
                        ],
                    ),
                    dcc.Tab(
                        label="🎲 Монте-Карло",
                        value="tab-mc",
                        children=[
                            html.Button("Запустить Monte Carlo", id="run-mc", n_clicks=0),
                            dcc.Graph(id="mc-fig"),
                            html.Div(
                                [
                                    html.Label("Диапазон (бин)"),
                                    dcc.Dropdown(id="mc-bin-select", clearable=False),
                                ]
                            ),
                            html.Div(id="mc-stats"),
                            dcc.Store(id="mc-store"),
                        ],
                    ),
                    dcc.Tab(
                        label="🌪️ Чувствительность",
                        value="tab-sobol",
                        children=[
                            html.Button("Рассчитать Sobol", id="run-sobol", n_clicks=0),
                            dcc.Graph(id="sobol-fig"),
                            html.Div([html.Label("Фактор"), dcc.Dropdown(id="sobol-factor-select", clearable=False)]),
                            html.Div(id="sobol-stats"),
                            dcc.Store(id="sobol-store"),
                        ],
                    ),
                ],
            ),
        ],
        style={"maxWidth": "1400px", "margin": "0 auto", "padding": "12px"},
    )


@callback(
    Output("scenario-output", "children"),
    Input("run-scenario", "n_clicks"),
    State("asset-type", "value"),
    State("ticker", "value"),
    State("regime", "value"),
    State("oil", "value"),
    State("key-rate", "value"),
    State("usd-rub", "value"),
    State("inflation", "value"),
    prevent_initial_call=True,
)
def run_scenario_cb(
    _n: int,
    asset_type: str,
    ticker: str,
    regime: str,
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
) -> html.Div:
    df = _load_dataset()
    if asset_type == "stock":
        pred = predict_stock_scenario(ticker, oil, key_rate, usd_rub, inflation, regime=regime)
        cur = _asset_current(df, "stock", ticker)
        label = f"Акция {ticker}"
    else:
        pred = predict_scenario(oil, key_rate, usd_rub, inflation, regime=regime)
        cur = _asset_current(df, "imoex", ticker)
        label = "IMOEX"
    delta = pred - cur
    pct = delta / cur if cur else 0.0
    return html.Div(
        [
            html.H4(f"Прогноз {label}: {pred:,.2f}"),
            html.Div(f"Текущее: {cur:,.2f} | Δ: {delta:,.2f} ({pct:+.2%})"),
        ]
    )


@callback(
    Output("mc-fig", "figure"),
    Output("mc-store", "data"),
    Output("mc-bin-select", "options"),
    Output("mc-bin-select", "value"),
    Input("run-mc", "n_clicks"),
    State("asset-type", "value"),
    State("ticker", "value"),
    State("regime", "value"),
    State("oil", "value"),
    State("key-rate", "value"),
    State("usd-rub", "value"),
    State("inflation", "value"),
    prevent_initial_call=True,
)
def run_mc_cb(
    _n: int,
    asset_type: str,
    ticker: str,
    regime: str,
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
):
    controls = {
        "oil": _safe_float(oil),
        "key_rate": _safe_float(key_rate),
        "usd_rub": _safe_float(usd_rub),
        "inflation": _safe_float(inflation),
        "uncertainty_scale": 1.0,
    }
    result = run_monte_carlo_service(
        controls=controls,
        mc_runs=5000,
        asset_type=asset_type,
        ticker=ticker if asset_type == "stock" else None,
        adjustments={},
        regime=regime,
        portfolio_tickers=[],
        portfolio_weights=None,
    )
    bins = result["hist_bins"].copy()
    bins["label"] = bins.apply(lambda r: f"{int(r['bin_index'])+1}: {r['left']:.1f} .. {r['right']:.1f}", axis=1)
    options = [{"label": l, "value": l} for l in bins["label"].tolist()]
    value = options[0]["value"] if options else None
    store = {
        "var_5": result["var_5"],
        "p50": result["p50"],
        "var_95": result["var_95"],
        "prob_drop_20": result["prob_drop_20"],
        "n_simulations": result["n_simulations"],
        "asset_label": result["asset_label"],
        "current_level": result["current_level"],
        "hist_bins": bins.to_dict("records"),
        "figure": result["figure"].to_dict(),
    }
    return result["figure"], store, options, value


@callback(
    Output("mc-bin-select", "value", allow_duplicate=True),
    Input("mc-fig", "clickData"),
    State("mc-store", "data"),
    prevent_initial_call=True,
)
def mc_click_sync(click_data: dict[str, Any] | None, mc_store: dict[str, Any] | None):
    if not click_data or not mc_store:
        return dash.no_update
    pts = click_data.get("points", [])
    if not pts:
        return dash.no_update
    point = pts[-1]
    custom = point.get("customdata")
    if isinstance(custom, list) and custom:
        idx = int(custom[0])
        bins = mc_store.get("hist_bins", [])
        if 0 <= idx < len(bins):
            return bins[idx]["label"]
    return dash.no_update


@callback(
    Output("mc-stats", "children"),
    Input("mc-bin-select", "value"),
    State("mc-store", "data"),
)
def mc_detail_cb(selected_label: str | None, mc_store: dict[str, Any] | None):
    if not mc_store:
        return html.Div("Запустите Monte Carlo")
    bins = mc_store.get("hist_bins", [])
    if not bins:
        return html.Div("Нет данных по бинам")
    row = next((b for b in bins if b.get("label") == selected_label), bins[0])
    interp = interpret_monte_carlo_bin(mc_store, row)
    return html.Div(
        [
            html.H4(f"P5={mc_store['var_5']:.1f} | P50={mc_store['p50']:.1f} | P95={mc_store['var_95']:.1f}"),
            html.Div(f"Вероятность падения >20%: {mc_store['prob_drop_20']:.2%}"),
            html.Br(),
            html.B(interp["headline"]),
            html.Div(interp["summary"]),
            html.Ul([html.Li(x) for x in interp["bullets"]]),
        ]
    )


@callback(
    Output("sobol-fig", "figure"),
    Output("sobol-store", "data"),
    Output("sobol-factor-select", "options"),
    Output("sobol-factor-select", "value"),
    Input("run-sobol", "n_clicks"),
    State("asset-type", "value"),
    State("ticker", "value"),
    State("regime", "value"),
    prevent_initial_call=True,
)
def run_sobol_cb(_n: int, asset_type: str, ticker: str, regime: str):
    result = run_sobol_service(
        n_samples=256,
        asset_type=asset_type,
        ticker=ticker if asset_type == "stock" else None,
        regime=regime,
        portfolio_tickers=[],
        portfolio_weights=None,
    )
    sobol_df = result["sobol_df"].copy()
    labels = sobol_df["factor_label"].astype(str).tolist()
    options = [{"label": x, "value": x} for x in labels]
    value = labels[0] if labels else None
    store = {
        "sobol_df": sobol_df.to_dict("records"),
        "top_factor": result["top_factor"],
        "top_s1": result["top_s1"],
        "y_variance": result["y_variance"],
        "problem": result["problem"],
    }
    return result["figure"], store, options, value


@callback(
    Output("sobol-factor-select", "value", allow_duplicate=True),
    Input("sobol-fig", "clickData"),
    State("sobol-store", "data"),
    prevent_initial_call=True,
)
def sobol_click_sync(click_data: dict[str, Any] | None, sobol_store: dict[str, Any] | None):
    if not click_data or not sobol_store:
        return dash.no_update
    points = click_data.get("points", [])
    if not points:
        return dash.no_update
    y = points[-1].get("y")
    if isinstance(y, str):
        return y
    return dash.no_update


@callback(
    Output("sobol-stats", "children"),
    Input("sobol-factor-select", "value"),
    State("sobol-store", "data"),
)
def sobol_detail_cb(selected_factor_label: str | None, sobol_store: dict[str, Any] | None):
    if not sobol_store:
        return html.Div("Запустите Sobol-анализ")
    sobol_rows = sobol_store.get("sobol_df", [])
    if not sobol_rows:
        return html.Div("Нет Sobol-данных")
    row = next((r for r in sobol_rows if r.get("factor_label") == selected_factor_label), sobol_rows[0])
    interp = interpret_sobol_factor(sobol_store, row)
    return html.Div(
        [
            html.H4(f"Топ-фактор: {sobol_store['top_factor']} (S1={sobol_store['top_s1']:.3f})"),
            html.B(interp["headline"]),
            html.Div(interp["summary"]),
            html.Ul([html.Li(x) for x in interp["bullets"]]),
        ]
    )


def create_app() -> Dash:
    df = _load_dataset()
    app = Dash(__name__)
    app.title = "Экономический симулятор РФ (Dash)"
    app.layout = build_layout(df)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8050, debug=True)

