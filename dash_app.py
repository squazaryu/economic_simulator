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
        template="plotly_dark",
        title="Историческая динамика: IMOEX и макрофакторы",
        xaxis_title="Дата",
        yaxis=dict(title="IMOEX"),
        yaxis2=dict(title="Макрофакторы", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=40, t=60, b=40),
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
    fig.update_layout(
        template="plotly_dark",
        title="Корреляционная матрица",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=40, t=60, b=40),
    )
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


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font=dict(size=16, color="#cbd5e1"),
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _parse_portfolio_weights(weight_text: str, n: int) -> list[float]:
    if n <= 0:
        return []
    try:
        parts = [p.strip() for p in (weight_text or "").split(",") if p.strip()]
        vals = [float(x) for x in parts]
    except Exception:
        vals = []
    if len(vals) != n or sum(vals) <= 0:
        return [1.0 / n] * n
    s = sum(vals)
    return [v / s for v in vals]


def _predict_portfolio_snapshot(
    tickers: list[str],
    weights: list[float],
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
    regime: str,
) -> tuple[float, float]:
    if not tickers:
        raise ValueError("Портфель пуст")
    pred = 0.0
    cur = 0.0
    for t, w in zip(tickers, weights):
        p = predict_stock_scenario(t, oil, key_rate, usd_rub, inflation, regime=regime)
        s = get_stock_history_service(t)
        col = f"{t.lower()}_close"
        c = float(pd.to_numeric(s[col], errors="coerce").dropna().iloc[-1]) if col in s.columns else 0.0
        pred += w * float(p)
        cur += w * c
    return float(pred), float(cur)


def build_layout(df: pd.DataFrame) -> html.Div:
    default = df.iloc[-1]
    slider_common = {
        "marks": None,
        "tooltip": {"placement": "bottom", "always_visible": True},
    }
    return html.Div(
        [
            html.H2("Экономический симулятор сценариев (Dash)", className="title"),
            dcc.Tabs(
                id="tabs",
                value="tab-history",
                className="tabs",
                children=[
                    dcc.Tab(
                        label="📊 Исторические данные",
                        value="tab-history",
                        className="tab",
                        selected_className="tab--selected",
                        children=[
                            html.Div(
                                [
                                    dcc.Graph(id="hist-chart", figure=_build_historical_chart(df), className="graph"),
                                    dcc.Graph(id="corr-chart", figure=_build_corr_fig(df), className="graph"),
                                    html.Div(
                                        "Пояснение: корреляция показывает линейную связь факторов от -1 до +1.",
                                        className="hint",
                                    ),
                                ],
                                className="card",
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="🎯 Сценарный анализ",
                        value="tab-scenario",
                        className="tab",
                        selected_className="tab--selected",
                        children=[
                            html.Div(
                                [
                                    html.Label("Целевой актив", className="label"),
                                    dcc.RadioItems(
                                        id="asset-type",
                                        className="dash-radio-items",
                                        options=[
                                            {"label": "IMOEX", "value": "imoex"},
                                            {"label": "Акция MOEX", "value": "stock"},
                                            {"label": "Портфель MOEX", "value": "portfolio"},
                                        ],
                                        value="imoex",
                                        inline=True,
                                    ),
                                    html.Label("Тикер (для акции)", className="label"),
                                    dcc.Dropdown(id="ticker", options=_ticker_options(), value="LKOH", clearable=False),
                                    html.Label("Тикеры портфеля (для портфеля)", className="label"),
                                    dcc.Dropdown(
                                        id="portfolio-tickers",
                                        options=_ticker_options(),
                                        value=["LKOH", "SBER", "GAZP"],
                                        multi=True,
                                        clearable=False,
                                    ),
                                    html.Label("Веса портфеля через запятую, % (например: 40,35,25)", className="label"),
                                    dcc.Input(
                                        id="portfolio-weights",
                                        className="dash-input",
                                        type="text",
                                        value="40,35,25",
                                        style={"width": "100%"},
                                    ),
                                    html.Label("Режим модели", className="label"),
                                    dcc.Dropdown(id="regime", options=REGIME_OPTIONS, value="all", clearable=False),
                                    html.Label("Brent", className="label"),
                                    dcc.Slider(
                                        id="oil",
                                        min=30,
                                        max=130,
                                        step=5,
                                        value=float(default["brent_usd"]),
                                        **slider_common,
                                    ),
                                    html.Label("Ключевая ставка", className="label"),
                                    dcc.Slider(
                                        id="key-rate",
                                        min=5,
                                        max=30,
                                        step=0.5,
                                        value=float(default["key_rate"]),
                                        **slider_common,
                                    ),
                                    html.Label("USD/RUB", className="label"),
                                    dcc.Slider(
                                        id="usd-rub",
                                        min=60,
                                        max=150,
                                        step=1,
                                        value=float(default["usd_rub"]),
                                        **slider_common,
                                    ),
                                    html.Label("Инфляция", className="label"),
                                    dcc.Slider(
                                        id="inflation",
                                        min=2,
                                        max=20,
                                        step=0.5,
                                        value=float(default["inflation"]),
                                        **slider_common,
                                    ),
                                ]
                            , className="card"),
                            html.Br(),
                            html.Button("Рассчитать сценарий", id="run-scenario", n_clicks=0, className="btn-primary"),
                            html.Div(id="scenario-output", style={"marginTop": "16px"}, className="card"),
                        ],
                    ),
                    dcc.Tab(
                        label="🎲 Монте-Карло",
                        value="tab-mc",
                        className="tab",
                        selected_className="tab--selected",
                        children=[
                            html.Button("Запустить Monte Carlo", id="run-mc", n_clicks=0, className="btn-primary"),
                            dcc.Graph(
                                id="mc-fig",
                                className="graph",
                                figure=_empty_figure("Monte Carlo", "Нажмите «Запустить Monte Carlo»."),
                            ),
                            html.Div(
                                [
                                    html.Label("Диапазон (бин)", className="label"),
                                    dcc.Dropdown(id="mc-bin-select", clearable=False),
                                ]
                            , className="card"),
                            html.Div(id="mc-stats", className="card"),
                            dcc.Store(id="mc-store"),
                        ],
                    ),
                    dcc.Tab(
                        label="🌪️ Чувствительность",
                        value="tab-sobol",
                        className="tab",
                        selected_className="tab--selected",
                        children=[
                            html.Button("Рассчитать Sobol", id="run-sobol", n_clicks=0, className="btn-primary"),
                            dcc.Graph(
                                id="sobol-fig",
                                className="graph",
                                figure=_empty_figure("Sobol-анализ", "Нажмите «Рассчитать Sobol»."),
                            ),
                            html.Div([html.Label("Фактор", className="label"), dcc.Dropdown(id="sobol-factor-select", clearable=False)], className="card"),
                            html.Div(id="sobol-stats", className="card"),
                            dcc.Store(id="sobol-store"),
                        ],
                    ),
                ],
            ),
        ],
        style={"maxWidth": "1500px", "margin": "0 auto", "padding": "16px"},
        className="app-shell",
    )


@callback(
    Output("scenario-output", "children"),
    Input("run-scenario", "n_clicks"),
    State("asset-type", "value"),
    State("ticker", "value"),
    State("regime", "value"),
    State("portfolio-tickers", "value"),
    State("portfolio-weights", "value"),
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
    portfolio_tickers: list[str] | None,
    portfolio_weights_text: str,
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
) -> html.Div:
    df = _load_dataset()
    try:
        if asset_type == "stock":
            pred = predict_stock_scenario(ticker, oil, key_rate, usd_rub, inflation, regime=regime)
            cur = _asset_current(df, "stock", ticker)
            label = f"Акция {ticker}"
        elif asset_type == "portfolio":
            tickers = [str(t).upper().strip() for t in (portfolio_tickers or []) if str(t).strip()]
            weights = _parse_portfolio_weights(portfolio_weights_text, len(tickers))
            pred, cur = _predict_portfolio_snapshot(
                tickers,
                weights,
                oil=oil,
                key_rate=key_rate,
                usd_rub=usd_rub,
                inflation=inflation,
                regime=regime,
            )
            label = f"Портфель ({', '.join(tickers[:4])})"
        else:
            pred = predict_scenario(oil, key_rate, usd_rub, inflation, regime=regime)
            cur = _asset_current(df, "imoex", ticker)
            label = "IMOEX"
    except Exception as exc:
        return html.Div([html.B("Ошибка расчета сценария"), html.Div(str(exc))])
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
    State("portfolio-tickers", "value"),
    State("portfolio-weights", "value"),
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
    portfolio_tickers: list[str] | None,
    portfolio_weights_text: str,
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
    tickers = [str(t).upper().strip() for t in (portfolio_tickers or []) if str(t).strip()]
    weights = _parse_portfolio_weights(portfolio_weights_text, len(tickers))
    weight_map = {t: w for t, w in zip(tickers, weights)}
    try:
        result = run_monte_carlo_service(
            controls=controls,
            mc_runs=5000,
            asset_type=asset_type,
            ticker=ticker if asset_type == "stock" else None,
            adjustments={},
            regime=regime,
            portfolio_tickers=tickers if asset_type == "portfolio" else [],
            portfolio_weights=weight_map if asset_type == "portfolio" else None,
        )
    except Exception as exc:
        return (
            _empty_figure("Monte Carlo", f"Ошибка: {exc}"),
            {},
            [],
            None,
        )
    bins = result["hist_bins"].copy()
    bins["label"] = bins.apply(lambda r: f"{int(r['bin_index'])+1}: {r['left']:.1f} .. {r['right']:.1f}", axis=1)
    options = [{"label": l, "value": l} for l in bins["label"].tolist()]
    value = options[0]["value"] if options else None
    fig = go.Figure(result["figure"])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    safe_bins = bins.replace({np.nan: None}).to_dict("records")
    store = {
        "var_5": float(result["var_5"]),
        "p50": float(result["p50"]),
        "var_95": float(result["var_95"]),
        "prob_drop_20": float(result["prob_drop_20"]),
        "n_simulations": int(result["n_simulations"]),
        "asset_label": str(result["asset_label"]),
        "current_level": float(result["current_level"]),
        "hist_bins": safe_bins,
    }
    return fig, store, options, value


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
    State("portfolio-tickers", "value"),
    State("portfolio-weights", "value"),
    prevent_initial_call=True,
)
def run_sobol_cb(
    _n: int,
    asset_type: str,
    ticker: str,
    regime: str,
    portfolio_tickers: list[str] | None,
    portfolio_weights_text: str,
):
    tickers = [str(t).upper().strip() for t in (portfolio_tickers or []) if str(t).strip()]
    weights = _parse_portfolio_weights(portfolio_weights_text, len(tickers))
    weight_map = {t: w for t, w in zip(tickers, weights)}
    try:
        result = run_sobol_service(
            n_samples=256,
            asset_type=asset_type,
            ticker=ticker if asset_type == "stock" else None,
            regime=regime,
            portfolio_tickers=tickers if asset_type == "portfolio" else [],
            portfolio_weights=weight_map if asset_type == "portfolio" else None,
        )
    except Exception as exc:
        return (
            _empty_figure("Sobol-анализ", f"Ошибка: {exc}"),
            {},
            [],
            None,
        )
    sobol_df = result["sobol_df"].copy()
    labels = sobol_df["factor_label"].astype(str).tolist()
    options = [{"label": x, "value": x} for x in labels]
    value = labels[0] if labels else None
    fig = go.Figure(result["figure"])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    store = {
        "sobol_df": sobol_df.replace({np.nan: None}).to_dict("records"),
        "top_factor": str(result["top_factor"]),
        "top_s1": float(result["top_s1"]),
        "y_variance": float(result["y_variance"]),
        "problem": result["problem"],
    }
    return fig, store, options, value


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
