"""Русифицированный Streamlit-дашборд экономического симулятора РФ."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_loader import load_moex_stock, load_moex_tickers
from src.model import (
    apply_scenario_adjustments,
    explain_imoex_drivers,
    explain_stock_drivers,
    load_model_artifact,
    load_stock_model_artifact,
    predict_scenario,
    predict_stock_scenario,
)
from src.monte_carlo import run_monte_carlo
from src.preprocessing import build_processed_dataset
from src.sensitivity import run_sobol_sensitivity


st.set_page_config(
    page_title="Экономический симулятор РФ",
    page_icon="📈",
    layout="wide",
)


BASE_SLIDER_CONFIG = {
    "oil": {"label": "Нефть Brent ($)", "min": 30.0, "max": 130.0, "step": 5.0},
    "key_rate": {"label": "Ключевая ставка (%)", "min": 5.0, "max": 30.0, "step": 0.5},
    "usd_rub": {"label": "Курс USD/RUB", "min": 60.0, "max": 150.0, "step": 1.0},
    "inflation": {"label": "Инфляция (%)", "min": 2.0, "max": 20.0, "step": 0.5},
}

ADDITIONAL_SLIDER_CONFIG = {
    "market_sentiment": {"label": "Рыночный сентимент (%)", "min": -20.0, "max": 20.0, "step": 1.0},
    "liquidity_effect": {"label": "Эффект ликвидности (%)", "min": -15.0, "max": 15.0, "step": 1.0},
    "geopolitics_effect": {"label": "Геополитический фактор (%)", "min": -25.0, "max": 25.0, "step": 1.0},
    "regulatory_effect": {"label": "Регуляторный фактор (%)", "min": -20.0, "max": 20.0, "step": 1.0},
    "uncertainty_scale": {"label": "Масштаб волатильности", "min": 0.5, "max": 2.0, "step": 0.1},
}

REGIME_LABELS = {
    "Весь период (5 лет)": "all",
    "До февраля 2022": "pre_2022",
    "С февраля 2022": "post_2022",
}

FACTOR_LABELS = {
    "brent_usd": "Нефть Brent",
    "key_rate": "Ключевая ставка",
    "usd_rub": "USD/RUB",
    "inflation": "Инфляция",
    "imoex_close": "IMOEX",
}

RESERVE_TICKERS = ["LKOH", "ROSN", "TATN", "SBER", "GAZP", "NVTK", "GMKN", "MOEX", "VTBR", "T", "YDEX"]


def _clamp(value: float, config: dict[str, float]) -> float:
    return float(min(max(value, config["min"]), config["max"]))


def _normalize_weights(weight_map: dict[str, float]) -> dict[str, float]:
    items = [(k.upper().strip(), float(v)) for k, v in weight_map.items() if k and k.strip()]
    if not items:
        return {}
    arr = np.array([v for _, v in items], dtype=float)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if np.isclose(total, 0):
        arr = np.array([1.0 / len(items)] * len(items), dtype=float)
    else:
        arr = arr / total
    return {ticker: float(w) for (ticker, _), w in zip(items, arr)}


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return build_processed_dataset()


@st.cache_data(show_spinner=False)
def get_stock_data(ticker: str) -> pd.DataFrame:
    return load_moex_stock(ticker)


@st.cache_data(show_spinner=False)
def get_moex_ticker_universe() -> pd.DataFrame:
    df = load_moex_tickers().copy()
    if "ticker" not in df.columns:
        raise ValueError("Список тикеров MOEX не содержит колонку 'ticker'")
    if "shortname" not in df.columns:
        df["shortname"] = df["ticker"]

    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.strip()
    df["shortname"] = df["shortname"].fillna(df["ticker"]).astype(str).str.strip()
    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"], keep="first").sort_values("ticker").reset_index(drop=True)
    return df[["ticker", "shortname"]]


@st.cache_resource(show_spinner=False)
def get_imoex_model(regime: str) -> dict[str, Any]:
    return load_model_artifact(regime=regime)


@st.cache_resource(show_spinner=False)
def get_stock_model(ticker: str, regime: str) -> dict[str, Any]:
    return load_stock_model_artifact(ticker, regime=regime)


def _defaults_from_data(df: pd.DataFrame) -> dict[str, float]:
    last = df.iloc[-1]
    return {
        "oil": _clamp(float(last["brent_usd"]), BASE_SLIDER_CONFIG["oil"]),
        "key_rate": _clamp(float(last["key_rate"]), BASE_SLIDER_CONFIG["key_rate"]),
        "usd_rub": _clamp(float(last["usd_rub"]), BASE_SLIDER_CONFIG["usd_rub"]),
        "inflation": _clamp(float(last["inflation"]), BASE_SLIDER_CONFIG["inflation"]),
        "market_sentiment": 0.0,
        "liquidity_effect": 0.0,
        "geopolitics_effect": 0.0,
        "regulatory_effect": 0.0,
        "uncertainty_scale": 1.0,
    }


def _preset_values(defaults: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        "base": defaults,
        "optimistic": {
            "oil": 100.0,
            "key_rate": 8.0,
            "usd_rub": 75.0,
            "inflation": 4.0,
            "market_sentiment": 8.0,
            "liquidity_effect": 5.0,
            "geopolitics_effect": 3.0,
            "regulatory_effect": 2.0,
            "uncertainty_scale": 0.8,
        },
        "pessimistic": {
            "oil": 50.0,
            "key_rate": 20.0,
            "usd_rub": 120.0,
            "inflation": 12.0,
            "market_sentiment": -8.0,
            "liquidity_effect": -6.0,
            "geopolitics_effect": -10.0,
            "regulatory_effect": -7.0,
            "uncertainty_scale": 1.3,
        },
    }


def _apply_preset(preset_name: str, defaults: dict[str, float]) -> None:
    selected = _preset_values(defaults)[preset_name]
    for key, value in selected.items():
        cfg = BASE_SLIDER_CONFIG.get(key) or ADDITIONAL_SLIDER_CONFIG.get(key)
        st.session_state[key] = _clamp(float(value), cfg) if cfg else float(value)


def _ticker_ui_options() -> tuple[list[str], dict[str, str], bool]:
    try:
        universe = get_moex_ticker_universe()
        ticker_list = universe["ticker"].tolist()
        if not ticker_list:
            raise ValueError("Пустой список тикеров")
        names = {
            row["ticker"]: row["shortname"]
            for row in universe.to_dict("records")
        }
        return ticker_list, names, False
    except Exception:
        names = {ticker: ticker for ticker in RESERVE_TICKERS}
        return RESERVE_TICKERS.copy(), names, True


def _ticker_label(ticker: str, names: dict[str, str]) -> str:
    shortname = names.get(ticker, "").strip()
    if shortname and shortname.upper() != ticker.upper():
        return f"{ticker} — {shortname}"
    return ticker


def _get_portfolio_controls(ticker_options: list[str], names: dict[str, str]) -> tuple[list[str], dict[str, float]]:
    default_portfolio = [t for t in ("LKOH", "ROSN", "TATN") if t in ticker_options]
    if len(default_portfolio) < 2:
        default_portfolio = ticker_options[: min(3, len(ticker_options))]

    picked = st.sidebar.multiselect(
        "Тикеры портфеля (2-5)",
        ticker_options,
        default=default_portfolio,
        format_func=lambda x: _ticker_label(x, names),
    )

    selected = [p.upper().strip() for p in picked][:5]
    if len(selected) < 2:
        st.sidebar.warning("Для портфеля выберите минимум 2 тикера.")

    raw_weights: dict[str, float] = {}
    if selected:
        st.sidebar.markdown("**Веса портфеля (%)**")
        default_share = 100.0 / len(selected)
        for t in selected:
            key = f"weight_{t}"
            if key not in st.session_state:
                st.session_state[key] = default_share
            raw_weights[t] = st.sidebar.slider(
                f"Вес {t}",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                value=float(st.session_state[key]),
                key=key,
            )

    normalized = _normalize_weights(raw_weights)
    if normalized:
        st.sidebar.caption(
            "Нормализованные веса: "
            + ", ".join(f"{t} {w*100:.1f}%" for t, w in normalized.items())
        )

    return selected, normalized


def _render_sidebar_controls(
    defaults: dict[str, float],
) -> tuple[dict[str, float], str, str | None, str, int, int, list[str], dict[str, float]]:
    st.sidebar.header("Параметры сценария")

    if "oil" not in st.session_state:
        for key, value in defaults.items():
            st.session_state[key] = value

    asset_choice = st.sidebar.radio(
        "Целевой актив",
        ["IMOEX", "Акция MOEX", "Портфель MOEX"],
        horizontal=False,
    )

    ticker_options, ticker_names, fallback_used = _ticker_ui_options()
    if fallback_used:
        st.sidebar.warning(
            "Полный список бумаг с MOEX временно недоступен. Используется резервный набор тикеров."
        )

    regime_label = st.sidebar.selectbox("Режим модели", list(REGIME_LABELS.keys()), index=0)
    regime = REGIME_LABELS[regime_label]

    horizon_months = st.sidebar.slider("Горизонт прогноза (месяцев)", min_value=6, max_value=12, step=1, value=9)
    n_paths = st.sidebar.slider("Число траекторий (веер)", min_value=100, max_value=800, step=50, value=300)

    ticker: str | None = None
    portfolio_tickers: list[str] = []
    portfolio_weights: dict[str, float] = {}

    if asset_choice == "Акция MOEX":
        default_ticker = "SBER" if "SBER" in ticker_options else ticker_options[0]
        selected = st.sidebar.selectbox(
            "Тикер MOEX",
            ticker_options,
            index=ticker_options.index(default_ticker),
            format_func=lambda x: _ticker_label(x, ticker_names),
        )
        custom = st.sidebar.text_input("Или введите тикер вручную", value=selected)
        ticker = (custom or selected).upper().strip()
    elif asset_choice == "Портфель MOEX":
        portfolio_tickers, portfolio_weights = _get_portfolio_controls(ticker_options, ticker_names)

    b1, b2, b3 = st.sidebar.columns(3)
    if b1.button("Базовый", use_container_width=True):
        _apply_preset("base", defaults)
    if b2.button("Оптимистичный", use_container_width=True):
        _apply_preset("optimistic", defaults)
    if b3.button("Пессимистичный", use_container_width=True):
        _apply_preset("pessimistic", defaults)

    controls: dict[str, float] = {}
    st.sidebar.subheader("Базовые макропараметры")
    for key, cfg in BASE_SLIDER_CONFIG.items():
        controls[key] = st.sidebar.slider(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            step=float(cfg["step"]),
            value=float(st.session_state[key]),
            key=key,
        )

    st.sidebar.subheader("Дополнительные факторы")
    for key, cfg in ADDITIONAL_SLIDER_CONFIG.items():
        controls[key] = st.sidebar.slider(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            step=float(cfg["step"]),
            value=float(st.session_state[key]),
            key=key,
        )

    return (
        controls,
        "stock" if asset_choice == "Акция MOEX" else ("portfolio" if asset_choice == "Портфель MOEX" else "imoex"),
        ticker,
        regime,
        horizon_months,
        n_paths,
        portfolio_tickers,
        portfolio_weights,
    )


def _adjustments_from_controls(controls: dict[str, float]) -> dict[str, float]:
    return {
        "market_sentiment": controls["market_sentiment"],
        "liquidity_effect": controls["liquidity_effect"],
        "geopolitics_effect": controls["geopolitics_effect"],
        "regulatory_effect": controls["regulatory_effect"],
    }


def _portfolio_series(tickers: list[str], weights: dict[str, float]) -> pd.DataFrame:
    if len(tickers) < 2:
        return pd.DataFrame(columns=["date", "portfolio_index"])

    frames: list[pd.DataFrame] = []
    for symbol in tickers:
        col = f"{symbol.lower()}_close"
        s_df = get_stock_data(symbol)[["date", col]].copy()
        frames.append(s_df)

    merged = frames[0]
    for fr in frames[1:]:
        merged = merged.merge(fr, on="date", how="inner")

    merged = merged.sort_values("date").dropna()
    if merged.empty:
        return pd.DataFrame(columns=["date", "portfolio_index"])

    for symbol in tickers:
        col = f"{symbol.lower()}_close"
        base = float(merged[col].iloc[0])
        merged[f"{symbol.lower()}_idx"] = merged[col] / base * 100.0

    merged["portfolio_index"] = 0.0
    for symbol in tickers:
        merged["portfolio_index"] += merged[f"{symbol.lower()}_idx"] * float(weights.get(symbol, 0.0))

    return merged[["date", "portfolio_index"]]


def _build_plot_df(
    df: pd.DataFrame,
    asset_type: str,
    ticker: str | None,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float],
) -> pd.DataFrame:
    plot_df = df.copy()

    if asset_type == "stock" and ticker:
        stock_df = get_stock_data(ticker)
        stock_col = f"{ticker.lower()}_close"
        plot_df = plot_df.merge(stock_df[["date", stock_col]], on="date", how="left")

    if asset_type == "portfolio":
        pf = _portfolio_series(portfolio_tickers, portfolio_weights)
        if not pf.empty:
            plot_df = plot_df.merge(pf, on="date", how="left")

    return plot_df


def _historical_chart(
    df: pd.DataFrame,
    asset_type: str,
    ticker: str | None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["imoex_close"], name="IMOEX", line=dict(width=3, color="#1f77b4")),
        secondary_y=False,
    )

    if asset_type == "stock" and ticker:
        stock_col = f"{ticker.lower()}_close"
        if stock_col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["date"], y=df[stock_col], name=ticker.upper(), line=dict(width=2, color="#ff7f0e")),
                secondary_y=False,
            )

    if asset_type == "portfolio" and "portfolio_index" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["portfolio_index"],
                name="Портфель (индекс=100)",
                line=dict(width=2, color="#8c564b"),
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["usd_rub"], name="USD/RUB", line=dict(width=2, color="#d62728")),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["key_rate"], name="Ключевая ставка", line=dict(width=2, color="#2ca02c")),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["brent_usd"], name="Brent", line=dict(width=2, color="#9467bd")),
        secondary_y=True,
    )

    fig.update_layout(
        title="Историческая динамика рынка и макропоказателей",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Индекс/цена актива", secondary_y=False)
    fig.update_yaxes(title_text="Макропоказатели", secondary_y=True)
    return fig


def _correlation_columns(df: pd.DataFrame, asset_type: str, ticker: str | None) -> list[str]:
    cols = ["imoex_close", "key_rate", "usd_rub", "brent_usd", "inflation"]

    if asset_type == "stock" and ticker:
        stock_col = f"{ticker.lower()}_close"
        if stock_col in df.columns:
            cols.insert(0, stock_col)

    if asset_type == "portfolio" and "portfolio_index" in df.columns:
        cols.insert(0, "portfolio_index")

    return cols


def _factor_label_map(ticker: str | None = None) -> dict[str, str]:
    label_map = {
        "imoex_close": "IMOEX",
        "key_rate": "Ключевая ставка",
        "usd_rub": "USD/RUB",
        "brent_usd": "Brent",
        "inflation": "Инфляция",
        "portfolio_index": "Портфель",
    }
    if ticker:
        label_map[f"{ticker.lower()}_close"] = ticker.upper()
    return label_map


def _correlation_heatmap(df: pd.DataFrame, asset_type: str, ticker: str | None) -> go.Figure:
    cols = _correlation_columns(df, asset_type, ticker)
    labels = _factor_label_map(ticker)
    corr = df[cols].corr()
    x_labels = [labels.get(c, c) for c in corr.columns]
    y_labels = [labels.get(c, c) for c in corr.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=x_labels,
            y=y_labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar=dict(title="Corr"),
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            customdata=np.array(corr.columns)[None, :].repeat(len(corr.index), axis=0),
        )
    )
    fig.update_layout(title="Корреляционная матрица", template="plotly_white")
    return fig


def _correlation_insights(df: pd.DataFrame, asset_type: str, ticker: str | None) -> list[str]:
    cols = _correlation_columns(df, asset_type, ticker)
    corr = df[cols].corr()
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(corr.columns):
        for j, right in enumerate(corr.columns):
            if j <= i:
                continue
            pairs.append((left, right, float(corr.iloc[i, j])))

    if not pairs:
        return []

    strongest_abs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:3]

    label_map = _factor_label_map(ticker)

    insights = []
    for left, right, value in strongest_abs:
        direction = "прямая" if value >= 0 else "обратная"
        strength = "сильная" if abs(value) >= 0.7 else ("умеренная" if abs(value) >= 0.4 else "слабая")
        insights.append(
            f"**{label_map.get(left, left)} ↔ {label_map.get(right, right)}**: "
            f"{value:+.2f} ({strength}, {direction} связь)"
        )
    return insights


def _correlation_drilldown_figure(df: pd.DataFrame, x_col: str, y_col: str, ticker: str | None) -> go.Figure:
    subset = df[[x_col, y_col]].dropna().copy()
    labels = _factor_label_map(ticker)
    x_label = labels.get(x_col, x_col)
    y_label = labels.get(y_col, y_col)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            marker=dict(size=8, opacity=0.75, color="#1f77b4"),
            name="Наблюдения",
        )
    )

    corr_val = float(subset[x_col].corr(subset[y_col])) if len(subset) >= 2 else float("nan")
    if len(subset) >= 2:
        slope, intercept = np.polyfit(subset[x_col], subset[y_col], 1)
        x_line = np.linspace(float(subset[x_col].min()), float(subset[x_col].max()), 50)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="#d62728", width=2),
                name="Линейный тренд",
            )
        )

    fig.update_layout(
        title=f"Drilldown: {x_label} vs {y_label} (corr={corr_val:+.2f})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
    )
    return fig


def _scenario_stress_index(controls: dict[str, float], adjustments: dict[str, float]) -> float:
    stress_components = [
        abs(controls["key_rate"] - 10.0) / 20.0,
        abs(controls["usd_rub"] - 85.0) / 65.0,
        abs(controls["inflation"] - 4.0) / 16.0,
        abs(controls["oil"] - 80.0) / 50.0,
        max(0.0, controls["uncertainty_scale"] - 1.0) / 1.0,
        max(0.0, -adjustments["market_sentiment"]) / 20.0,
        max(0.0, -adjustments["liquidity_effect"]) / 15.0,
        max(0.0, -adjustments["geopolitics_effect"]) / 25.0,
        max(0.0, -adjustments["regulatory_effect"]) / 20.0,
    ]
    raw = float(np.mean([min(max(x, 0.0), 1.0) for x in stress_components]))
    return min(max(raw * 100.0, 0.0), 100.0)


def _adjustment_multiplier(adjustments: dict[str, float]) -> float:
    return 1.0 + float(sum(adjustments.values())) / 100.0


def _linear_predict(artifact: dict[str, Any], features: dict[str, np.ndarray | float]) -> np.ndarray:
    feature_columns = list(artifact.get("feature_columns", []))
    coefs = artifact.get("coefs")
    intercept = artifact.get("intercept")

    if (not isinstance(coefs, dict) or intercept is None) and artifact.get("model") is not None:
        model = artifact["model"]
        raw_coef = np.asarray(getattr(model, "coef_", []), dtype=float).ravel()
        if feature_columns and len(raw_coef) == len(feature_columns):
            coefs = {col: float(value) for col, value in zip(feature_columns, raw_coef)}
        if intercept is None and hasattr(model, "intercept_"):
            intercept = float(model.intercept_)

    if not isinstance(coefs, dict):
        coefs = {}
    if intercept is None:
        intercept = 0.0
    if not feature_columns:
        feature_columns = list(features.keys())

    first_key = next(iter(features))
    first_val = np.asarray(features[first_key], dtype=float)
    out = np.full(first_val.shape, float(intercept), dtype=float)

    for feature in feature_columns:
        out += float(coefs.get(feature, 0.0)) * np.asarray(features[feature], dtype=float)
    return out


def _gauge_chart(prediction: float, current: float, asset_label: str) -> go.Figure:
    axis_max = max(prediction, current) * 1.35
    axis_min = min(prediction, current) * 0.65

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": current, "relative": True, "valueformat": ".1%"},
            number={"suffix": f" {asset_label}", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [axis_min, axis_max]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [axis_min, current], "color": "#fde0dd"},
                    {"range": [current, axis_max], "color": "#e5f5e0"},
                ],
                "threshold": {
                    "line": {"color": "#111111", "width": 2},
                    "thickness": 0.75,
                    "value": current,
                },
            },
            title={"text": f"Прогноз {asset_label}"},
        )
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=80, b=20))
    return fig


def _current_asset_value(
    df: pd.DataFrame,
    asset_type: str,
    ticker: str | None,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float],
) -> float:
    if asset_type == "stock":
        if not ticker:
            raise ValueError("Не указан тикер акции")
        s_df = get_stock_data(ticker)
        return float(s_df[f"{ticker.lower()}_close"].dropna().iloc[-1])

    if asset_type == "portfolio":
        if len(portfolio_tickers) < 2:
            return 100.0
        pf = _portfolio_series(portfolio_tickers, portfolio_weights)
        if pf.empty:
            return 100.0
        return float(pf["portfolio_index"].dropna().iloc[-1])

    return float(df["imoex_close"].dropna().iloc[-1])


def _predict_asset_snapshot(
    asset_type: str,
    ticker: str | None,
    controls: dict[str, float],
    adjustments: dict[str, float],
    regime: str,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float],
) -> tuple[float, float, str, dict[str, Any] | None, dict[str, float]]:
    base_imoex = predict_scenario(
        oil=controls["oil"],
        key_rate=controls["key_rate"],
        usd_rub=controls["usd_rub"],
        inflation=controls["inflation"],
        regime=regime,
    )

    if asset_type == "stock":
        if not ticker:
            raise ValueError("Не указан тикер акции")
        raw_prediction = predict_stock_scenario(
            ticker=ticker,
            oil=controls["oil"],
            key_rate=controls["key_rate"],
            usd_rub=controls["usd_rub"],
            inflation=controls["inflation"],
            imoex_value=base_imoex,
            adjustments=None,
            regime=regime,
        )
        adjusted_prediction = apply_scenario_adjustments(raw_prediction, adjustments)
        return adjusted_prediction, raw_prediction, ticker.upper(), get_stock_model(ticker, regime), {}

    if asset_type == "portfolio":
        if len(portfolio_tickers) < 2:
            raise ValueError("Для портфеля выберите минимум 2 тикера")

        weights = portfolio_weights or _normalize_weights({t: 1.0 for t in portfolio_tickers})
        stock_pred_map: dict[str, float] = {}
        weighted_return = 0.0
        for symbol in portfolio_tickers:
            current = float(get_stock_data(symbol)[f"{symbol.lower()}_close"].dropna().iloc[-1])
            pred = predict_stock_scenario(
                ticker=symbol,
                oil=controls["oil"],
                key_rate=controls["key_rate"],
                usd_rub=controls["usd_rub"],
                inflation=controls["inflation"],
                imoex_value=base_imoex,
                adjustments=None,
                regime=regime,
            )
            stock_pred_map[symbol] = pred
            weighted_return += float(weights.get(symbol, 0.0)) * (pred / current - 1.0)

        raw_prediction = 100.0 * (1.0 + weighted_return)
        adjusted_prediction = apply_scenario_adjustments(raw_prediction, adjustments)
        return adjusted_prediction, raw_prediction, "Портфель (индекс=100)", None, stock_pred_map

    raw_prediction = base_imoex
    adjusted_prediction = apply_scenario_adjustments(raw_prediction, adjustments)
    return adjusted_prediction, raw_prediction, "IMOEX", get_imoex_model(regime), {}


def _driver_comment(
    asset_type: str,
    ticker: str | None,
    controls: dict[str, float],
    adjustments: dict[str, float],
    regime: str,
    raw_imoex: float,
    stock_pred_map: dict[str, float],
    portfolio_tickers: list[str],
) -> str:
    total_adj = float(sum(adjustments.values()))

    if asset_type == "imoex":
        drivers = explain_imoex_drivers(
            oil=controls["oil"],
            key_rate=controls["key_rate"],
            usd_rub=controls["usd_rub"],
            inflation=controls["inflation"],
            regime=regime,
        )
    elif asset_type == "stock" and ticker:
        drivers = explain_stock_drivers(
            ticker=ticker,
            oil=controls["oil"],
            key_rate=controls["key_rate"],
            usd_rub=controls["usd_rub"],
            inflation=controls["inflation"],
            imoex_value=raw_imoex,
            regime=regime,
        )
    else:
        # Для портфеля берем драйверы через канал IMOEX + лидеров по акциям.
        drivers = explain_imoex_drivers(
            oil=controls["oil"],
            key_rate=controls["key_rate"],
            usd_rub=controls["usd_rub"],
            inflation=controls["inflation"],
            regime=regime,
        )

    positives = drivers[drivers["contribution"] > 0]
    negatives = drivers[drivers["contribution"] < 0]

    pos_line = ""
    neg_line = ""
    if not positives.empty:
        row = positives.iloc[0]
        pos_line = f"Позитивный драйвер: {FACTOR_LABELS.get(row['factor'], row['factor'])} ({row['contribution']:+.1f})"
    if not negatives.empty:
        row = negatives.iloc[0]
        neg_line = f"Негативный драйвер: {FACTOR_LABELS.get(row['factor'], row['factor'])} ({row['contribution']:+.1f})"

    addon = f"Сумма доп. поправок сценария: {total_adj:+.2f}%"

    if asset_type == "portfolio" and stock_pred_map:
        leader = max(stock_pred_map.items(), key=lambda x: x[1])[0]
        laggard = min(stock_pred_map.items(), key=lambda x: x[1])[0]
        pf_note = f"Лидер портфеля по прогнозу: {leader}. Самый слабый вклад: {laggard}."
        return " | ".join([x for x in [pos_line, neg_line, addon, pf_note] if x])

    return " | ".join([x for x in [pos_line, neg_line, addon] if x])


def _simulate_macro_paths(
    controls: dict[str, float],
    horizon_months: int,
    n_paths: int,
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    means = {
        "oil": float(df["brent_usd"].mean()),
        "key_rate": float(df["key_rate"].mean()),
        "usd_rub": float(df["usd_rub"].mean()),
        "inflation": float(df["inflation"].mean()),
    }
    stds = {
        "oil": float(df["brent_usd"].std(ddof=1)),
        "key_rate": float(df["key_rate"].std(ddof=1)),
        "usd_rub": float(df["usd_rub"].std(ddof=1)),
        "inflation": float(df["inflation"].std(ddof=1)),
    }

    scale = float(controls["uncertainty_scale"])
    alpha = 0.25
    shock_scale = 0.35

    rng = np.random.default_rng(42)
    paths = {k: np.zeros((n_paths, horizon_months), dtype=float) for k in ["oil", "key_rate", "usd_rub", "inflation"]}

    for key in paths:
        paths[key][:, 0] = float(controls[key])

    for t in range(1, horizon_months):
        for key in paths:
            prev = paths[key][:, t - 1]
            mu = means[key]
            sigma = max(stds[key], 1e-6) * scale * shock_scale
            noise = rng.normal(0.0, sigma, size=n_paths)
            nxt = prev + alpha * (mu - prev) + noise

            cfg = BASE_SLIDER_CONFIG[key]
            nxt = np.clip(nxt, cfg["min"], cfg["max"])
            paths[key][:, t] = nxt

    return paths


def _predict_paths(
    asset_type: str,
    ticker: str | None,
    paths: dict[str, np.ndarray],
    adjustments: dict[str, float],
    regime: str,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float],
) -> np.ndarray:
    oil = paths["oil"]
    key_rate = paths["key_rate"]
    usd = paths["usd_rub"]
    infl = paths["inflation"]

    n_paths, horizon = oil.shape
    mult = _adjustment_multiplier(adjustments)

    imoex_artifact = get_imoex_model(regime)
    imoex_raw = _linear_predict(
        imoex_artifact,
        {
            "key_rate": key_rate,
            "usd_rub": usd,
            "brent_usd": oil,
            "inflation": infl,
        },
    )

    if asset_type == "imoex":
        return imoex_raw * mult

    if asset_type == "stock":
        if not ticker:
            raise ValueError("Не указан тикер акции")
        stock_artifact = get_stock_model(ticker, regime)
        stock_raw = _linear_predict(
            stock_artifact,
            {
                "key_rate": key_rate,
                "usd_rub": usd,
                "brent_usd": oil,
                "inflation": infl,
                "imoex_close": imoex_raw,
            },
        )
        return stock_raw * mult

    if len(portfolio_tickers) < 2:
        raise ValueError("Для портфеля выберите минимум 2 тикера")

    weights = portfolio_weights or _normalize_weights({t: 1.0 for t in portfolio_tickers})
    portfolio_ret = np.zeros((n_paths, horizon), dtype=float)
    for symbol in portfolio_tickers:
        stock_artifact = get_stock_model(symbol, regime)
        stock_raw = _linear_predict(
            stock_artifact,
            {
                "key_rate": key_rate,
                "usd_rub": usd,
                "brent_usd": oil,
                "inflation": infl,
                "imoex_close": imoex_raw,
            },
        )
        current = float(get_stock_data(symbol)[f"{symbol.lower()}_close"].dropna().iloc[-1])
        stock_ret = stock_raw / current - 1.0
        portfolio_ret += float(weights.get(symbol, 0.0)) * stock_ret

    return 100.0 * (1.0 + portfolio_ret) * mult


def _trajectory_figure(
    current_value: float,
    forecast_paths: np.ndarray,
    horizon_months: int,
    asset_label: str,
) -> go.Figure:
    p10 = np.percentile(forecast_paths, 10, axis=0)
    p50 = np.percentile(forecast_paths, 50, axis=0)
    p90 = np.percentile(forecast_paths, 90, axis=0)

    future_dates = pd.date_range(start=pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0), periods=horizon_months + 1, freq="ME")

    median_line = np.concatenate([[current_value], p50])
    low_line = np.concatenate([[current_value], p10])
    high_line = np.concatenate([[current_value], p90])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=high_line,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=low_line,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(width=0),
            name="Диапазон P10-P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=median_line,
            mode="lines+markers",
            name="Медианный прогноз",
            line=dict(width=3, color="#1f77b4"),
        )
    )

    fig.update_layout(
        title=f"Динамический прогноз {asset_label} на {horizon_months} мес.",
        xaxis_title="Дата",
        yaxis_title=f"Значение {asset_label}",
        template="plotly_white",
    )
    return fig


def _timeline_seed_frame(controls: dict[str, float], horizon_months: int) -> pd.DataFrame:
    dates = pd.date_range(
        start=pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(1),
        periods=horizon_months,
        freq="ME",
    )
    return pd.DataFrame(
        {
            "date": dates,
            "oil": [float(controls["oil"])] * horizon_months,
            "key_rate": [float(controls["key_rate"])] * horizon_months,
            "usd_rub": [float(controls["usd_rub"])] * horizon_months,
            "inflation": [float(controls["inflation"])] * horizon_months,
        }
    )


def _timeline_path_figure(
    current_value: float,
    timeline_df: pd.DataFrame,
    predictions: np.ndarray,
    asset_label: str,
) -> go.Figure:
    dates = [pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0), *pd.to_datetime(timeline_df["date"]).tolist()]
    values = [float(current_value), *predictions.tolist()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            line=dict(width=3, color="#ff7f0e"),
            name="Траектория по таймлайну",
        )
    )
    fig.update_layout(
        title=f"Ручной таймлайн-прогноз {asset_label}",
        xaxis_title="Дата",
        yaxis_title=f"Значение {asset_label}",
        template="plotly_white",
    )
    return fig


def _scenario_comparison_seed(defaults: dict[str, float], controls: dict[str, float]) -> pd.DataFrame:
    presets = _preset_values(defaults)
    rows = [
        {
            "Сценарий": "Текущий",
            "Нефть": controls["oil"],
            "Ставка": controls["key_rate"],
            "USD/RUB": controls["usd_rub"],
            "Инфляция": controls["inflation"],
            "Сентимент": controls["market_sentiment"],
            "Ликвидность": controls["liquidity_effect"],
            "Геополитика": controls["geopolitics_effect"],
            "Регуляторика": controls["regulatory_effect"],
            "Волатильность": controls["uncertainty_scale"],
        },
        {
            "Сценарий": "Оптимистичный",
            "Нефть": presets["optimistic"]["oil"],
            "Ставка": presets["optimistic"]["key_rate"],
            "USD/RUB": presets["optimistic"]["usd_rub"],
            "Инфляция": presets["optimistic"]["inflation"],
            "Сентимент": presets["optimistic"]["market_sentiment"],
            "Ликвидность": presets["optimistic"]["liquidity_effect"],
            "Геополитика": presets["optimistic"]["geopolitics_effect"],
            "Регуляторика": presets["optimistic"]["regulatory_effect"],
            "Волатильность": presets["optimistic"]["uncertainty_scale"],
        },
        {
            "Сценарий": "Пессимистичный",
            "Нефть": presets["pessimistic"]["oil"],
            "Ставка": presets["pessimistic"]["key_rate"],
            "USD/RUB": presets["pessimistic"]["usd_rub"],
            "Инфляция": presets["pessimistic"]["inflation"],
            "Сентимент": presets["pessimistic"]["market_sentiment"],
            "Ликвидность": presets["pessimistic"]["liquidity_effect"],
            "Геополитика": presets["pessimistic"]["geopolitics_effect"],
            "Регуляторика": presets["pessimistic"]["regulatory_effect"],
            "Волатильность": presets["pessimistic"]["uncertainty_scale"],
        },
    ]
    return pd.DataFrame(rows)


def _scenario_comparison_from_editor(
    scenarios_df: pd.DataFrame,
    asset_type: str,
    ticker: str | None,
    regime: str,
    current: float,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for idx, row in scenarios_df.iterrows():
        def _num(value: Any, default: float) -> float:
            parsed = pd.to_numeric(value, errors="coerce")
            return float(default) if pd.isna(parsed) else float(parsed)

        name = str(row.get("Сценарий", f"Сценарий {idx + 1}")).strip() or f"Сценарий {idx + 1}"
        oil = _clamp(_num(row.get("Нефть", 80.0), 80.0), BASE_SLIDER_CONFIG["oil"])
        key_rate = _clamp(_num(row.get("Ставка", 10.0), 10.0), BASE_SLIDER_CONFIG["key_rate"])
        usd_rub = _clamp(_num(row.get("USD/RUB", 90.0), 90.0), BASE_SLIDER_CONFIG["usd_rub"])
        inflation = _clamp(_num(row.get("Инфляция", 6.0), 6.0), BASE_SLIDER_CONFIG["inflation"])
        uncertainty = _clamp(_num(row.get("Волатильность", 1.0), 1.0), ADDITIONAL_SLIDER_CONFIG["uncertainty_scale"])

        controls_sc = {
            "oil": oil,
            "key_rate": key_rate,
            "usd_rub": usd_rub,
            "inflation": inflation,
            "uncertainty_scale": uncertainty,
        }
        adjustments_sc = {
            "market_sentiment": _clamp(_num(row.get("Сентимент", 0.0), 0.0), ADDITIONAL_SLIDER_CONFIG["market_sentiment"]),
            "liquidity_effect": _clamp(_num(row.get("Ликвидность", 0.0), 0.0), ADDITIONAL_SLIDER_CONFIG["liquidity_effect"]),
            "geopolitics_effect": _clamp(_num(row.get("Геополитика", 0.0), 0.0), ADDITIONAL_SLIDER_CONFIG["geopolitics_effect"]),
            "regulatory_effect": _clamp(_num(row.get("Регуляторика", 0.0), 0.0), ADDITIONAL_SLIDER_CONFIG["regulatory_effect"]),
        }

        pred, raw, _, _, _ = _predict_asset_snapshot(
            asset_type=asset_type,
            ticker=ticker,
            controls=controls_sc,
            adjustments=adjustments_sc,
            regime=regime,
            portfolio_tickers=portfolio_tickers,
            portfolio_weights=portfolio_weights,
        )

        stress = _scenario_stress_index({**controls_sc, **adjustments_sc}, adjustments_sc)

        rows.append(
            {
                "Сценарий": name,
                "Базовый прогноз": round(raw, 2),
                "Итоговый прогноз": round(pred, 2),
                "Δ к текущему": round(pred - current, 2),
                "Δ %": round((pred / current - 1.0) * 100.0 if current else 0.0, 2),
                "Индекс стресса": round(stress, 1),
            }
        )

    return pd.DataFrame(rows)


def _simulation_context_key(
    asset_type: str,
    ticker: str | None,
    regime: str,
    controls: dict[str, float],
    adjustments: dict[str, float],
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float],
) -> tuple[Any, ...]:
    weights = tuple(round(float(portfolio_weights.get(t, 0.0)), 6) for t in portfolio_tickers)
    return (
        asset_type,
        ticker or "",
        regime,
        tuple(portfolio_tickers),
        weights,
        round(float(controls["oil"]), 6),
        round(float(controls["key_rate"]), 6),
        round(float(controls["usd_rub"]), 6),
        round(float(controls["inflation"]), 6),
        round(float(controls["uncertainty_scale"]), 6),
        round(float(adjustments["market_sentiment"]), 6),
        round(float(adjustments["liquidity_effect"]), 6),
        round(float(adjustments["geopolitics_effect"]), 6),
        round(float(adjustments["regulatory_effect"]), 6),
    )


def _resize_timeline_editor_source(source: Any, controls: dict[str, float], row_count: int) -> pd.DataFrame:
    rows = max(int(row_count), 1)
    out = _timeline_seed_frame(controls, rows)
    if not isinstance(source, pd.DataFrame) or source.empty:
        return out

    copy_rows = min(len(source), rows)
    for col in ("oil", "key_rate", "usd_rub", "inflation"):
        if col in source.columns:
            parsed = pd.to_numeric(source[col].iloc[:copy_rows], errors="coerce")
            out.loc[: copy_rows - 1, col] = parsed.to_numpy()
    return out


def _resize_compare_editor_source(
    source: Any,
    defaults: dict[str, float],
    controls: dict[str, float],
    row_count: int,
) -> pd.DataFrame:
    rows = min(max(int(row_count), 2), 8)
    template = _scenario_comparison_seed(defaults, controls)
    expected_cols = list(template.columns)

    seed_rows: list[dict[str, Any]] = []
    for idx in range(rows):
        if idx < len(template):
            seed_rows.append(template.iloc[idx].to_dict())
        else:
            extra = template.iloc[0].to_dict()
            extra["Сценарий"] = f"Сценарий {idx + 1}"
            seed_rows.append(extra)
    out = pd.DataFrame(seed_rows, columns=expected_cols)

    if not isinstance(source, pd.DataFrame) or source.empty:
        return out

    copy_rows = min(len(source), rows)
    for col in expected_cols:
        if col in source.columns:
            out.loc[: copy_rows - 1, col] = source[col].iloc[:copy_rows].to_numpy()
    return out[expected_cols]


def _plotly_selected_points(selection: Any) -> list[dict[str, Any]]:
    if selection is None:
        return []

    candidates: list[Any] = []
    if isinstance(selection, dict):
        candidates.append(selection)
    else:
        sel_attr = getattr(selection, "selection", None)
        if sel_attr is not None:
            candidates.append(sel_attr)

        to_dict = getattr(selection, "to_dict", None)
        if callable(to_dict):
            try:
                candidates.append(to_dict())
            except Exception:
                pass

        try:
            candidates.append(dict(selection))
        except Exception:
            pass

    for candidate in candidates:
        if hasattr(candidate, "to_dict"):
            try:
                candidate = candidate.to_dict()
            except Exception:
                continue

        if isinstance(candidate, dict):
            points = candidate.get("points", [])
            if isinstance(points, list):
                normalized: list[dict[str, Any]] = []
                for p in points:
                    if isinstance(p, dict):
                        normalized.append(p)
                        continue
                    to_dict = getattr(p, "to_dict", None)
                    if callable(to_dict):
                        try:
                            pdict = to_dict()
                            if isinstance(pdict, dict):
                                normalized.append(pdict)
                                continue
                        except Exception:
                            pass
                    try:
                        normalized.append(dict(p))
                        continue
                    except Exception:
                        pass
                    pvars = vars(p) if hasattr(p, "__dict__") else None
                    if isinstance(pvars, dict):
                        normalized.append(pvars)
                return normalized

    return []


def _selected_point_index(points: list[dict[str, Any]]) -> int | None:
    if not points:
        return None
    point = points[0]
    for key in ("point_index", "pointNumber", "point_number", "pointIndex"):
        raw = point.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return None


def _render_monte_carlo_bin_details(mc_result: dict[str, Any], selection: Any, detail_key_prefix: str) -> None:
    bins = mc_result.get("hist_bins")
    if not isinstance(bins, pd.DataFrame) or bins.empty:
        return

    points = _plotly_selected_points(selection)
    clicked_idx = _selected_point_index(points)
    if clicked_idx is not None and (clicked_idx < 0 or clicked_idx >= len(bins)):
        clicked_idx = None

    labels = [
        f"{int(row['bin_index']) + 1}: {float(row['left']):.1f} .. {float(row['right']):.1f}"
        for _, row in bins.iterrows()
    ]
    default_idx = int(np.argmax(bins["count"].to_numpy())) if len(labels) else 0
    select_key = f"{detail_key_prefix}_bin_select"
    if labels:
        if select_key not in st.session_state:
            st.session_state[select_key] = labels[default_idx]
        if clicked_idx is not None:
            st.session_state[select_key] = labels[clicked_idx]

    st.markdown("**Детализация столбца Monte Carlo**")
    st.caption("Кликните на столбец гистограммы или выберите диапазон вручную.")
    selected_label = st.selectbox(
        "Диапазон прогноза (бин)",
        labels,
        key=select_key,
    )
    selected_idx = labels.index(selected_label)
    row = bins.iloc[selected_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Диапазон", f"{float(row['left']):.1f} .. {float(row['right']):.1f}")
    c2.metric("Сценариев в бине", f"{int(row['count'])}")
    c3.metric("Вероятность бина", f"{float(row['probability']):.2%}")
    c4.metric("Средний прогноз в бине", f"{float(row['prediction_mean']):.1f}")

    detail_df = pd.DataFrame(
        [
            {"Параметр": "Нефть Brent", "Среднее в бине": float(row["oil_mean"])},
            {"Параметр": "Ключевая ставка", "Среднее в бине": float(row["key_rate_mean"])},
            {"Параметр": "USD/RUB", "Среднее в бине": float(row["usd_rub_mean"])},
            {"Параметр": "Инфляция", "Среднее в бине": float(row["inflation_mean"])},
        ]
    )
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    base = mc_result.get("base_params", {})
    std_map = mc_result.get("std_map", {})
    st.markdown(
        f"""
        **Как посчитан выбранный столбец**
        - Для каждого макрофактора генерируются сценарии по нормальному закону:
          `X ~ N(μ, σ)`, где `μ` — базовое значение, `σ` — историческая волатильность.
        - Затем модель считает прогноз актива для каждого сценария, после чего применяется суммарная корректировка сценария.
        - Текущий столбец агрегирует прогнозы, попавшие в интервал **[{float(row['left']):.1f}; {float(row['right']):.1f})**.
        - В нём **{int(row['count'])}** сценариев из **{int(mc_result.get('n_simulations', 0))}**.
        """
    )
    st.caption(
        "Базовые μ: "
        f"oil={float(base.get('oil', np.nan)):.2f}, "
        f"key_rate={float(base.get('key_rate', np.nan)):.2f}, "
        f"usd_rub={float(base.get('usd_rub', np.nan)):.2f}, "
        f"inflation={float(base.get('inflation', np.nan)):.2f}"
    )
    st.caption(
        "Использованные σ: "
        f"oil={float(std_map.get('oil', np.nan)):.2f}, "
        f"key_rate={float(std_map.get('key_rate', np.nan)):.2f}, "
        f"usd_rub={float(std_map.get('usd_rub', np.nan)):.2f}, "
        f"inflation={float(std_map.get('inflation', np.nan)):.2f}"
    )


def _ensure_mc_detail_payload(mc_result: dict[str, Any], controls: dict[str, float]) -> dict[str, Any]:
    if not isinstance(mc_result, dict):
        return mc_result

    bins = mc_result.get("hist_bins")
    if isinstance(bins, pd.DataFrame) and not bins.empty:
        return mc_result

    results_raw = mc_result.get("results")
    try:
        results = np.asarray(results_raw, dtype=float)
    except Exception:
        return mc_result
    if results.size == 0:
        return mc_result

    counts, edges = np.histogram(results, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = len(results)
    fallback_bins = pd.DataFrame(
        {
            "bin_index": np.arange(len(counts), dtype=int),
            "left": edges[:-1],
            "right": edges[1:],
            "center": centers,
            "count": counts.astype(int),
            "probability": counts / total if total else 0.0,
            "oil_mean": np.nan,
            "key_rate_mean": np.nan,
            "usd_rub_mean": np.nan,
            "inflation_mean": np.nan,
            "prediction_mean": np.nan,
            "prediction_std": np.nan,
        }
    )

    out = dict(mc_result)
    out["hist_bins"] = fallback_bins
    if "base_params" not in out:
        out["base_params"] = {
            "oil": float(controls["oil"]),
            "key_rate": float(controls["key_rate"]),
            "usd_rub": float(controls["usd_rub"]),
            "inflation": float(controls["inflation"]),
        }
    if "n_simulations" not in out:
        out["n_simulations"] = int(total)
    if "std_map" not in out:
        out["std_map"] = {}
    return out


def _render_sobol_factor_details(sobol_result: dict[str, Any], selection: Any, detail_key_prefix: str) -> None:
    sobol_df = sobol_result.get("sobol_df")
    problem = sobol_result.get("problem", {})
    if not isinstance(sobol_df, pd.DataFrame) or sobol_df.empty:
        return

    points = _plotly_selected_points(selection)
    clicked_idx = _selected_point_index(points)
    if clicked_idx is not None and (clicked_idx < 0 or clicked_idx >= len(sobol_df)):
        clicked_idx = None

    labels = sobol_df["factor_label"].astype(str).tolist()
    select_key = f"{detail_key_prefix}_factor_select"
    if labels:
        if select_key not in st.session_state:
            st.session_state[select_key] = labels[0]
        if clicked_idx is not None:
            st.session_state[select_key] = labels[clicked_idx]

    st.markdown("**Детализация выбранного столбца Sobol**")
    st.caption("Кликните на столбец Tornado Chart или выберите фактор вручную.")
    selected_label = st.selectbox("Фактор", labels, key=select_key)
    selected_idx = labels.index(selected_label)
    row = sobol_df.iloc[selected_idx]

    factor = str(row["factor"])
    bounds_map = {
        str(name): (float(bound[0]), float(bound[1]))
        for name, bound in zip(problem.get("names", []), problem.get("bounds", []))
    }
    low, high = bounds_map.get(factor, (float("nan"), float("nan")))
    sign_text = "позитивное" if int(row.get("impact_sign", 1)) > 0 else "негативное"
    s1 = float(row["S1"])
    s1_conf = float(row.get("S1_conf", np.nan))

    d = len(problem.get("names", []))
    n_samples = int(sobol_result.get("n_samples", 0))
    n_evaluations = int(sobol_result.get("n_evaluations", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S1", f"{s1:.3f}")
    c2.metric("S1 ± conf", f"{s1_conf:.3f}")
    c3.metric("Диапазон фактора", f"{low:.2f} .. {high:.2f}")
    c4.metric("Направление", sign_text)

    st.markdown(
        f"""
        **Как посчитан вклад фактора `{selected_label}`**
        - Используется разложение Соболя первого порядка:
          `S1ᵢ = Var(E[Y|Xᵢ]) / Var(Y)`.
        - Значение `S1={s1:.3f}` означает долю вариации прогноза, объясняемую только этим фактором.
        - Для расчета берутся Saltelli-сэмплы по диапазону **[{low:.2f}; {high:.2f}]**.
        - В этом запуске: `D={d}` факторов, `N={n_samples}` базовых сэмплов, фактических прогонов модели: **{n_evaluations}**.
        """
    )


def main() -> None:
    st.title("Симулятор экономических сценариев для рынка РФ")

    try:
        df = get_data()
    except Exception as exc:
        st.error(f"Не удалось загрузить данные: {exc}")
        st.stop()

    defaults = _defaults_from_data(df)
    (
        controls,
        asset_type,
        ticker,
        regime,
        horizon_months,
        n_paths,
        portfolio_tickers,
        portfolio_weights,
    ) = _render_sidebar_controls(defaults)

    cpu_saver = st.sidebar.toggle(
        "Экономный режим CPU",
        value=True,
        help="Уменьшает размер тяжёлых расчётов (веер, Monte Carlo, Sobol) для более плавной работы UI.",
    )
    effective_n_paths = min(int(n_paths), 200) if cpu_saver else int(n_paths)
    if cpu_saver and effective_n_paths < int(n_paths):
        st.sidebar.caption(f"Для веерного прогноза используется {effective_n_paths} траекторий вместо {n_paths}.")

    adjustments = _adjustments_from_controls(controls)
    sim_context_key = _simulation_context_key(
        asset_type=asset_type,
        ticker=ticker,
        regime=regime,
        controls=controls,
        adjustments=adjustments,
        portfolio_tickers=portfolio_tickers,
        portfolio_weights=portfolio_weights,
    )
    context_changed = st.session_state.get("sim_context_key") != sim_context_key
    if context_changed:
        st.session_state["sim_context_key"] = sim_context_key

    plot_df = _build_plot_df(df, asset_type, ticker, portfolio_tickers, portfolio_weights)

    tabs = st.tabs([
        "📊 Исторические данные",
        "🎯 Сценарный анализ",
        "🎲 Монте-Карло",
        "🌪️ Чувствительность",
    ])

    with tabs[0]:
        st.markdown(
            """
            **Описание вкладки**
            Здесь показана историческая динамика IMOEX, макрофакторов и выбранного актива (акция/портфель).
            График помогает увидеть тренды, а корреляционная матрица — силу линейной связи между факторами.
            """
        )
        st.plotly_chart(_historical_chart(plot_df, asset_type, ticker), use_container_width=True)
        corr_fig = _correlation_heatmap(plot_df, asset_type, ticker)
        corr_cols = _correlation_columns(plot_df, asset_type, ticker)
        labels = _factor_label_map(ticker)
        st.plotly_chart(corr_fig, use_container_width=True)
        st.caption("Выбор пары для drilldown делается через селекторы ниже (стабильный режим).")

        with st.expander("Как читать корреляционную матрицу"):
            st.markdown(
                """
                - Значение корреляции находится в диапазоне от `-1` до `+1`.
                - Ближе к `+1`: факторы чаще движутся в одном направлении.
                - Ближе к `-1`: факторы чаще движутся в противоположных направлениях.
                - Ближе к `0`: явной линейной связи почти нет.
                - Корреляция не доказывает причинность.
                """
            )

        st.markdown("**Drilldown по паре факторов**")
        if len(corr_cols) >= 2:
            pair = st.session_state.get("corr_pair", (corr_cols[0], corr_cols[1]))
            x_default = pair[0] if pair[0] in corr_cols else corr_cols[0]
            y_pool = [c for c in corr_cols if c != x_default]
            if not y_pool:
                y_pool = corr_cols
            y_default = pair[1] if pair[1] in y_pool else y_pool[0]

            c_left, c_right = st.columns(2)
            x_selected = c_left.selectbox(
                "Фактор X",
                corr_cols,
                index=corr_cols.index(x_default),
                format_func=lambda c: labels.get(c, c),
            )
            y_options = [c for c in corr_cols if c != x_selected] or corr_cols
            y_selected = c_right.selectbox(
                "Фактор Y",
                y_options,
                index=y_options.index(y_default if y_default in y_options else y_options[0]),
                format_func=lambda c: labels.get(c, c),
            )
            st.session_state["corr_pair"] = (x_selected, y_selected)

            st.plotly_chart(
                _correlation_drilldown_figure(plot_df, x_selected, y_selected, ticker),
                use_container_width=True,
            )

        insights = _correlation_insights(plot_df, asset_type, ticker)
        if insights:
            st.markdown("**Автоинсайты по корреляциям (топ-3):**")
            for line in insights:
                st.write(f"- {line}")

    with tabs[1]:
        st.markdown(
            """
            **Описание вкладки**
            Прогноз рассчитывается по выбранному режиму модели (`весь период`, `до 2022`, `после 2022`).
            Сначала формируется базовый прогноз по макропараметрам, затем применяются дополнительные поправки сценария.
            Ниже показан веерный прогноз на 6-12 месяцев, чтобы оценить возможную динамику, а не только одну точку.
            """
        )

        try:
            prediction, raw_prediction, asset_label, model_artifact, stock_pred_map = _predict_asset_snapshot(
                asset_type=asset_type,
                ticker=ticker,
                controls=controls,
                adjustments=adjustments,
                regime=regime,
                portfolio_tickers=portfolio_tickers,
                portfolio_weights=portfolio_weights,
            )
            current = _current_asset_value(
                df,
                asset_type=asset_type,
                ticker=ticker,
                portfolio_tickers=portfolio_tickers,
                portfolio_weights=portfolio_weights,
            )
        except Exception as exc:
            st.error(f"Ошибка сценарного прогноза: {exc}")
            st.stop()

        delta_abs = prediction - current
        delta_pct = delta_abs / current if current else 0.0
        extra_effect = prediction - raw_prediction
        stress_index = _scenario_stress_index(controls, adjustments)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Текущее значение {asset_label}", f"{current:,.2f}")
        m2.metric(f"Базовый прогноз {asset_label}", f"{raw_prediction:,.2f}")
        m3.metric(f"Итоговый прогноз {asset_label}", f"{prediction:,.2f}", delta=f"{delta_abs:,.2f}")
        m4.metric("Эффект доп. факторов", f"{extra_effect:,.2f}", delta=f"{delta_pct:.2%}")

        st.markdown("**Индекс стресса сценария**")
        st.progress(int(stress_index))
        if stress_index >= 70:
            st.warning(f"Сценарий стрессовый: {stress_index:.0f}/100")
        elif stress_index >= 40:
            st.info(f"Сценарий умеренно напряженный: {stress_index:.0f}/100")
        else:
            st.success(f"Сценарий ближе к базовому: {stress_index:.0f}/100")

        st.plotly_chart(_gauge_chart(prediction, current, asset_label=asset_label), use_container_width=True)

        raw_imoex = predict_scenario(
            oil=controls["oil"],
            key_rate=controls["key_rate"],
            usd_rub=controls["usd_rub"],
            inflation=controls["inflation"],
            regime=regime,
        )
        comment = _driver_comment(
            asset_type=asset_type,
            ticker=ticker,
            controls=controls,
            adjustments=adjustments,
            regime=regime,
            raw_imoex=raw_imoex,
            stock_pred_map=stock_pred_map,
            portfolio_tickers=portfolio_tickers,
        )
        st.info(f"**Что двигает результат:** {comment}")

        if model_artifact is not None:
            st.caption(
                f"Метрики модели ({regime}): "
                f"R²={model_artifact['metrics']['r2']:.3f}, "
                f"MAE={model_artifact['metrics']['mae']:.2f}, "
                f"RMSE={model_artifact['metrics']['rmse']:.2f}"
            )
        else:
            st.caption("Для портфеля используется агрегирование прогнозов по акциям и весам.")

        st.markdown("**Динамический прогноз будущей траектории**")
        try:
            macro_paths = _simulate_macro_paths(
                controls,
                horizon_months=horizon_months,
                n_paths=effective_n_paths,
                df=df,
            )
            forecast_paths = _predict_paths(
                asset_type=asset_type,
                ticker=ticker,
                paths=macro_paths,
                adjustments=adjustments,
                regime=regime,
                portfolio_tickers=portfolio_tickers,
                portfolio_weights=portfolio_weights,
            )
            st.plotly_chart(
                _trajectory_figure(current, forecast_paths, horizon_months=horizon_months, asset_label=asset_label),
                use_container_width=True,
                key=f"fan_{asset_type}_{ticker or 'imoex'}_{regime}_{horizon_months}_{effective_n_paths}",
            )
            st.caption(f"Размер веера: {effective_n_paths} траекторий.")
        except Exception as exc:
            st.warning(f"Не удалось построить траектории прогноза: {exc}")

        st.markdown("**Интерактивный таймлайн сценария (ручной ввод по месяцам)**")
        timeline_seed_key = f"timeline_seed_{asset_type}_{ticker or 'imoex'}_{regime}_{horizon_months}"
        timeline_rows_key = f"timeline_rows_{asset_type}_{ticker or 'imoex'}_{regime}_v3"
        timeline_rev_key = f"timeline_rev_{asset_type}_{ticker or 'imoex'}_{regime}_v3"
        timeline_horizon_key = f"timeline_last_h_{asset_type}_{ticker or 'imoex'}_{regime}_v3"

        if timeline_rows_key not in st.session_state:
            st.session_state[timeline_rows_key] = int(horizon_months)
        if timeline_seed_key not in st.session_state:
            st.session_state[timeline_seed_key] = _timeline_seed_frame(controls, horizon_months)
        if timeline_rev_key not in st.session_state:
            st.session_state[timeline_rev_key] = 0

        last_horizon = st.session_state.get(timeline_horizon_key)
        if last_horizon is None:
            st.session_state[timeline_horizon_key] = int(horizon_months)
        elif int(last_horizon) != int(horizon_months):
            st.session_state[timeline_rows_key] = int(horizon_months)
            st.session_state[timeline_seed_key] = _timeline_seed_frame(controls, horizon_months)
            st.session_state[timeline_horizon_key] = int(horizon_months)
            st.session_state[timeline_rev_key] += 1

        r1, r2, r3 = st.columns([1.1, 1.4, 1.8])
        if r1.button("Сбросить таймлайн под текущие параметры", key=f"reset_{timeline_seed_key}"):
            st.session_state[timeline_seed_key] = _timeline_seed_frame(controls, horizon_months)
            st.session_state[timeline_rows_key] = int(horizon_months)
            st.session_state[timeline_rev_key] += 1
        if r2.button("Синхронизировать строки с горизонтом", key=f"sync_rows_{timeline_seed_key}"):
            st.session_state[timeline_rows_key] = int(horizon_months)
            st.session_state[timeline_rev_key] += 1
        r3.number_input("Строк в таймлайне", min_value=1, max_value=24, step=1, key=timeline_rows_key)

        timeline_row_count = int(st.session_state[timeline_rows_key])
        timeline_df = _resize_timeline_editor_source(
            st.session_state.get(timeline_seed_key),
            controls=controls,
            row_count=timeline_row_count,
        )

        st.caption("Стабильный режим редактирования: выберите строку и измените параметры ниже.")
        st.dataframe(
            timeline_df[["date", "oil", "key_rate", "usd_rub", "inflation"]],
            hide_index=True,
            use_container_width=True,
        )

        if len(timeline_df) > 0:
            row_labels = [
                f"{idx + 1} — {pd.to_datetime(timeline_df.loc[idx, 'date']).strftime('%b %Y')}"
                for idx in range(len(timeline_df))
            ]
            selector_key = f"timeline_row_selector_{timeline_seed_key}_{st.session_state[timeline_rev_key]}"
            selected_row_label = st.selectbox("Редактируемая строка таймлайна", row_labels, key=selector_key)
            selected_idx = row_labels.index(selected_row_label)
            row = timeline_df.iloc[selected_idx]

            edit_ns = f"timeline_edit_{timeline_seed_key}_{st.session_state[timeline_rev_key]}_{selected_idx}"
            e1, e2, e3, e4 = st.columns(4)
            timeline_df.loc[selected_idx, "oil"] = _clamp(
                e1.number_input(
                    "Нефть",
                    min_value=float(BASE_SLIDER_CONFIG["oil"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["oil"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["oil"]["step"]),
                    value=float(row["oil"]),
                    key=f"{edit_ns}_oil",
                ),
                BASE_SLIDER_CONFIG["oil"],
            )
            timeline_df.loc[selected_idx, "key_rate"] = _clamp(
                e2.number_input(
                    "Ставка",
                    min_value=float(BASE_SLIDER_CONFIG["key_rate"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["key_rate"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["key_rate"]["step"]),
                    value=float(row["key_rate"]),
                    key=f"{edit_ns}_key_rate",
                ),
                BASE_SLIDER_CONFIG["key_rate"],
            )
            timeline_df.loc[selected_idx, "usd_rub"] = _clamp(
                e3.number_input(
                    "USD/RUB",
                    min_value=float(BASE_SLIDER_CONFIG["usd_rub"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["usd_rub"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["usd_rub"]["step"]),
                    value=float(row["usd_rub"]),
                    key=f"{edit_ns}_usd_rub",
                ),
                BASE_SLIDER_CONFIG["usd_rub"],
            )
            timeline_df.loc[selected_idx, "inflation"] = _clamp(
                e4.number_input(
                    "Инфляция",
                    min_value=float(BASE_SLIDER_CONFIG["inflation"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["inflation"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["inflation"]["step"]),
                    value=float(row["inflation"]),
                    key=f"{edit_ns}_inflation",
                ),
                BASE_SLIDER_CONFIG["inflation"],
            )

        st.session_state[timeline_seed_key] = _resize_timeline_editor_source(
            timeline_df,
            controls=controls,
            row_count=timeline_row_count,
        )

        try:
            tl = st.session_state[timeline_seed_key].copy()
            timeline_rows = len(tl)
            tl["date"] = pd.date_range(
                start=pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(1),
                periods=timeline_rows,
                freq="ME",
            )
            for factor in ("oil", "key_rate", "usd_rub", "inflation"):
                cfg = BASE_SLIDER_CONFIG[factor]
                tl[factor] = pd.to_numeric(tl[factor], errors="coerce").fillna(float(controls[factor])).clip(cfg["min"], cfg["max"])

            timeline_predictions: list[float] = []
            for _, step in tl.iterrows():
                controls_step = dict(controls)
                controls_step["oil"] = float(step["oil"])
                controls_step["key_rate"] = float(step["key_rate"])
                controls_step["usd_rub"] = float(step["usd_rub"])
                controls_step["inflation"] = float(step["inflation"])
                pred_step, _, _, _, _ = _predict_asset_snapshot(
                    asset_type=asset_type,
                    ticker=ticker,
                    controls=controls_step,
                    adjustments=adjustments,
                    regime=regime,
                    portfolio_tickers=portfolio_tickers,
                    portfolio_weights=portfolio_weights,
                )
                timeline_predictions.append(float(pred_step))

            timeline_pred_arr = np.asarray(timeline_predictions, dtype=float)
            st.plotly_chart(
                _timeline_path_figure(current, tl, timeline_pred_arr, asset_label=asset_label),
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Не удалось рассчитать ручной таймлайн: {exc}")

        st.markdown("**Интерактивное сравнение 2-3 сценариев**")
        st.caption("Стабильный режим: таблица + редактирование выбранной строки.")
        compare_seed_key = f"compare_seed_{asset_type}_{ticker or 'imoex'}_{regime}"
        scenario_rows_key = f"scenario_rows_{asset_type}_{ticker or 'imoex'}_{regime}_v3"
        compare_rev_key = f"compare_rev_{asset_type}_{ticker or 'imoex'}_{regime}_v3"

        if scenario_rows_key not in st.session_state:
            st.session_state[scenario_rows_key] = 3
        if compare_seed_key not in st.session_state:
            st.session_state[compare_seed_key] = _scenario_comparison_seed(defaults, controls)
        if compare_rev_key not in st.session_state:
            st.session_state[compare_rev_key] = 0

        c1, c2, c3 = st.columns([1.1, 1.2, 1.4])
        if c1.button("Сбросить таблицу сценариев", key=f"reset_{compare_seed_key}"):
            st.session_state[compare_seed_key] = _scenario_comparison_seed(defaults, controls)
            st.session_state[scenario_rows_key] = 3
            st.session_state[compare_rev_key] += 1
        if c2.button("Синхронизировать строки с 3 базовыми сценариями", key=f"sync_{compare_seed_key}"):
            st.session_state[scenario_rows_key] = 3
            st.session_state[compare_rev_key] += 1
        c3.number_input("Строк в таблице сценариев", min_value=2, max_value=8, step=1, key=scenario_rows_key)

        scenario_row_count = int(st.session_state[scenario_rows_key])
        editor_df = _resize_compare_editor_source(
            st.session_state.get(compare_seed_key),
            defaults=defaults,
            controls=controls,
            row_count=scenario_row_count,
        )

        st.dataframe(editor_df, use_container_width=True, hide_index=True)

        if len(editor_df) > 0:
            row_labels = [f"{idx + 1} — {str(editor_df.loc[idx, 'Сценарий'])}" for idx in range(len(editor_df))]
            selector_key = f"compare_row_selector_{compare_seed_key}_{st.session_state[compare_rev_key]}"
            selected_row_label = st.selectbox("Редактируемая строка сценария", row_labels, key=selector_key)
            selected_idx = row_labels.index(selected_row_label)
            row = editor_df.iloc[selected_idx]
            edit_ns = f"compare_edit_{compare_seed_key}_{st.session_state[compare_rev_key]}_{selected_idx}"

            n1, n2, n3 = st.columns(3)
            editor_df.loc[selected_idx, "Сценарий"] = n1.text_input(
                "Название сценария",
                value=str(row["Сценарий"]),
                key=f"{edit_ns}_name",
            ).strip() or f"Сценарий {selected_idx + 1}"
            editor_df.loc[selected_idx, "Нефть"] = _clamp(
                n2.number_input(
                    "Нефть",
                    min_value=float(BASE_SLIDER_CONFIG["oil"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["oil"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["oil"]["step"]),
                    value=float(row["Нефть"]),
                    key=f"{edit_ns}_oil",
                ),
                BASE_SLIDER_CONFIG["oil"],
            )
            editor_df.loc[selected_idx, "Ставка"] = _clamp(
                n3.number_input(
                    "Ставка",
                    min_value=float(BASE_SLIDER_CONFIG["key_rate"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["key_rate"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["key_rate"]["step"]),
                    value=float(row["Ставка"]),
                    key=f"{edit_ns}_key_rate",
                ),
                BASE_SLIDER_CONFIG["key_rate"],
            )

            n4, n5, n6 = st.columns(3)
            editor_df.loc[selected_idx, "USD/RUB"] = _clamp(
                n4.number_input(
                    "USD/RUB",
                    min_value=float(BASE_SLIDER_CONFIG["usd_rub"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["usd_rub"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["usd_rub"]["step"]),
                    value=float(row["USD/RUB"]),
                    key=f"{edit_ns}_usd_rub",
                ),
                BASE_SLIDER_CONFIG["usd_rub"],
            )
            editor_df.loc[selected_idx, "Инфляция"] = _clamp(
                n5.number_input(
                    "Инфляция",
                    min_value=float(BASE_SLIDER_CONFIG["inflation"]["min"]),
                    max_value=float(BASE_SLIDER_CONFIG["inflation"]["max"]),
                    step=float(BASE_SLIDER_CONFIG["inflation"]["step"]),
                    value=float(row["Инфляция"]),
                    key=f"{edit_ns}_inflation",
                ),
                BASE_SLIDER_CONFIG["inflation"],
            )
            editor_df.loc[selected_idx, "Волатильность"] = _clamp(
                n6.number_input(
                    "Волатильность",
                    min_value=float(ADDITIONAL_SLIDER_CONFIG["uncertainty_scale"]["min"]),
                    max_value=float(ADDITIONAL_SLIDER_CONFIG["uncertainty_scale"]["max"]),
                    step=float(ADDITIONAL_SLIDER_CONFIG["uncertainty_scale"]["step"]),
                    value=float(row["Волатильность"]),
                    key=f"{edit_ns}_uncertainty_scale",
                ),
                ADDITIONAL_SLIDER_CONFIG["uncertainty_scale"],
            )

            n7, n8, n9 = st.columns(3)
            editor_df.loc[selected_idx, "Сентимент"] = _clamp(
                n7.number_input(
                    "Сентимент",
                    min_value=float(ADDITIONAL_SLIDER_CONFIG["market_sentiment"]["min"]),
                    max_value=float(ADDITIONAL_SLIDER_CONFIG["market_sentiment"]["max"]),
                    step=float(ADDITIONAL_SLIDER_CONFIG["market_sentiment"]["step"]),
                    value=float(row["Сентимент"]),
                    key=f"{edit_ns}_market_sentiment",
                ),
                ADDITIONAL_SLIDER_CONFIG["market_sentiment"],
            )
            editor_df.loc[selected_idx, "Ликвидность"] = _clamp(
                n8.number_input(
                    "Ликвидность",
                    min_value=float(ADDITIONAL_SLIDER_CONFIG["liquidity_effect"]["min"]),
                    max_value=float(ADDITIONAL_SLIDER_CONFIG["liquidity_effect"]["max"]),
                    step=float(ADDITIONAL_SLIDER_CONFIG["liquidity_effect"]["step"]),
                    value=float(row["Ликвидность"]),
                    key=f"{edit_ns}_liquidity_effect",
                ),
                ADDITIONAL_SLIDER_CONFIG["liquidity_effect"],
            )
            editor_df.loc[selected_idx, "Геополитика"] = _clamp(
                n9.number_input(
                    "Геополитика",
                    min_value=float(ADDITIONAL_SLIDER_CONFIG["geopolitics_effect"]["min"]),
                    max_value=float(ADDITIONAL_SLIDER_CONFIG["geopolitics_effect"]["max"]),
                    step=float(ADDITIONAL_SLIDER_CONFIG["geopolitics_effect"]["step"]),
                    value=float(row["Геополитика"]),
                    key=f"{edit_ns}_geopolitics_effect",
                ),
                ADDITIONAL_SLIDER_CONFIG["geopolitics_effect"],
            )

            reg_col = st.columns(1)[0]
            editor_df.loc[selected_idx, "Регуляторика"] = _clamp(
                reg_col.number_input(
                    "Регуляторика",
                    min_value=float(ADDITIONAL_SLIDER_CONFIG["regulatory_effect"]["min"]),
                    max_value=float(ADDITIONAL_SLIDER_CONFIG["regulatory_effect"]["max"]),
                    step=float(ADDITIONAL_SLIDER_CONFIG["regulatory_effect"]["step"]),
                    value=float(row["Регуляторика"]),
                    key=f"{edit_ns}_regulatory_effect",
                ),
                ADDITIONAL_SLIDER_CONFIG["regulatory_effect"],
            )

        st.session_state[compare_seed_key] = _resize_compare_editor_source(
            editor_df,
            defaults=defaults,
            controls=controls,
            row_count=scenario_row_count,
        )

        normalized_editor_df = st.session_state[compare_seed_key]

        if len(normalized_editor_df) < 2:
            st.info("Добавьте минимум 2 строки сценариев для сравнения.")
        else:
            try:
                cmp_df = _scenario_comparison_from_editor(
                    scenarios_df=normalized_editor_df,
                    asset_type=asset_type,
                    ticker=ticker,
                    regime=regime,
                    current=current,
                    portfolio_tickers=portfolio_tickers,
                    portfolio_weights=portfolio_weights,
                )
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                fig_cmp = go.Figure()
                fig_cmp.add_trace(
                    go.Bar(
                        x=cmp_df["Сценарий"],
                        y=cmp_df["Итоговый прогноз"],
                        marker_color=["#1f77b4", "#2ca02c", "#d62728"][: len(cmp_df)],
                        name="Итоговый прогноз",
                    )
                )
                fig_cmp.add_trace(
                    go.Scatter(
                        x=cmp_df["Сценарий"],
                        y=cmp_df["Δ %"],
                        mode="lines+markers",
                        yaxis="y2",
                        line=dict(color="#ff7f0e", width=2),
                        name="Δ %",
                    )
                )
                fig_cmp.update_layout(
                    title="Сравнение сценариев: уровень и относительное изменение",
                    xaxis_title="Сценарий",
                    yaxis_title=f"Прогноз {asset_label}",
                    yaxis2=dict(title="Δ %", overlaying="y", side="right"),
                    template="plotly_white",
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
            except Exception as exc:
                st.warning(f"Не удалось построить сравнение сценариев: {exc}")

    with tabs[2]:
        st.markdown(
            """
            **Описание вкладки**
            Monte Carlo генерирует 10 000 случайных сценариев вокруг введенных параметров.
            Выводятся квантили P5/P50/P95, стандартное отклонение и вероятность падения более чем на 20%.
            """
        )
        mc_result = st.session_state.get("mc_result")
        mc_context_key = st.session_state.get("mc_context_key")
        mc_stale = mc_result is not None and mc_context_key != sim_context_key
        if context_changed or mc_stale:
            st.info("Параметры изменились. Пересчитайте Monte Carlo для нового сценария.")

        if st.button("Запустить симуляцию Монте-Карло", key="run_mc"):
            progress = st.progress(0, text="Подготовка симуляции...")
            progress.progress(15, text="Загрузка параметров...")
            try:
                mc_runs = 5000 if cpu_saver else 10000
                result = run_monte_carlo(
                    {
                        "oil": controls["oil"],
                        "key_rate": controls["key_rate"],
                        "usd_rub": controls["usd_rub"],
                        "inflation": controls["inflation"],
                    },
                    n_simulations=mc_runs,
                    asset_type=asset_type,
                    ticker=ticker,
                    adjustments=adjustments,
                    uncertainty_scale=controls["uncertainty_scale"],
                    regime=regime,
                    portfolio_tickers=portfolio_tickers,
                    portfolio_weights=[portfolio_weights[t] for t in portfolio_tickers] if portfolio_weights else None,
                )
                st.session_state["mc_result"] = result
                st.session_state["mc_context_key"] = sim_context_key
                progress.progress(100, text="Готово")
            except Exception as exc:
                progress.empty()
                st.error(f"Ошибка Monte Carlo: {exc}")

        mc_result = st.session_state.get("mc_result")
        mc_context_key = st.session_state.get("mc_context_key")
        mc_stale = mc_result is not None and mc_context_key != sim_context_key
        if mc_result:
            mc_result = _ensure_mc_detail_payload(mc_result, controls=controls)
            st.session_state["mc_result"] = mc_result
            if mc_stale:
                st.warning("Показан результат Monte Carlo для предыдущих параметров. Нажмите кнопку для пересчета.")
            mc_chart_key = f"mc_hist_{asset_type}_{ticker or 'imoex'}_{regime}"
            mc_selection = st.plotly_chart(
                mc_result["figure"],
                use_container_width=True,
                key=mc_chart_key,
                on_select="rerun",
                selection_mode=("points",),
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P5", f"{mc_result['var_5']:.1f}")
            c2.metric("P50", f"{mc_result['p50']:.1f}")
            c3.metric("P95", f"{mc_result['var_95']:.1f}")
            c4.metric("Вероятность падения > 20%", f"{mc_result['prob_drop_20']:.2%}")

            st.caption(
                f"Актив: {mc_result['asset_label']} | "
                f"Корректировка: {mc_result['adjustment_pct']:+.2f}% | "
                f"Масштаб волатильности: {controls['uncertainty_scale']:.1f}x | "
                f"Режим: {regime} | "
                f"Симуляций: {5000 if cpu_saver else 10000}"
            )
            _render_monte_carlo_bin_details(
                mc_result,
                selection=mc_selection,
                detail_key_prefix=f"{mc_chart_key}_{len(mc_result.get('results', []))}",
            )
            if "hist_bins" in mc_result and isinstance(mc_result["hist_bins"], pd.DataFrame):
                hist_bins_df = mc_result["hist_bins"]
                if hist_bins_df["oil_mean"].isna().all():
                    st.caption("Это результат старого запуска. Для полной детализации по макропараметрам пересчитайте Monte Carlo.")

    with tabs[3]:
        st.markdown(
            """
            **Описание вкладки**
            Sobol-анализ измеряет вклад каждого макрофактора в разброс прогноза.
            Индекс S1 показывает индивидуальный вклад фактора: чем выше S1, тем сильнее влияние.
            """
        )
        sobol_result = st.session_state.get("sobol_result")
        sobol_context_key = st.session_state.get("sobol_context_key")
        sobol_stale = sobol_result is not None and sobol_context_key != sim_context_key
        if context_changed or sobol_stale:
            st.info("Параметры изменились. Пересчитайте Sobol-анализ для нового сценария.")

        if st.button("Рассчитать чувствительность Sobol", key="run_sobol"):
            with st.spinner("Выполняю анализ чувствительности..."):
                try:
                    sobol_samples = 256 if cpu_saver else 512
                    sobol_result = run_sobol_sensitivity(
                        n_samples=sobol_samples,
                        asset_type=asset_type,
                        ticker=ticker,
                        regime=regime,
                        portfolio_tickers=portfolio_tickers,
                        portfolio_weights=[portfolio_weights[t] for t in portfolio_tickers] if portfolio_weights else None,
                    )
                    st.session_state["sobol_result"] = sobol_result
                    st.session_state["sobol_context_key"] = sim_context_key
                except Exception as exc:
                    st.error(f"Ошибка Sobol-анализа: {exc}")

        sobol_result = st.session_state.get("sobol_result")
        sobol_context_key = st.session_state.get("sobol_context_key")
        sobol_stale = sobol_result is not None and sobol_context_key != sim_context_key
        if sobol_result:
            if sobol_stale:
                st.warning("Показан результат Sobol для предыдущих параметров. Нажмите кнопку для пересчета.")
            sobol_chart_key = f"sobol_{asset_type}_{ticker or 'imoex'}_{regime}"
            sobol_selection = st.plotly_chart(
                sobol_result["figure"],
                use_container_width=True,
                key=sobol_chart_key,
                on_select="rerun",
                selection_mode=("points",),
            )
            st.info(
                "Наибольшее влияние оказывает "
                f"`{sobol_result['top_factor']}` (S1={sobol_result['top_s1']:.2f})"
            )
            st.caption(
                "Дополнительные сценарные поправки (сентимент/ликвидность/геополитика/регуляторика) "
                "не входят в Sobol-разложение, так как применяются пост-коррекцией к прогнозу."
            )
            st.caption(f"Объем выборки Sobol: {256 if cpu_saver else 512}")
            _render_sobol_factor_details(
                sobol_result,
                selection=sobol_selection,
                detail_key_prefix=f"{sobol_chart_key}_{sobol_result.get('n_evaluations', 0)}",
            )


if __name__ == "__main__":
    main()
