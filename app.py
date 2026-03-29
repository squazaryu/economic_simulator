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


def _correlation_heatmap(df: pd.DataFrame, asset_type: str, ticker: str | None) -> go.Figure:
    cols = ["imoex_close", "key_rate", "usd_rub", "brent_usd", "inflation"]

    if asset_type == "stock" and ticker:
        stock_col = f"{ticker.lower()}_close"
        if stock_col in df.columns:
            cols.insert(0, stock_col)

    if asset_type == "portfolio" and "portfolio_index" in df.columns:
        cols.insert(0, "portfolio_index")

    corr = df[cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar=dict(title="Corr"),
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Корреляционная матрица", template="plotly_white")
    return fig


def _correlation_insights(df: pd.DataFrame, asset_type: str, ticker: str | None) -> list[str]:
    cols = ["imoex_close", "key_rate", "usd_rub", "brent_usd", "inflation"]

    if asset_type == "stock" and ticker:
        stock_col = f"{ticker.lower()}_close"
        if stock_col in df.columns:
            cols.insert(0, stock_col)

    if asset_type == "portfolio" and "portfolio_index" in df.columns:
        cols.insert(0, "portfolio_index")

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

    insights = []
    for left, right, value in strongest_abs:
        direction = "прямая" if value >= 0 else "обратная"
        strength = "сильная" if abs(value) >= 0.7 else ("умеренная" if abs(value) >= 0.4 else "слабая")
        insights.append(
            f"**{label_map.get(left, left)} ↔ {label_map.get(right, right)}**: "
            f"{value:+.2f} ({strength}, {direction} связь)"
        )
    return insights


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


def _scenario_comparison(
    defaults: dict[str, float],
    asset_type: str,
    ticker: str | None,
    regime: str,
    current: float,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float],
) -> pd.DataFrame:
    presets = _preset_values(defaults)
    rows: list[dict[str, Any]] = []

    for key, cfg in [("base", "Базовый"), ("optimistic", "Оптимистичный"), ("pessimistic", "Пессимистичный")]:
        sc = presets[key]
        controls_sc = {
            "oil": sc["oil"],
            "key_rate": sc["key_rate"],
            "usd_rub": sc["usd_rub"],
            "inflation": sc["inflation"],
            "uncertainty_scale": sc["uncertainty_scale"],
        }
        adjustments_sc = {
            "market_sentiment": sc["market_sentiment"],
            "liquidity_effect": sc["liquidity_effect"],
            "geopolitics_effect": sc["geopolitics_effect"],
            "regulatory_effect": sc["regulatory_effect"],
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
                "Сценарий": cfg,
                "Базовый прогноз": round(raw, 2),
                "Итоговый прогноз": round(pred, 2),
                "Δ к текущему": round(pred - current, 2),
                "Δ %": round((pred / current - 1.0) * 100.0 if current else 0.0, 2),
                "Индекс стресса": round(stress, 1),
            }
        )

    return pd.DataFrame(rows)


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

    adjustments = _adjustments_from_controls(controls)
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
        st.plotly_chart(_correlation_heatmap(plot_df, asset_type, ticker), use_container_width=True)

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
            macro_paths = _simulate_macro_paths(controls, horizon_months=horizon_months, n_paths=n_paths, df=df)
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
            )
        except Exception as exc:
            st.warning(f"Не удалось построить траектории прогноза: {exc}")

        st.markdown("**Сравнение сценариев (side-by-side)**")
        try:
            cmp_df = _scenario_comparison(
                defaults=defaults,
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
                    marker_color=["#1f77b4", "#2ca02c", "#d62728"],
                    name="Итоговый прогноз",
                )
            )
            fig_cmp.update_layout(
                title="Сравнение итогового прогноза по сценариям",
                xaxis_title="Сценарий",
                yaxis_title=f"Прогноз {asset_label}",
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

        if st.button("Запустить симуляцию Монте-Карло", key="run_mc"):
            progress = st.progress(0, text="Подготовка симуляции...")
            progress.progress(15, text="Загрузка параметров...")
            try:
                result = run_monte_carlo(
                    {
                        "oil": controls["oil"],
                        "key_rate": controls["key_rate"],
                        "usd_rub": controls["usd_rub"],
                        "inflation": controls["inflation"],
                    },
                    n_simulations=10000,
                    asset_type=asset_type,
                    ticker=ticker,
                    adjustments=adjustments,
                    uncertainty_scale=controls["uncertainty_scale"],
                    regime=regime,
                    portfolio_tickers=portfolio_tickers,
                    portfolio_weights=[portfolio_weights[t] for t in portfolio_tickers] if portfolio_weights else None,
                )
                st.session_state["mc_result"] = result
                progress.progress(100, text="Готово")
            except Exception as exc:
                progress.empty()
                st.error(f"Ошибка Monte Carlo: {exc}")

        mc_result = st.session_state.get("mc_result")
        if mc_result:
            st.plotly_chart(mc_result["figure"], use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P5", f"{mc_result['var_5']:.1f}")
            c2.metric("P50", f"{mc_result['p50']:.1f}")
            c3.metric("P95", f"{mc_result['var_95']:.1f}")
            c4.metric("Вероятность падения > 20%", f"{mc_result['prob_drop_20']:.2%}")

            st.caption(
                f"Актив: {mc_result['asset_label']} | "
                f"Корректировка: {mc_result['adjustment_pct']:+.2f}% | "
                f"Масштаб волатильности: {controls['uncertainty_scale']:.1f}x | "
                f"Режим: {regime}"
            )

    with tabs[3]:
        st.markdown(
            """
            **Описание вкладки**
            Sobol-анализ измеряет вклад каждого макрофактора в разброс прогноза.
            Индекс S1 показывает индивидуальный вклад фактора: чем выше S1, тем сильнее влияние.
            """
        )

        if st.button("Рассчитать чувствительность Sobol", key="run_sobol"):
            with st.spinner("Выполняю анализ чувствительности..."):
                try:
                    sobol_result = run_sobol_sensitivity(
                        n_samples=512,
                        asset_type=asset_type,
                        ticker=ticker,
                        regime=regime,
                        portfolio_tickers=portfolio_tickers,
                        portfolio_weights=[portfolio_weights[t] for t in portfolio_tickers] if portfolio_weights else None,
                    )
                    st.session_state["sobol_result"] = sobol_result
                except Exception as exc:
                    st.error(f"Ошибка Sobol-анализа: {exc}")

        sobol_result = st.session_state.get("sobol_result")
        if sobol_result:
            st.plotly_chart(sobol_result["figure"], use_container_width=True)
            st.info(
                "Наибольшее влияние оказывает "
                f"`{sobol_result['top_factor']}` (S1={sobol_result['top_s1']:.2f})"
            )
            st.caption(
                "Дополнительные сценарные поправки (сентимент/ликвидность/геополитика/регуляторика) "
                "не входят в Sobol-разложение, так как применяются пост-коррекцией к прогнозу."
            )


if __name__ == "__main__":
    main()
