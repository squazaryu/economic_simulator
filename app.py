"""Русифицированный Streamlit-дашборд экономического симулятора РФ."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_loader import load_moex_stock
from src.model import (
    apply_scenario_adjustments,
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
    "uncertainty_scale": {"label": "Масштаб волатильности (Monte Carlo)", "min": 0.5, "max": 2.0, "step": 0.1},
}

POPULAR_TICKERS = ["LKOH", "ROSN", "TATN", "SBER", "GAZP", "NVTK", "GMKN", "MOEX", "VTBR"]


def _clamp(value: float, config: dict[str, float]) -> float:
    return float(min(max(value, config["min"]), config["max"]))


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return build_processed_dataset()


@st.cache_data(show_spinner=False)
def get_stock_data(ticker: str) -> pd.DataFrame:
    return load_moex_stock(ticker)


@st.cache_resource(show_spinner=False)
def get_imoex_model() -> dict[str, Any]:
    return load_model_artifact()


@st.cache_resource(show_spinner=False)
def get_stock_model(ticker: str) -> dict[str, Any]:
    return load_stock_model_artifact(ticker)


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
        config = BASE_SLIDER_CONFIG.get(key) or ADDITIONAL_SLIDER_CONFIG.get(key)
        st.session_state[key] = _clamp(float(value), config) if config else float(value)


def _render_sidebar_controls(defaults: dict[str, float]) -> tuple[dict[str, float], str, str | None]:
    st.sidebar.header("Параметры сценария")

    if "oil" not in st.session_state:
        for key, value in defaults.items():
            st.session_state[key] = value

    asset_choice = st.sidebar.radio("Целевой актив", ["IMOEX", "Акция MOEX"], horizontal=False)
    ticker: str | None = None
    if asset_choice == "Акция MOEX":
        selected = st.sidebar.selectbox("Популярный тикер", POPULAR_TICKERS, index=0)
        custom = st.sidebar.text_input("Или введите тикер вручную", value=selected)
        ticker = (custom or selected).upper().strip()

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

    return controls, ("stock" if asset_choice == "Акция MOEX" else "imoex"), ticker


def _adjustments_from_controls(controls: dict[str, float]) -> dict[str, float]:
    return {
        "market_sentiment": controls["market_sentiment"],
        "liquidity_effect": controls["liquidity_effect"],
        "geopolitics_effect": controls["geopolitics_effect"],
        "regulatory_effect": controls["regulatory_effect"],
    }


def _build_plot_df(df: pd.DataFrame, ticker: str | None) -> pd.DataFrame:
    plot_df = df.copy()
    if ticker:
        stock_df = get_stock_data(ticker)
        stock_col = f"{ticker.lower()}_close"
        plot_df = plot_df.merge(stock_df[["date", stock_col]], on="date", how="left")
    return plot_df


def _historical_chart(df: pd.DataFrame, ticker: str | None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df["date"], y=df["imoex_close"], name="IMOEX", line=dict(width=3, color="#1f77b4")),
        secondary_y=False,
    )

    if ticker:
        stock_col = f"{ticker.lower()}_close"
        if stock_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[stock_col],
                    name=f"{ticker.upper()}",
                    line=dict(width=2, color="#ff7f0e"),
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


def _correlation_heatmap(df: pd.DataFrame, ticker: str | None) -> go.Figure:
    cols = ["imoex_close", "key_rate", "usd_rub", "brent_usd", "inflation"]
    if ticker:
        stock_col = f"{ticker.lower()}_close"
        if stock_col in df.columns:
            cols.insert(0, stock_col)

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


def _predict_asset(
    asset_type: str,
    ticker: str | None,
    controls: dict[str, float],
    adjustments: dict[str, float],
) -> tuple[float, float, str, dict[str, Any]]:
    base_imoex = predict_scenario(
        oil=controls["oil"],
        key_rate=controls["key_rate"],
        usd_rub=controls["usd_rub"],
        inflation=controls["inflation"],
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
        )
        adjusted_prediction = apply_scenario_adjustments(raw_prediction, adjustments)
        stock_model = get_stock_model(ticker)
        return adjusted_prediction, raw_prediction, ticker.upper(), stock_model

    raw_prediction = base_imoex
    adjusted_prediction = apply_scenario_adjustments(raw_prediction, adjustments)
    return adjusted_prediction, raw_prediction, "IMOEX", get_imoex_model()


def _current_asset_value(df: pd.DataFrame, asset_type: str, ticker: str | None) -> float:
    if asset_type == "stock":
        if not ticker:
            raise ValueError("Не указан тикер акции")
        stock_col = f"{ticker.lower()}_close"
        stock_df = get_stock_data(ticker)
        return float(stock_df[stock_col].dropna().iloc[-1])
    return float(df["imoex_close"].dropna().iloc[-1])


def main() -> None:
    st.title("Симулятор экономических сценариев для рынка РФ")

    try:
        df = get_data()
    except Exception as exc:
        st.error(f"Не удалось загрузить данные: {exc}")
        st.stop()

    defaults = _defaults_from_data(df)
    controls, asset_type, ticker = _render_sidebar_controls(defaults)
    adjustments = _adjustments_from_controls(controls)

    plot_df = _build_plot_df(df, ticker=ticker if asset_type == "stock" else None)

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
            Здесь показана историческая динамика IMOEX, макрофакторов и (при выборе) отдельной акции.
            График помогает увидеть синхронность/расхождение трендов, а корреляционная матрица — силу линейной связи между факторами.
            Используйте эту вкладку для проверки, насколько выбранная акция чувствительна к нефти, ставке и валютному курсу.
            """
        )
        st.plotly_chart(_historical_chart(plot_df, ticker if asset_type == "stock" else None), use_container_width=True)
        st.plotly_chart(_correlation_heatmap(plot_df, ticker if asset_type == "stock" else None), use_container_width=True)

    with tabs[1]:
        st.markdown(
            """
            **Описание вкладки**
            Сценарный анализ строит прогноз на основе введенных макропараметров.
            Сначала считается базовый прогноз модели (только исторические факторы), затем применяются дополнительные поправки сценария:
            рыночный сентимент, ликвидность, геополитика и регуляторный фактор.
            Итог показывает ожидаемое значение актива и отклонение от текущего уровня.
            """
        )
        try:
            prediction, raw_prediction, asset_label, model_artifact = _predict_asset(
                asset_type=asset_type,
                ticker=ticker,
                controls=controls,
                adjustments=adjustments,
            )
            current = _current_asset_value(df, asset_type=asset_type, ticker=ticker)
        except Exception as exc:
            st.error(f"Ошибка сценарного прогноза: {exc}")
            st.stop()

        delta_abs = prediction - current
        delta_pct = delta_abs / current if current else 0.0
        extra_effect = prediction - raw_prediction

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Текущее значение {asset_label}", f"{current:,.2f}")
        m2.metric(f"Базовый прогноз {asset_label}", f"{raw_prediction:,.2f}")
        m3.metric(f"Итоговый прогноз {asset_label}", f"{prediction:,.2f}", delta=f"{delta_abs:,.2f}")
        m4.metric("Эффект доп. факторов", f"{extra_effect:,.2f}", delta=f"{delta_pct:.2%}")

        st.plotly_chart(_gauge_chart(prediction, current, asset_label=asset_label), use_container_width=True)

        total_adj = sum(adjustments.values())
        st.caption(
            "Суммарная корректировка сценария: "
            f"{total_adj:+.2f}% | "
            f"R²={model_artifact['metrics']['r2']:.3f}, "
            f"MAE={model_artifact['metrics']['mae']:.2f}, "
            f"RMSE={model_artifact['metrics']['rmse']:.2f}"
        )

    with tabs[2]:
        st.markdown(
            """
            **Описание вкладки**
            Monte Carlo генерирует 10 000 случайных сценариев вокруг ваших базовых параметров.
            Для каждого фактора используется нормальное распределение с историческим стандартным отклонением.
            По результату выводятся квантили P5/P50/P95 и вероятность падения актива более чем на 20% от текущего значения.
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
                f"Суммарная корректировка: {mc_result['adjustment_pct']:+.2f}% | "
                f"Масштаб волатильности: {controls['uncertainty_scale']:.1f}x"
            )

    with tabs[3]:
        st.markdown(
            """
            **Описание вкладки**
            Sobol-анализ измеряет вклад каждого макрофактора в разброс прогноза.
            Индекс S1 (первого порядка) показывает, насколько сильно фактор влияет сам по себе.
            Чем выше S1, тем важнее фактор в модели; цвет столбца отражает знак влияния (зеленый — позитивный, красный — негативный).
            """
        )
        if st.button("Рассчитать чувствительность Sobol", key="run_sobol"):
            with st.spinner("Выполняю анализ чувствительности..."):
                try:
                    sobol_result = run_sobol_sensitivity(
                        n_samples=512,
                        asset_type=asset_type,
                        ticker=ticker,
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
                "Важно: дополнительные сценарные поправки (сентимент/ликвидность/геополитика/регуляторика) "
                "являются пост-коррекцией прогноза и не участвуют в Sobol-разложении."
            )


if __name__ == "__main__":
    main()
