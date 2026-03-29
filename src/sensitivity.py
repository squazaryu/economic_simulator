"""Глобальный анализ чувствительности Sobol для IMOEX и акций MOEX."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from SALib.analyze import sobol
from SALib.sample import saltelli

from src.model import load_model_artifact, load_stock_model_artifact
from src.preprocessing import load_processed_dataset

PARAM_TO_FEATURE = {
    "oil": "brent_usd",
    "key_rate": "key_rate",
    "usd_rub": "usd_rub",
    "inflation": "inflation",
}

PARAM_DISPLAY = {
    "oil": "Нефть Brent",
    "key_rate": "Ключевая ставка",
    "usd_rub": "USD/RUB",
    "inflation": "Инфляция",
}


def _build_problem_definition() -> dict[str, Any]:
    df = load_processed_dataset()
    names = list(PARAM_TO_FEATURE.keys())
    bounds: list[list[float]] = []

    for param in names:
        feature = PARAM_TO_FEATURE[param]
        col = pd.to_numeric(df[feature], errors="coerce").dropna()
        low = float(col.quantile(0.05))
        high = float(col.quantile(0.95))

        if np.isclose(low, high):
            high = low + 1e-3
        bounds.append([low, high])

    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }


def _predict_chain(sampled: pd.DataFrame, asset_type: str, ticker: str | None) -> np.ndarray:
    imoex_artifact = load_model_artifact()
    imoex_model = imoex_artifact["model"]
    imoex_features: list[str] = imoex_artifact["feature_columns"]

    imoex_X = pd.DataFrame(
        {
            "key_rate": sampled["key_rate"],
            "usd_rub": sampled["usd_rub"],
            "brent_usd": sampled["oil"],
            "inflation": sampled["inflation"],
        }
    )[imoex_features]
    imoex_pred = imoex_model.predict(imoex_X)

    if asset_type != "stock":
        return imoex_pred

    if not ticker:
        raise ValueError("Для анализа акции нужен ticker")

    stock_artifact = load_stock_model_artifact(ticker)
    stock_model = stock_artifact["model"]
    stock_features: list[str] = stock_artifact["feature_columns"]

    stock_X = pd.DataFrame(
        {
            "key_rate": sampled["key_rate"],
            "usd_rub": sampled["usd_rub"],
            "brent_usd": sampled["oil"],
            "inflation": sampled["inflation"],
            "imoex_close": imoex_pred,
        }
    )[stock_features]
    return stock_model.predict(stock_X)


def _impact_sign_map(asset_type: str, ticker: str | None) -> dict[str, int]:
    imoex_artifact = load_model_artifact()
    imoex_coef = dict(zip(imoex_artifact["feature_columns"], imoex_artifact["model"].coef_))

    if asset_type != "stock":
        return {
            param: 1 if float(imoex_coef.get(PARAM_TO_FEATURE[param], 0.0)) >= 0 else -1
            for param in PARAM_TO_FEATURE
        }

    if not ticker:
        raise ValueError("Для анализа акции нужен ticker")

    stock_artifact = load_stock_model_artifact(ticker)
    stock_coef = dict(zip(stock_artifact["feature_columns"], stock_artifact["model"].coef_))

    sign_map: dict[str, int] = {}
    beta_imoex_stock = float(stock_coef.get("imoex_close", 0.0))
    for param, feature in PARAM_TO_FEATURE.items():
        direct = float(stock_coef.get(feature, 0.0))
        indirect = beta_imoex_stock * float(imoex_coef.get(feature, 0.0))
        total = direct + indirect
        sign_map[param] = 1 if total >= 0 else -1
    return sign_map


def build_tornado_chart(sobol_df: pd.DataFrame, asset_label: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=sobol_df["S1"],
            y=sobol_df["factor_label"],
            orientation="h",
            marker_color=sobol_df["color"],
            text=[f"{x:.3f}" for x in sobol_df["S1"]],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"Анализ чувствительности Sobol (S1) для {asset_label}",
        xaxis_title="Индекс Соболя первого порядка (S1)",
        yaxis_title="Фактор",
        template="plotly_white",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(sobol_df["factor_label"]))
    return fig


def run_sobol_sensitivity(
    n_samples: int = 512,
    asset_type: str = "imoex",
    ticker: str | None = None,
) -> dict[str, Any]:
    """Выполнить Sobol-анализ для IMOEX или выбранной акции."""
    if n_samples < 128:
        raise ValueError("n_samples должен быть >= 128 для устойчивой оценки")

    problem = _build_problem_definition()
    sample_matrix = saltelli.sample(problem, n_samples, calc_second_order=False)

    sampled = pd.DataFrame(sample_matrix, columns=problem["names"])
    y_pred = _predict_chain(sampled, asset_type=asset_type, ticker=ticker)

    analysis = sobol.analyze(problem, y_pred, calc_second_order=False, print_to_console=False)
    sign_map = _impact_sign_map(asset_type=asset_type, ticker=ticker)

    factor_rows = []
    for param_name, s1 in zip(problem["names"], analysis["S1"]):
        sign = sign_map[param_name]
        factor_rows.append(
            {
                "factor": param_name,
                "factor_label": PARAM_DISPLAY[param_name],
                "S1": float(np.nan_to_num(s1, nan=0.0)),
                "impact_sign": sign,
                "color": "#31a354" if sign > 0 else "#de2d26",
            }
        )

    sobol_df = pd.DataFrame(factor_rows).sort_values("S1", ascending=False).reset_index(drop=True)
    asset_label = f"акции {ticker.upper()}" if asset_type == "stock" and ticker else "IMOEX"
    fig = build_tornado_chart(sobol_df, asset_label=asset_label)

    top_factor = sobol_df.iloc[0]["factor_label"]
    top_s1 = float(sobol_df.iloc[0]["S1"])

    return {
        "sobol_df": sobol_df,
        "top_factor": top_factor,
        "top_s1": top_s1,
        "figure": fig,
        "problem": problem,
        "asset_label": asset_label,
    }


if __name__ == "__main__":
    result = run_sobol_sensitivity(n_samples=512)
    print(result["sobol_df"])
    print(f"Главный фактор: {result['top_factor']} (S1={result['top_s1']:.3f})")
