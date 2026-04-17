"""Глобальный анализ чувствительности Sobol для IMOEX, акций MOEX и портфеля."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sampler

from src.data_loader import load_moex_stock
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


def _normalize_weights(tickers: list[str], weights: list[float] | None = None) -> dict[str, float]:
    symbols = [t.upper().strip() for t in tickers if t and t.strip()]
    if not symbols:
        raise ValueError("Пустой список тикеров портфеля")

    if weights is None:
        arr = np.array([1.0 / len(symbols)] * len(symbols), dtype=float)
    else:
        if len(weights) != len(symbols):
            raise ValueError("Число весов должно совпадать с числом тикеров")
        arr = np.array(weights, dtype=float)
        if np.any(arr < 0):
            raise ValueError("Вес не может быть отрицательным")
        if np.isclose(arr.sum(), 0):
            raise ValueError("Сумма весов равна нулю")
        arr = arr / arr.sum()

    return {s: float(w) for s, w in zip(symbols, arr)}


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


def _predict_imoex_chain(sampled: pd.DataFrame, regime: str = "all") -> np.ndarray:
    imoex_artifact = load_model_artifact(regime=regime)
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
    return imoex_model.predict(imoex_X)


def _predict_stock_chain(sampled: pd.DataFrame, ticker: str, regime: str = "all") -> np.ndarray:
    imoex_pred = _predict_imoex_chain(sampled, regime=regime)

    stock_artifact = load_stock_model_artifact(ticker, regime=regime)
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


def _predict_portfolio_chain(
    sampled: pd.DataFrame,
    portfolio_tickers: list[str],
    portfolio_weights: list[float] | None,
    regime: str = "all",
) -> np.ndarray:
    weight_map = _normalize_weights(portfolio_tickers, portfolio_weights)
    portfolio_return = np.zeros(len(sampled), dtype=float)

    for symbol, w in weight_map.items():
        pred = _predict_stock_chain(sampled, symbol, regime=regime)
        current = float(load_moex_stock(symbol)[f"{symbol.lower()}_close"].dropna().iloc[-1])
        ret = pred / current - 1.0
        portfolio_return += w * ret

    return 100.0 * (1.0 + portfolio_return)


def _predict_chain(
    sampled: pd.DataFrame,
    asset_type: str,
    ticker: str | None,
    regime: str = "all",
    portfolio_tickers: list[str] | None = None,
    portfolio_weights: list[float] | None = None,
) -> np.ndarray:
    if asset_type == "stock":
        if not ticker:
            raise ValueError("Для анализа акции нужен ticker")
        return _predict_stock_chain(sampled, ticker, regime=regime)

    if asset_type == "portfolio":
        return _predict_portfolio_chain(
            sampled,
            portfolio_tickers=portfolio_tickers or [],
            portfolio_weights=portfolio_weights,
            regime=regime,
        )

    return _predict_imoex_chain(sampled, regime=regime)


def _impact_sign_map_from_samples(sampled: pd.DataFrame, y_pred: np.ndarray) -> dict[str, int]:
    signs: dict[str, int] = {}
    for param in PARAM_TO_FEATURE:
        x = sampled[param].to_numpy(dtype=float)
        corr = np.corrcoef(x, y_pred)[0, 1]
        corr = 0.0 if np.isnan(corr) else float(corr)
        signs[param] = 1 if corr >= 0 else -1
    return signs


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
    regime: str = "all",
    portfolio_tickers: list[str] | None = None,
    portfolio_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Выполнить Sobol-анализ для IMOEX, акции или портфеля."""
    if n_samples < 128:
        raise ValueError("n_samples должен быть >= 128 для устойчивой оценки")

    problem = _build_problem_definition()
    sample_matrix = sobol_sampler.sample(problem, n_samples, calc_second_order=False)

    sampled = pd.DataFrame(sample_matrix, columns=problem["names"])
    y_pred = _predict_chain(
        sampled,
        asset_type=asset_type,
        ticker=ticker,
        regime=regime,
        portfolio_tickers=portfolio_tickers,
        portfolio_weights=portfolio_weights,
    )

    analysis = sobol.analyze(problem, y_pred, calc_second_order=False, print_to_console=False)
    sign_map = _impact_sign_map_from_samples(sampled, y_pred)

    factor_rows = []
    for param_name, s1, s1_conf in zip(problem["names"], analysis["S1"], analysis["S1_conf"]):
        sign = sign_map[param_name]
        factor_rows.append(
            {
                "factor": param_name,
                "factor_label": PARAM_DISPLAY[param_name],
                "S1": float(np.nan_to_num(s1, nan=0.0)),
                "S1_conf": float(np.nan_to_num(s1_conf, nan=0.0)),
                "impact_sign": sign,
                "color": "#31a354" if sign > 0 else "#de2d26",
            }
        )

    sobol_df = pd.DataFrame(factor_rows).sort_values("S1", ascending=False).reset_index(drop=True)

    if asset_type == "stock" and ticker:
        asset_label = f"акции {ticker.upper()}"
    elif asset_type == "portfolio":
        short_names = ", ".join((portfolio_tickers or [])[:4])
        asset_label = f"портфеля ({short_names})"
    else:
        asset_label = "IMOEX"

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
        "n_samples": int(n_samples),
        "n_evaluations": int(len(sampled)),
        "y_variance": float(np.var(y_pred, ddof=1)),
    }


if __name__ == "__main__":
    result = run_sobol_sensitivity(n_samples=256)
    print(result["sobol_df"])
    print(f"Главный фактор: {result['top_factor']} (S1={result['top_s1']:.3f})")
