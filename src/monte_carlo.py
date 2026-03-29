"""Monte Carlo симуляция для IMOEX, акций MOEX и портфеля."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data_loader import load_moex_stock
from src.model import load_model_artifact, load_stock_model_artifact
from src.preprocessing import load_processed_dataset

PARAM_TO_FEATURE = {
    "oil": "brent_usd",
    "key_rate": "key_rate",
    "usd_rub": "usd_rub",
    "inflation": "inflation",
}


def _historical_std_map(uncertainty_scale: float = 1.0) -> dict[str, float]:
    df = load_processed_dataset()
    stds: dict[str, float] = {}
    for param, feature in PARAM_TO_FEATURE.items():
        value = float(df[feature].std(ddof=1))
        value = value if value > 0 else 1e-6
        stds[param] = value * float(max(uncertainty_scale, 1e-6))
    return stds


def _normalize_weights(tickers: list[str], weights: list[float] | None = None) -> dict[str, float]:
    symbols = [t.upper().strip() for t in tickers if t and t.strip()]
    if not symbols:
        raise ValueError("Пустой список тикеров портфеля")

    if weights is None:
        w = np.array([1.0 / len(symbols)] * len(symbols), dtype=float)
    else:
        if len(weights) != len(symbols):
            raise ValueError("Число весов должно совпадать с числом тикеров")
        w = np.array(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("Вес не может быть отрицательным")
        if np.isclose(w.sum(), 0):
            raise ValueError("Сумма весов равна нулю")
        w = w / w.sum()

    return {t: float(v) for t, v in zip(symbols, w)}


def _asset_label(asset_type: str, ticker: str | None, portfolio_tickers: list[str] | None) -> str:
    if asset_type == "stock":
        return f"Акция {ticker.upper()}" if ticker else "Акция"
    if asset_type == "portfolio":
        names = ", ".join((portfolio_tickers or [])[:4])
        return f"Портфель ({names})"
    return "IMOEX"


def _current_asset_level(
    asset_type: str,
    ticker: str | None,
    portfolio_tickers: list[str] | None,
    portfolio_weights: list[float] | None,
) -> float:
    if asset_type == "stock":
        if not ticker:
            raise ValueError("Для расчета акции нужен тикер")
        col = f"{ticker.lower()}_close"
        stock_df = load_moex_stock(ticker)
        return float(stock_df[col].dropna().iloc[-1])

    if asset_type == "portfolio":
        weight_map = _normalize_weights(portfolio_tickers or [], portfolio_weights)
        weighted_return = 0.0
        for symbol, w in weight_map.items():
            col = f"{symbol.lower()}_close"
            s_df = load_moex_stock(symbol)
            s = s_df[col].dropna()
            ret = float(s.iloc[-1] / s.iloc[-2] - 1.0) if len(s) >= 2 else 0.0
            weighted_return += w * ret
        return 100.0 * (1.0 + weighted_return)

    return float(load_processed_dataset()["imoex_close"].dropna().iloc[-1])


def build_monte_carlo_histogram(results: np.ndarray, p5: float, p95: float, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=results,
            nbinsx=60,
            name=f"Распределение {label}",
            marker=dict(color="#2b8cbe"),
            opacity=0.85,
        )
    )

    fig.add_vline(x=p5, line_color="#ef3b2c", line_width=2, line_dash="dash")
    fig.add_vline(x=p95, line_color="#31a354", line_width=2, line_dash="dash")

    fig.add_annotation(x=p5, y=1, yref="paper", text=f"P5: {p5:.1f}", showarrow=False, xanchor="left")
    fig.add_annotation(x=p95, y=1, yref="paper", text=f"P95: {p95:.1f}", showarrow=False, xanchor="right")

    fig.update_layout(
        title=f"Monte Carlo: распределение прогноза {label}",
        xaxis_title=f"Прогноз {label}",
        yaxis_title="Частота",
        bargap=0.02,
        template="plotly_white",
    )
    return fig


def _predict_imoex_for_samples(sampled: dict[str, np.ndarray], regime: str = "all") -> np.ndarray:
    artifact = load_model_artifact(regime=regime)
    model = artifact["model"]
    feature_columns: list[str] = artifact["feature_columns"]

    scenarios = pd.DataFrame(
        {
            "key_rate": sampled["key_rate"],
            "usd_rub": sampled["usd_rub"],
            "brent_usd": sampled["oil"],
            "inflation": sampled["inflation"],
        }
    )[feature_columns]

    return model.predict(scenarios).astype(float)


def _predict_stock_for_samples(
    ticker: str,
    sampled: dict[str, np.ndarray],
    imoex_pred: np.ndarray,
    regime: str = "all",
) -> np.ndarray:
    artifact = load_stock_model_artifact(ticker, regime=regime)
    model = artifact["model"]
    feature_columns: list[str] = artifact["feature_columns"]

    scenarios = pd.DataFrame(
        {
            "key_rate": sampled["key_rate"],
            "usd_rub": sampled["usd_rub"],
            "brent_usd": sampled["oil"],
            "inflation": sampled["inflation"],
            "imoex_close": imoex_pred,
        }
    )[feature_columns]

    return model.predict(scenarios).astype(float)


def _predict_portfolio_for_samples(
    sampled: dict[str, np.ndarray],
    imoex_pred: np.ndarray,
    portfolio_tickers: list[str],
    portfolio_weights: list[float] | None,
    regime: str = "all",
) -> np.ndarray:
    weight_map = _normalize_weights(portfolio_tickers, portfolio_weights)
    results = np.zeros_like(imoex_pred, dtype=float)

    for symbol, w in weight_map.items():
        stock_pred = _predict_stock_for_samples(symbol, sampled, imoex_pred, regime=regime)
        current = float(load_moex_stock(symbol)[f"{symbol.lower()}_close"].dropna().iloc[-1])
        stock_return = stock_pred / current - 1.0
        results += w * stock_return

    # Синтетическая «цена» портфеля как индекс от 100.
    return 100.0 * (1.0 + results)


def run_monte_carlo(
    base_params: dict[str, float],
    n_simulations: int = 10000,
    random_state: int | None = 42,
    asset_type: str = "imoex",
    ticker: str | None = None,
    adjustments: dict[str, float] | None = None,
    uncertainty_scale: float = 1.0,
    regime: str = "all",
    portfolio_tickers: list[str] | None = None,
    portfolio_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Запустить Monte Carlo для IMOEX, акции или портфеля."""
    required = set(PARAM_TO_FEATURE.keys())
    missing = required.difference(base_params.keys())
    if missing:
        raise ValueError(f"Отсутствуют ключи в base_params: {sorted(missing)}")
    if n_simulations < 100:
        raise ValueError("n_simulations должен быть >= 100")

    std_map = _historical_std_map(uncertainty_scale=uncertainty_scale)
    rng = np.random.default_rng(random_state)

    sampled = {
        param: rng.normal(
            loc=float(base_params[param]),
            scale=std_map[param],
            size=n_simulations,
        )
        for param in PARAM_TO_FEATURE
    }

    imoex_pred = _predict_imoex_for_samples(sampled, regime=regime)
    if asset_type == "stock":
        if not ticker:
            raise ValueError("Для asset_type='stock' нужен ticker")
        raw_results = _predict_stock_for_samples(ticker, sampled, imoex_pred, regime=regime)
    elif asset_type == "portfolio":
        raw_results = _predict_portfolio_for_samples(
            sampled,
            imoex_pred,
            portfolio_tickers=portfolio_tickers or [],
            portfolio_weights=portfolio_weights,
            regime=regime,
        )
    else:
        raw_results = imoex_pred

    total_adjustment_pct = float(sum((adjustments or {}).values()))
    adjustment_multiplier = 1.0 + total_adjustment_pct / 100.0
    results = raw_results * adjustment_multiplier

    p5, p50, p95 = np.percentile(results, [5, 50, 95])
    mean_val = float(np.mean(results))
    std_val = float(np.std(results, ddof=1))

    current_level = _current_asset_level(
        asset_type,
        ticker,
        portfolio_tickers=portfolio_tickers,
        portfolio_weights=portfolio_weights,
    )
    drop_threshold = current_level * 0.8
    prob_drop_20 = float(np.mean(results <= drop_threshold))

    label = _asset_label(asset_type, ticker, portfolio_tickers)
    fig = build_monte_carlo_histogram(results, p5, p95, label=label)

    return {
        "results": results,
        "var_5": float(p5),
        "p50": float(p50),
        "var_95": float(p95),
        "mean": mean_val,
        "std": std_val,
        "prob_drop_20": prob_drop_20,
        "figure": fig,
        "std_map": std_map,
        "asset_label": label,
        "adjustment_pct": total_adjustment_pct,
    }


if __name__ == "__main__":
    baseline = {"oil": 80.0, "key_rate": 12.0, "usd_rub": 95.0, "inflation": 7.0}

    out_imoex = run_monte_carlo(baseline, n_simulations=2000)
    print({k: out_imoex[k] for k in ["asset_label", "var_5", "p50", "var_95", "prob_drop_20"]})

    out_stock = run_monte_carlo(baseline, n_simulations=2000, asset_type="stock", ticker="LKOH")
    print({k: out_stock[k] for k in ["asset_label", "var_5", "p50", "var_95", "prob_drop_20"]})
