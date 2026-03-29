"""Monte Carlo симуляция для IMOEX и выбранной акции MOEX."""

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


def _asset_label(asset_type: str, ticker: str | None) -> str:
    if asset_type == "stock":
        return f"Акция {ticker.upper()}" if ticker else "Акция"
    return "IMOEX"


def _current_asset_level(asset_type: str, ticker: str | None) -> float:
    if asset_type == "stock":
        if not ticker:
            raise ValueError("Для расчета акции нужен тикер")
        col = f"{ticker.lower()}_close"
        stock_df = load_moex_stock(ticker)
        return float(stock_df[col].dropna().iloc[-1])

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


def _predict_imoex_for_samples(sampled: dict[str, np.ndarray]) -> np.ndarray:
    artifact = load_model_artifact()
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


def _predict_stock_for_samples(ticker: str, sampled: dict[str, np.ndarray], imoex_pred: np.ndarray) -> np.ndarray:
    artifact = load_stock_model_artifact(ticker)
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


def run_monte_carlo(
    base_params: dict[str, float],
    n_simulations: int = 10000,
    random_state: int | None = 42,
    asset_type: str = "imoex",
    ticker: str | None = None,
    adjustments: dict[str, float] | None = None,
    uncertainty_scale: float = 1.0,
) -> dict[str, Any]:
    """Запустить Monte Carlo для IMOEX или выбранной акции."""
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

    imoex_pred = _predict_imoex_for_samples(sampled)
    if asset_type == "stock":
        if not ticker:
            raise ValueError("Для asset_type='stock' нужен ticker")
        raw_results = _predict_stock_for_samples(ticker, sampled, imoex_pred)
    else:
        raw_results = imoex_pred

    total_adjustment_pct = float(sum((adjustments or {}).values()))
    adjustment_multiplier = 1.0 + total_adjustment_pct / 100.0
    results = raw_results * adjustment_multiplier

    p5, p50, p95 = np.percentile(results, [5, 50, 95])
    mean_val = float(np.mean(results))
    std_val = float(np.std(results, ddof=1))

    current_level = _current_asset_level(asset_type, ticker)
    drop_threshold = current_level * 0.8
    prob_drop_20 = float(np.mean(results <= drop_threshold))

    label = _asset_label(asset_type, ticker)
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

    out_imoex = run_monte_carlo(baseline, n_simulations=3000)
    print({k: out_imoex[k] for k in ["asset_label", "var_5", "p50", "var_95", "prob_drop_20"]})

    out_stock = run_monte_carlo(baseline, n_simulations=3000, asset_type="stock", ticker="LKOH")
    print({k: out_stock[k] for k in ["asset_label", "var_5", "p50", "var_95", "prob_drop_20"]})
