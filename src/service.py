"""Service layer: application orchestration between UI and engine/system APIs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.engine import run_monte_carlo_engine, run_sobol_engine
from src.system_apis import (
    fetch_imoex_model_artifact,
    fetch_moex_ticker_universe,
    fetch_processed_dataset,
    fetch_stock_history,
    fetch_stock_model_artifact,
)


def get_processed_dataset_service() -> pd.DataFrame:
    return fetch_processed_dataset()


def get_stock_history_service(ticker: str) -> pd.DataFrame:
    return fetch_stock_history(ticker)


def get_moex_ticker_universe_service() -> pd.DataFrame:
    return fetch_moex_ticker_universe()


def get_imoex_model_service(regime: str) -> dict[str, Any]:
    return fetch_imoex_model_artifact(regime)


def get_stock_model_service(ticker: str, regime: str) -> dict[str, Any]:
    return fetch_stock_model_artifact(ticker, regime)


def run_monte_carlo_service(
    *,
    controls: dict[str, float],
    mc_runs: int,
    asset_type: str,
    ticker: str | None,
    adjustments: dict[str, float] | None,
    regime: str,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float] | None,
) -> dict[str, Any]:
    weights_list = [portfolio_weights[t] for t in portfolio_tickers] if portfolio_weights else None
    return run_monte_carlo_engine(
        base_params={
            "oil": float(controls["oil"]),
            "key_rate": float(controls["key_rate"]),
            "usd_rub": float(controls["usd_rub"]),
            "inflation": float(controls["inflation"]),
        },
        n_simulations=int(mc_runs),
        asset_type=asset_type,
        ticker=ticker,
        adjustments=adjustments,
        uncertainty_scale=float(controls["uncertainty_scale"]),
        regime=regime,
        portfolio_tickers=portfolio_tickers,
        portfolio_weights=weights_list,
    )


def run_sobol_service(
    *,
    n_samples: int,
    asset_type: str,
    ticker: str | None,
    regime: str,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, float] | None,
    random_seed: int = 42,
) -> dict[str, Any]:
    weights_list = [portfolio_weights[t] for t in portfolio_tickers] if portfolio_weights else None
    return run_sobol_engine(
        n_samples=int(n_samples),
        asset_type=asset_type,
        ticker=ticker,
        regime=regime,
        portfolio_tickers=portfolio_tickers,
        portfolio_weights=weights_list,
        random_seed=int(random_seed),
    )
