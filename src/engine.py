"""Engine layer: pure computation wrappers for simulations and sensitivity analysis."""

from __future__ import annotations

from typing import Any

from src.monte_carlo import run_monte_carlo
from src.sensitivity import run_sobol_sensitivity


def run_monte_carlo_engine(
    base_params: dict[str, float],
    n_simulations: int,
    asset_type: str,
    ticker: str | None,
    adjustments: dict[str, float] | None,
    uncertainty_scale: float,
    regime: str,
    portfolio_tickers: list[str] | None,
    portfolio_weights: list[float] | None,
) -> dict[str, Any]:
    return run_monte_carlo(
        base_params=base_params,
        n_simulations=n_simulations,
        asset_type=asset_type,
        ticker=ticker,
        adjustments=adjustments,
        uncertainty_scale=uncertainty_scale,
        regime=regime,
        portfolio_tickers=portfolio_tickers,
        portfolio_weights=portfolio_weights,
    )


def run_sobol_engine(
    n_samples: int,
    asset_type: str,
    ticker: str | None,
    regime: str,
    portfolio_tickers: list[str] | None,
    portfolio_weights: list[float] | None,
) -> dict[str, Any]:
    return run_sobol_sensitivity(
        n_samples=n_samples,
        asset_type=asset_type,
        ticker=ticker,
        regime=regime,
        portfolio_tickers=portfolio_tickers,
        portfolio_weights=portfolio_weights,
    )

