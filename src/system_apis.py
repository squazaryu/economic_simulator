"""System APIs layer: direct access to external/local data sources and model artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_loader import load_moex_stock, load_moex_tickers
from src.model import load_model_artifact, load_stock_model_artifact
from src.preprocessing import build_processed_dataset


def fetch_processed_dataset() -> pd.DataFrame:
    return build_processed_dataset()


def fetch_stock_history(ticker: str) -> pd.DataFrame:
    return load_moex_stock(ticker)


def fetch_moex_ticker_universe() -> pd.DataFrame:
    return load_moex_tickers()


def fetch_imoex_model_artifact(regime: str) -> dict[str, Any]:
    return load_model_artifact(regime=regime)


def fetch_stock_model_artifact(ticker: str, regime: str) -> dict[str, Any]:
    return load_stock_model_artifact(ticker, regime=regime)

