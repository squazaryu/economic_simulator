"""Utilities for loading and merging macro/market data for the simulator."""

from __future__ import annotations

import logging
from io import StringIO
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

try:
    import cbrapi as cbr
except ImportError:  # pragma: no cover - handled by fallback logic
    cbr = None


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MOEX_BASE_URL = "https://iss.moex.com/iss"
DEFAULT_TIMEOUT = 30

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _date_window(years: int = 5) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    return start_date, end_date


def _safe_read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Fallback CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _save_csv(df: pd.DataFrame, file_name: str) -> None:
    output_path = RAW_DIR / file_name
    df.to_csv(output_path, index=False)


def _normalize_monthly_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp("M")


def _request_get(url: str, *, retries: int = 3, backoff_seconds: float = 1.0, **kwargs: Any) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=DEFAULT_TIMEOUT, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:  # pragma: no cover - network variability
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    assert last_error is not None
    raise last_error


def _normalize_time_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    return pd.to_datetime(index)


def _to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def _fetch_cbr_key_rate(start_date: date, end_date: date) -> pd.DataFrame:
    if cbr is None:
        raise ImportError("cbrapi is not installed")

    key_rate_series = cbr.get_key_rate(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        period="D",
    )
    if key_rate_series.empty:
        raise ValueError("CBR key rate series is empty")

    df = key_rate_series.rename("key_rate").to_frame()
    df.index = _normalize_time_index(df.index)
    df = df.resample("ME").last().reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df[["date", "key_rate"]]


def _fetch_cbr_usd_rub(start_date: date, end_date: date) -> pd.DataFrame:
    if cbr is None:
        raise ImportError("cbrapi is not installed")

    usd_series = cbr.get_time_series(
        "USD",
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        period="D",
    )
    if usd_series.empty:
        raise ValueError("CBR USD/RUB series is empty")

    df = usd_series.rename("usd_rub").to_frame()
    df.index = _normalize_time_index(df.index)
    df = df.resample("ME").last().reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    return df[["date", "usd_rub"]]


def _fetch_cbr_inflation(start_date: date, end_date: date) -> pd.DataFrame:
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": start_date.strftime("%d.%m.%Y"),
        "UniDbQuery.To": end_date.strftime("%d.%m.%Y"),
    }
    url = "https://www.cbr.ru/hd_base/infl/"
    response = _request_get(url, params=params)

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError("CBR inflation table is empty")

    raw = tables[0].copy()
    col_date = next((c for c in raw.columns if "Дата" in str(c)), raw.columns[0])
    col_infl = next((c for c in raw.columns if "Инфляция" in str(c)), None)
    if col_infl is None:
        raise ValueError("CBR inflation column was not found")

    raw[col_date] = pd.to_datetime(raw[col_date].astype(str), format="%m.%Y", errors="coerce")
    raw["date"] = raw[col_date].dt.to_period("M").dt.to_timestamp("M")

    inflation = _to_numeric(raw[col_infl])
    # In CBR HTML table values are often presented as integer basis points (e.g. 559 -> 5.59)
    inflation = inflation.where(inflation <= 100, inflation / 100.0)

    df = pd.DataFrame({"date": raw["date"], "inflation": inflation})
    df = df.dropna(subset=["date", "inflation"]).sort_values("date")
    return df


def load_cbr_data() -> pd.DataFrame:
    """Load CBR data: key rate, USD/RUB and inflation on monthly frequency."""
    start_ts, end_ts = _date_window(years=5)
    output_file = RAW_DIR / "cbr_data.csv"

    try:
        key_rate_df = _fetch_cbr_key_rate(start_ts.date(), end_ts.date())
        usd_rub_df = _fetch_cbr_usd_rub(start_ts.date(), end_ts.date())
        inflation_df = _fetch_cbr_inflation(start_ts.date(), end_ts.date())

        merged = key_rate_df.merge(usd_rub_df, on="date", how="outer")
        merged = merged.merge(inflation_df, on="date", how="outer")
        merged["date"] = _normalize_monthly_date(merged["date"])
        merged = merged.sort_values("date").reset_index(drop=True)
        _save_csv(merged, output_file.name)
        return merged
    except Exception as exc:  # pragma: no cover - runtime/network variability
        LOGGER.exception("Failed to load CBR data from API: %s", exc)
        return _safe_read_csv(output_file)


def _fetch_moex_candles(
    security: str,
    market: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    url = f"{MOEX_BASE_URL}/engines/stock/markets/{market}/securities/{security}/candles.json"
    start = 0
    rows: list[dict[str, Any]] = []

    while True:
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "till": end_date.strftime("%Y-%m-%d"),
            "interval": 31,
            "start": start,
        }
        response = _request_get(url, params=params)
        payload = response.json().get("candles", {})

        columns = payload.get("columns", [])
        data = payload.get("data", [])
        if not data:
            break

        rows.extend(dict(zip(columns, item)) for item in data)
        if len(data) < 100:
            break
        start += len(data)

    if not rows:
        raise ValueError(f"MOEX returned no candles for {security}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["begin"]).dt.to_period("M").dt.to_timestamp("M")
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    return df


def load_moex_index() -> pd.DataFrame:
    """Load monthly close for IMOEX and save oil-sector reference quotes."""
    start_ts, end_ts = _date_window(years=5)
    imoex_file = RAW_DIR / "moex_imoex.csv"
    oil_file = RAW_DIR / "moex_oil_stocks.csv"

    try:
        imoex_raw = _fetch_moex_candles("IMOEX", "index", start_ts.date(), end_ts.date())
        imoex_df = imoex_raw[["date", "close"]].rename(columns={"close": "imoex_close"})

        oil_frames: list[pd.DataFrame] = []
        for symbol in ("LKOH", "ROSN", "TATN"):
            share_raw = _fetch_moex_candles(symbol, "shares", start_ts.date(), end_ts.date())
            one = share_raw[["date", "close"]].rename(columns={"close": f"{symbol.lower()}_close"})
            oil_frames.append(one)

        oil_df = oil_frames[0]
        for frame in oil_frames[1:]:
            oil_df = oil_df.merge(frame, on="date", how="outer")

        _save_csv(imoex_df, imoex_file.name)
        _save_csv(oil_df, oil_file.name)
        return imoex_df
    except Exception as exc:  # pragma: no cover - runtime/network variability
        LOGGER.exception("Failed to load MOEX data from API: %s", exc)
        return _safe_read_csv(imoex_file)


def load_moex_stock(ticker: str) -> pd.DataFrame:
    """Load monthly close for a selected MOEX stock ticker."""
    start_ts, end_ts = _date_window(years=5)
    symbol = ticker.upper().strip()
    close_col = f"{symbol.lower()}_close"
    output_file = RAW_DIR / f"moex_{symbol.lower()}.csv"

    try:
        stock_raw = _fetch_moex_candles(symbol, "shares", start_ts.date(), end_ts.date())
        stock_df = stock_raw[["date", "close"]].rename(columns={"close": close_col})
        _save_csv(stock_df, output_file.name)
        return stock_df
    except Exception as exc:  # pragma: no cover - runtime/network variability
        LOGGER.exception("Failed to load MOEX stock %s: %s", symbol, exc)
        if output_file.exists():
            return _safe_read_csv(output_file)

        oil_file = RAW_DIR / "moex_oil_stocks.csv"
        if oil_file.exists():
            oil_df = _safe_read_csv(oil_file)
            if close_col in oil_df.columns:
                return oil_df[["date", close_col]]
        raise


def load_oil_price() -> pd.DataFrame:
    """Load Brent monthly close from Yahoo Finance and save to raw CSV."""
    start_ts, end_ts = _date_window(years=5)
    output_file = RAW_DIR / "oil_brent.csv"

    try:
        ticker = yf.Ticker("BZ=F")
        hist = ticker.history(
            start=start_ts.strftime("%Y-%m-%d"),
            end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        if hist.empty:
            raise ValueError("Brent history is empty")

        if isinstance(hist.columns, pd.MultiIndex):
            close_series = hist[("Close", "BZ=F")]
        else:
            close_series = hist["Close"]

        oil_df = (
            close_series.to_frame("brent_usd")
            .reset_index()
            .rename(columns={"Date": "date"})
        )
        oil_df["date"] = (
            pd.to_datetime(oil_df["date"], utc=True).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp("M")
        )
        oil_df = oil_df.groupby("date", as_index=False)["brent_usd"].last()

        _save_csv(oil_df, output_file.name)
        return oil_df
    except Exception as exc:  # pragma: no cover - runtime/network variability
        LOGGER.exception("Failed to load Brent data from API: %s", exc)
        return _safe_read_csv(output_file)


def merge_all_data() -> pd.DataFrame:
    """Merge all source series into one monthly dataframe and save to processed CSV."""
    cbr_df = load_cbr_data()
    imoex_df = load_moex_index()
    oil_df = load_oil_price()

    for frame in (cbr_df, imoex_df, oil_df):
        frame["date"] = _normalize_monthly_date(frame["date"])

    merged = cbr_df.merge(imoex_df, on="date", how="outer")
    merged = merged.merge(oil_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.ffill()

    output_path = PROCESSED_DIR / "merged_data.csv"
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Loading CBR data...")
    print(load_cbr_data().tail())

    print("\nLoading IMOEX data...")
    print(load_moex_index().tail())

    print("\nLoading Brent data...")
    print(load_oil_price().tail())

    print("\nMerging all data...")
    print(merge_all_data().tail())
