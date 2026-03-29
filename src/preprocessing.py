"""Preprocessing pipeline for the economic simulator dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT, merge_all_data

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = ["key_rate", "usd_rub", "inflation", "imoex_close", "brent_usd"]


def clean_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates, enforce monthly index and fill missing values."""
    processed = df.copy()
    processed["date"] = pd.to_datetime(processed["date"]).dt.to_period("M").dt.to_timestamp("M")
    processed = processed.drop_duplicates(subset=["date"]).sort_values("date")

    for col in FEATURE_COLUMNS:
        if col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors="coerce")

    full_index = pd.date_range(
        start=processed["date"].min(),
        end=processed["date"].max(),
        freq="ME",
    )

    processed = processed.set_index("date").reindex(full_index)
    processed.index.name = "date"

    num_cols = [c for c in FEATURE_COLUMNS if c in processed.columns]
    processed[num_cols] = processed[num_cols].interpolate(method="time")
    processed[num_cols] = processed[num_cols].ffill().bfill()

    # Optional target variant for modeling sensitivity to relative changes.
    if "imoex_close" in processed.columns:
        processed["imoex_pct_change"] = processed["imoex_close"].pct_change() * 100
        processed["imoex_pct_change"] = processed["imoex_pct_change"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return processed.reset_index()


def build_processed_dataset(output_name: str = "dataset_monthly.csv") -> pd.DataFrame:
    """Run full preprocessing: load, merge, clean, and save processed dataset."""
    merged_df = merge_all_data()
    processed_df = clean_and_interpolate(merged_df)

    output_path = PROCESSED_DIR / output_name
    processed_df.to_csv(output_path, index=False)
    return processed_df


def load_processed_dataset(file_name: str = "dataset_monthly.csv") -> pd.DataFrame:
    path = PROCESSED_DIR / file_name
    if not path.exists():
        return build_processed_dataset(output_name=file_name)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    dataset = build_processed_dataset()
    print(dataset.tail())
