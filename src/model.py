"""Линейные модели прогноза IMOEX и отдельных акций MOEX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import PROJECT_ROOT, load_moex_stock
from src.preprocessing import load_processed_dataset


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMOEX_MODEL_PATH = MODEL_DIR / "regression_model.pkl"

IMOEX_FEATURE_COLUMNS = ["key_rate", "usd_rub", "brent_usd", "inflation"]
STOCK_FEATURE_COLUMNS = ["key_rate", "usd_rub", "brent_usd", "inflation", "imoex_close"]

TARGET_CLOSE = "imoex_close"
TARGET_RET = "imoex_pct_change"


@dataclass(frozen=True)
class ModelMetrics:
    r2: float
    mae: float
    rmse: float


def _build_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(X) < 12:
        raise ValueError("Недостаточно наблюдений для обучения. Минимум 12 строк.")

    split_idx = max(int(len(X) * (1 - test_size)), 1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if X_test.empty:
        raise ValueError("Тестовая выборка пустая. Уменьшите test_size.")

    return X_train, X_test, y_train, y_test


def _fit_and_pack(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    save_path: Path,
    model_name: str,
) -> dict[str, Any]:
    X = frame[feature_columns]
    y = frame[target_column]

    X_train, X_test, y_train, y_test = _build_train_test_split(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = ModelMetrics(
        r2=float(r2_score(y_test, predictions)),
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
    )

    artifact: dict[str, Any] = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "metrics": metrics.__dict__,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "last_date": frame["date"].max(),
    }

    joblib.dump(artifact, save_path)
    return artifact


def _prepare_imoex_training_frame(use_returns_target: bool = False) -> tuple[pd.DataFrame, str]:
    dataset = load_processed_dataset()
    target_col = TARGET_RET if use_returns_target else TARGET_CLOSE

    required_cols = IMOEX_FEATURE_COLUMNS + [target_col]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки для обучения IMOEX: {missing_cols}")

    frame = dataset[["date", *required_cols]].dropna().sort_values("date")
    return frame, target_col


def train_regression_model(
    use_returns_target: bool = False,
    save_path: Path = IMOEX_MODEL_PATH,
) -> dict[str, Any]:
    """Обучить модель для прогноза IMOEX."""
    frame, target_col = _prepare_imoex_training_frame(use_returns_target=use_returns_target)
    return _fit_and_pack(
        frame=frame,
        feature_columns=IMOEX_FEATURE_COLUMNS,
        target_column=target_col,
        save_path=save_path,
        model_name="IMOEX Linear Regression",
    )


def load_model_artifact(path: Path = IMOEX_MODEL_PATH) -> dict[str, Any]:
    if path.exists():
        return joblib.load(path)
    return train_regression_model(save_path=path)


def _stock_model_path(ticker: str) -> Path:
    return MODEL_DIR / f"stock_model_{ticker.lower()}.pkl"


def _prepare_stock_training_frame(ticker: str) -> tuple[pd.DataFrame, str]:
    symbol = ticker.upper().strip()
    target_col = f"{symbol.lower()}_close"

    macro = load_processed_dataset()[["date", *STOCK_FEATURE_COLUMNS]].copy()
    stock = load_moex_stock(symbol)

    frame = macro.merge(stock, on="date", how="inner")
    frame = frame.dropna().sort_values("date")

    if target_col not in frame.columns:
        raise ValueError(f"Не удалось получить ряд цены для тикера {symbol}")

    if len(frame) < 12:
        raise ValueError(f"Недостаточно истории для тикера {symbol}: {len(frame)} строк")

    return frame, target_col


def train_stock_model(ticker: str) -> dict[str, Any]:
    """Обучить модель для выбранной акции MOEX."""
    symbol = ticker.upper().strip()
    save_path = _stock_model_path(symbol)
    frame, target_col = _prepare_stock_training_frame(symbol)

    artifact = _fit_and_pack(
        frame=frame,
        feature_columns=STOCK_FEATURE_COLUMNS,
        target_column=target_col,
        save_path=save_path,
        model_name=f"{symbol} Linear Regression",
    )
    artifact["ticker"] = symbol
    joblib.dump(artifact, save_path)
    return artifact


def load_stock_model_artifact(ticker: str) -> dict[str, Any]:
    symbol = ticker.upper().strip()
    path = _stock_model_path(symbol)
    if path.exists():
        return joblib.load(path)
    return train_stock_model(symbol)


def apply_scenario_adjustments(
    base_value: float,
    adjustments: dict[str, float] | None = None,
) -> float:
    """Применить пользовательские поправки сценария в процентах к базовому прогнозу."""
    if not adjustments:
        return float(base_value)

    total_pct = float(sum(float(v) for v in adjustments.values()))
    adjusted = float(base_value) * (1.0 + total_pct / 100.0)
    return adjusted


def predict_scenario(
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
    model_path: Path = IMOEX_MODEL_PATH,
    adjustments: dict[str, float] | None = None,
) -> float:
    """Прогноз IMOEX по пользовательскому сценарию."""
    artifact = load_model_artifact(path=model_path)
    model: LinearRegression = artifact["model"]
    feature_columns: list[str] = artifact["feature_columns"]

    scenario_map = {
        "brent_usd": float(oil),
        "key_rate": float(key_rate),
        "usd_rub": float(usd_rub),
        "inflation": float(inflation),
    }

    X_scenario = pd.DataFrame([{col: scenario_map[col] for col in feature_columns}])
    prediction = float(model.predict(X_scenario)[0])
    return apply_scenario_adjustments(prediction, adjustments)


def predict_stock_scenario(
    ticker: str,
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
    imoex_value: float | None = None,
    adjustments: dict[str, float] | None = None,
) -> float:
    """Прогноз цены выбранной акции MOEX."""
    symbol = ticker.upper().strip()
    artifact = load_stock_model_artifact(symbol)
    model: LinearRegression = artifact["model"]
    feature_columns: list[str] = artifact["feature_columns"]

    imoex_level = float(imoex_value) if imoex_value is not None else predict_scenario(
        oil=oil,
        key_rate=key_rate,
        usd_rub=usd_rub,
        inflation=inflation,
    )

    scenario_map = {
        "brent_usd": float(oil),
        "key_rate": float(key_rate),
        "usd_rub": float(usd_rub),
        "inflation": float(inflation),
        "imoex_close": imoex_level,
    }

    X_scenario = pd.DataFrame([{col: scenario_map[col] for col in feature_columns}])
    prediction = float(model.predict(X_scenario)[0])
    return apply_scenario_adjustments(prediction, adjustments)


if __name__ == "__main__":
    imoex_artifact = train_regression_model()
    print("IMOEX metrics:", imoex_artifact["metrics"])

    demo_imoex = predict_scenario(oil=80.0, key_rate=12.0, usd_rub=95.0, inflation=7.0)
    print("IMOEX scenario prediction:", round(demo_imoex, 2))

    demo_stock = predict_stock_scenario("LKOH", oil=80.0, key_rate=12.0, usd_rub=95.0, inflation=7.0)
    print("LKOH scenario prediction:", round(demo_stock, 2))
