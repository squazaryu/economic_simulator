"""Линейные модели прогноза IMOEX, акций MOEX и портфельной аналитики."""

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

REGIME_CUTOFF = pd.Timestamp("2022-02-01")
REGIME_OPTIONS = {"all", "pre_2022", "post_2022"}


@dataclass(frozen=True)
class ModelMetrics:
    r2: float
    mae: float
    rmse: float


def _validate_regime(regime: str) -> str:
    regime_norm = (regime or "all").strip().lower()
    if regime_norm not in REGIME_OPTIONS:
        raise ValueError(f"Неизвестный режим: {regime}. Допустимо: {sorted(REGIME_OPTIONS)}")
    return regime_norm


def _safe_artifact_regime(raw_regime: Any) -> str:
    try:
        return _validate_regime(str(raw_regime or "all"))
    except ValueError:
        return "all"


def _regime_suffix(regime: str) -> str:
    regime_norm = _validate_regime(regime)
    return "" if regime_norm == "all" else f"_{regime_norm}"


def _apply_regime_filter(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    regime_norm = _validate_regime(regime)
    if regime_norm == "all":
        return frame

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    if regime_norm == "pre_2022":
        out = out[out["date"] < REGIME_CUTOFF]
    else:  # post_2022
        out = out[out["date"] >= REGIME_CUTOFF]
    return out


def _build_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(X) < 8:
        raise ValueError("Недостаточно наблюдений для обучения. Минимум 8 строк.")

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
    regime: str,
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

    feature_means = {col: float(X[col].mean()) for col in feature_columns}
    feature_stds = {col: float(X[col].std(ddof=1)) for col in feature_columns}

    artifact: dict[str, Any] = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "metrics": metrics.__dict__,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "last_date": pd.to_datetime(frame["date"]).max(),
        "regime": _validate_regime(regime),
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "intercept": float(model.intercept_),
        "coefs": {col: float(coef) for col, coef in zip(feature_columns, model.coef_)},
    }

    joblib.dump(artifact, save_path)
    return artifact


def _build_feature_stats(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    X = frame[feature_columns]
    means = {col: float(X[col].mean()) for col in feature_columns}
    stds = {col: float(X[col].std(ddof=1)) for col in feature_columns}
    return means, stds


def _normalize_loaded_artifact(
    artifact: dict[str, Any],
    *,
    feature_columns: list[str],
    regime: str,
    stats_frame: pd.DataFrame | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    out = dict(artifact)
    cols = list(out.get("feature_columns") or feature_columns)
    out["feature_columns"] = cols
    out["regime"] = _safe_artifact_regime(out.get("regime", regime))

    model = out.get("model")
    if ("intercept" not in out or out.get("intercept") is None) and model is not None and hasattr(model, "intercept_"):
        out["intercept"] = float(model.intercept_)

    coefs = out.get("coefs")
    if not isinstance(coefs, dict):
        if model is not None and hasattr(model, "coef_"):
            raw_coef = np.asarray(model.coef_, dtype=float).ravel()
            if len(raw_coef) == len(cols):
                out["coefs"] = {col: float(value) for col, value in zip(cols, raw_coef)}

    means = out.get("feature_means")
    stds = out.get("feature_stds")
    if not isinstance(means, dict) or not isinstance(stds, dict):
        if stats_frame is not None and all(col in stats_frame.columns for col in cols):
            calc_means, calc_stds = _build_feature_stats(stats_frame, cols)
        else:
            calc_means = {col: 0.0 for col in cols}
            calc_stds = {col: 0.0 for col in cols}
        out["feature_means"] = calc_means
        out["feature_stds"] = calc_stds

    if ticker:
        out["ticker"] = ticker.upper().strip()
    return out


def _artifact_is_usable(artifact: dict[str, Any]) -> bool:
    if "model" not in artifact:
        return False
    cols = artifact.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        return False
    if artifact.get("intercept") is None:
        return False
    if not isinstance(artifact.get("coefs"), dict):
        return False
    if not isinstance(artifact.get("feature_means"), dict):
        return False
    if not isinstance(artifact.get("feature_stds"), dict):
        return False
    return True


def _artifact_coefs(artifact: dict[str, Any]) -> dict[str, float]:
    coefs = artifact.get("coefs")
    if isinstance(coefs, dict):
        return {str(k): float(v) for k, v in coefs.items()}

    cols = [str(c) for c in artifact.get("feature_columns", [])]
    model = artifact.get("model")
    if model is not None and cols and hasattr(model, "coef_"):
        raw_coef = np.asarray(model.coef_, dtype=float).ravel()
        if len(raw_coef) == len(cols):
            return {col: float(value) for col, value in zip(cols, raw_coef)}
    return {col: 0.0 for col in cols}


def _artifact_feature_means(artifact: dict[str, Any]) -> dict[str, float]:
    cols = [str(c) for c in artifact.get("feature_columns", [])]
    means = artifact.get("feature_means")
    if isinstance(means, dict):
        return {col: float(means.get(col, 0.0)) for col in cols}
    return {col: 0.0 for col in cols}


def _prepare_imoex_training_frame(use_returns_target: bool = False, regime: str = "all") -> tuple[pd.DataFrame, str]:
    dataset = load_processed_dataset()
    target_col = TARGET_RET if use_returns_target else TARGET_CLOSE

    required_cols = IMOEX_FEATURE_COLUMNS + [target_col]
    missing_cols = [col for col in required_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют колонки для обучения IMOEX: {missing_cols}")

    frame = dataset[["date", *required_cols]].dropna().sort_values("date")
    frame = _apply_regime_filter(frame, regime)
    if len(frame) < 8:
        raise ValueError(f"Слишком мало данных в режиме {regime}: {len(frame)} строк")
    return frame, target_col


def _imoex_model_path(regime: str = "all") -> Path:
    regime_norm = _validate_regime(regime)
    if regime_norm == "all":
        return IMOEX_MODEL_PATH
    return MODEL_DIR / f"regression_model{_regime_suffix(regime_norm)}.pkl"


def train_regression_model(
    use_returns_target: bool = False,
    save_path: Path | None = None,
    regime: str = "all",
) -> dict[str, Any]:
    """Обучить модель для прогноза IMOEX."""
    regime_norm = _validate_regime(regime)
    frame, target_col = _prepare_imoex_training_frame(use_returns_target=use_returns_target, regime=regime_norm)
    out_path = save_path or _imoex_model_path(regime_norm)
    return _fit_and_pack(
        frame=frame,
        feature_columns=IMOEX_FEATURE_COLUMNS,
        target_column=target_col,
        save_path=out_path,
        model_name="IMOEX Linear Regression",
        regime=regime_norm,
    )


def load_model_artifact(path: Path | None = None, regime: str = "all") -> dict[str, Any]:
    regime_norm = _validate_regime(regime)
    model_path = path or _imoex_model_path(regime_norm)
    if model_path.exists():
        artifact = joblib.load(model_path)
        artifact_regime = _safe_artifact_regime(artifact.get("regime", "all"))
        if artifact_regime == regime_norm:
            should_persist = (
                not isinstance(artifact.get("coefs"), dict)
                or artifact.get("intercept") is None
                or not isinstance(artifact.get("feature_means"), dict)
                or not isinstance(artifact.get("feature_stds"), dict)
                or not isinstance(artifact.get("feature_columns"), list)
                or _safe_artifact_regime(artifact.get("regime", "all")) != regime_norm
            )
            stats_frame: pd.DataFrame | None = None
            if not isinstance(artifact.get("feature_means"), dict) or not isinstance(artifact.get("feature_stds"), dict):
                try:
                    stats_frame, _ = _prepare_imoex_training_frame(use_returns_target=False, regime=regime_norm)
                except Exception:
                    stats_frame = None
            normalized = _normalize_loaded_artifact(
                artifact,
                feature_columns=IMOEX_FEATURE_COLUMNS,
                regime=regime_norm,
                stats_frame=stats_frame,
            )
            if _artifact_is_usable(normalized):
                if should_persist:
                    joblib.dump(normalized, model_path)
                return normalized
    return train_regression_model(save_path=model_path, regime=regime_norm)


def _stock_model_path(ticker: str, regime: str = "all") -> Path:
    symbol = ticker.upper().strip()
    return MODEL_DIR / f"stock_model_{symbol.lower()}{_regime_suffix(regime)}.pkl"


def _prepare_stock_training_frame(ticker: str, regime: str = "all") -> tuple[pd.DataFrame, str]:
    symbol = ticker.upper().strip()
    target_col = f"{symbol.lower()}_close"

    macro = load_processed_dataset()[["date", *STOCK_FEATURE_COLUMNS]].copy()
    stock = load_moex_stock(symbol)

    frame = macro.merge(stock, on="date", how="inner")
    frame = frame.dropna().sort_values("date")
    frame = _apply_regime_filter(frame, regime)

    if target_col not in frame.columns:
        raise ValueError(f"Не удалось получить ряд цены для тикера {symbol}")

    if len(frame) < 8:
        raise ValueError(f"Недостаточно истории для тикера {symbol} в режиме {regime}: {len(frame)} строк")

    return frame, target_col


def train_stock_model(ticker: str, regime: str = "all") -> dict[str, Any]:
    """Обучить модель для выбранной акции MOEX."""
    symbol = ticker.upper().strip()
    regime_norm = _validate_regime(regime)
    save_path = _stock_model_path(symbol, regime=regime_norm)
    frame, target_col = _prepare_stock_training_frame(symbol, regime=regime_norm)

    artifact = _fit_and_pack(
        frame=frame,
        feature_columns=STOCK_FEATURE_COLUMNS,
        target_column=target_col,
        save_path=save_path,
        model_name=f"{symbol} Linear Regression",
        regime=regime_norm,
    )
    artifact["ticker"] = symbol
    joblib.dump(artifact, save_path)
    return artifact


def load_stock_model_artifact(ticker: str, regime: str = "all") -> dict[str, Any]:
    symbol = ticker.upper().strip()
    regime_norm = _validate_regime(regime)
    path = _stock_model_path(symbol, regime=regime_norm)
    if path.exists():
        artifact = joblib.load(path)
        artifact_regime = _safe_artifact_regime(artifact.get("regime", "all"))
        if artifact_regime == regime_norm:
            should_persist = (
                not isinstance(artifact.get("coefs"), dict)
                or artifact.get("intercept") is None
                or not isinstance(artifact.get("feature_means"), dict)
                or not isinstance(artifact.get("feature_stds"), dict)
                or not isinstance(artifact.get("feature_columns"), list)
                or _safe_artifact_regime(artifact.get("regime", "all")) != regime_norm
            )
            stats_frame: pd.DataFrame | None = None
            if not isinstance(artifact.get("feature_means"), dict) or not isinstance(artifact.get("feature_stds"), dict):
                try:
                    stats_frame, _ = _prepare_stock_training_frame(symbol, regime=regime_norm)
                except Exception:
                    stats_frame = None
            normalized = _normalize_loaded_artifact(
                artifact,
                feature_columns=STOCK_FEATURE_COLUMNS,
                regime=regime_norm,
                stats_frame=stats_frame,
                ticker=symbol,
            )
            if _artifact_is_usable(normalized):
                if should_persist:
                    joblib.dump(normalized, path)
                return normalized
    return train_stock_model(symbol, regime=regime_norm)


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
    model_path: Path | None = None,
    adjustments: dict[str, float] | None = None,
    regime: str = "all",
) -> float:
    """Прогноз IMOEX по пользовательскому сценарию."""
    artifact = load_model_artifact(path=model_path, regime=regime)
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
    regime: str = "all",
) -> float:
    """Прогноз цены выбранной акции MOEX."""
    symbol = ticker.upper().strip()
    artifact = load_stock_model_artifact(symbol, regime=regime)
    model: LinearRegression = artifact["model"]
    feature_columns: list[str] = artifact["feature_columns"]

    imoex_level = float(imoex_value) if imoex_value is not None else predict_scenario(
        oil=oil,
        key_rate=key_rate,
        usd_rub=usd_rub,
        inflation=inflation,
        regime=regime,
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


def explain_imoex_drivers(
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
    regime: str = "all",
) -> pd.DataFrame:
    artifact = load_model_artifact(regime=regime)
    coefs = _artifact_coefs(artifact)
    means = _artifact_feature_means(artifact)

    feature_values = {
        "brent_usd": float(oil),
        "key_rate": float(key_rate),
        "usd_rub": float(usd_rub),
        "inflation": float(inflation),
    }

    rows = []
    for feat, value in feature_values.items():
        coef = float(coefs.get(feat, 0.0))
        delta = float(value - float(means.get(feat, value)))
        contribution = coef * delta
        rows.append({"factor": feat, "coef": coef, "delta": delta, "contribution": contribution})

    return pd.DataFrame(rows).sort_values("contribution", key=lambda s: s.abs(), ascending=False)


def explain_stock_drivers(
    ticker: str,
    oil: float,
    key_rate: float,
    usd_rub: float,
    inflation: float,
    imoex_value: float,
    regime: str = "all",
) -> pd.DataFrame:
    artifact = load_stock_model_artifact(ticker, regime=regime)
    coefs = _artifact_coefs(artifact)
    means = _artifact_feature_means(artifact)

    feature_values = {
        "brent_usd": float(oil),
        "key_rate": float(key_rate),
        "usd_rub": float(usd_rub),
        "inflation": float(inflation),
        "imoex_close": float(imoex_value),
    }

    rows = []
    for feat, value in feature_values.items():
        coef = float(coefs.get(feat, 0.0))
        delta = float(value - float(means.get(feat, value)))
        contribution = coef * delta
        rows.append({"factor": feat, "coef": coef, "delta": delta, "contribution": contribution})

    return pd.DataFrame(rows).sort_values("contribution", key=lambda s: s.abs(), ascending=False)


if __name__ == "__main__":
    imoex_artifact = train_regression_model(regime="all")
    print("IMOEX metrics:", imoex_artifact["metrics"])

    demo_imoex = predict_scenario(oil=80.0, key_rate=12.0, usd_rub=95.0, inflation=7.0, regime="post_2022")
    print("IMOEX scenario prediction:", round(demo_imoex, 2))

    demo_stock = predict_stock_scenario("LKOH", oil=80.0, key_rate=12.0, usd_rub=95.0, inflation=7.0, regime="all")
    print("LKOH scenario prediction:", round(demo_stock, 2))
