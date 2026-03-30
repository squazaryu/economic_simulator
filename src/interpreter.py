"""Интерпретатор результатов Monte Carlo и Sobol для дашборда."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if np.isnan(out):
        return default
    return out


def _risk_bucket(prob_drop_20: float) -> str:
    if prob_drop_20 >= 0.35:
        return "высокий риск"
    if prob_drop_20 >= 0.20:
        return "умеренно высокий риск"
    if prob_drop_20 >= 0.10:
        return "умеренный риск"
    return "умеренно низкий риск"


def _spread_bucket(width_ratio: float) -> str:
    if width_ratio >= 0.35:
        return "широкий диапазон исходов (высокая неопределенность)"
    if width_ratio >= 0.20:
        return "средний диапазон исходов"
    return "узкий диапазон исходов (относительно стабильный сценарный фон)"


def _s1_bucket(s1: float) -> str:
    if s1 >= 0.50:
        return "доминирующий драйвер"
    if s1 >= 0.30:
        return "сильный драйвер"
    if s1 >= 0.15:
        return "заметный драйвер"
    return "второстепенный драйвер"


def interpret_monte_carlo_bin(
    mc_result: dict[str, Any],
    selected_bin: dict[str, Any],
) -> dict[str, Any]:
    current = _safe_float(mc_result.get("current_level"))
    p5 = _safe_float(mc_result.get("var_5"))
    p50 = _safe_float(mc_result.get("p50"))
    p95 = _safe_float(mc_result.get("var_95"))
    prob_drop_20 = _safe_float(mc_result.get("prob_drop_20"), 0.0)
    n_sim = int(_safe_float(mc_result.get("n_simulations"), 0.0))
    label = str(mc_result.get("asset_label", "актива"))

    left = _safe_float(selected_bin.get("left"))
    right = _safe_float(selected_bin.get("right"))
    prob_bin = _safe_float(selected_bin.get("probability"), 0.0)
    count_bin = int(_safe_float(selected_bin.get("count"), 0.0))
    pred_mean = _safe_float(selected_bin.get("prediction_mean"))
    if not np.isfinite(pred_mean):
        pred_mean = (left + right) / 2.0

    p50_delta = (p50 / current - 1.0) if np.isfinite(current) and current != 0 else float("nan")
    bin_delta = (pred_mean / current - 1.0) if np.isfinite(current) and current != 0 else float("nan")
    width_ratio = (p95 - p5) / p50 if np.isfinite(p50) and p50 != 0 else float("nan")

    headline = (
        f"Для {label} текущий сценарный фон оценивается как "
        f"**{_risk_bucket(prob_drop_20)}** с вероятностью падения >20% на уровне **{prob_drop_20:.1%}**."
    )
    summary = (
        f"Выбранный столбец описывает зону прогноза **[{left:.1f}; {right:.1f})**. "
        f"Это **{prob_bin:.1%}** сценариев ({count_bin} из {n_sim}). "
        f"Медианный прогноз по всем сценариям: **{p50:.1f}** "
        f"({p50_delta:+.2%} к текущему уровню)."
    )

    bullets = [
        f"Диапазон P5-P95: {p5:.1f} .. {p95:.1f}; это {_spread_bucket(width_ratio)}.",
        f"Средний результат выбранного бина: {pred_mean:.1f} ({bin_delta:+.2%} к текущему уровню).",
        (
            "Если выбранный бин ниже медианы, это консервативный/стрессовый срез. "
            if pred_mean < p50
            else "Если выбранный бин выше медианы, это более оптимистичный срез относительно базового распределения. "
        )
        + "Оценивайте его вместе с общей вероятностью риска падения."
    ]
    return {"headline": headline, "summary": summary, "bullets": bullets}


def interpret_sobol_factor(
    sobol_result: dict[str, Any],
    factor_row: dict[str, Any],
) -> dict[str, Any]:
    factor_label = str(factor_row.get("factor_label", "Фактор"))
    factor = str(factor_row.get("factor", "factor"))
    s1 = _safe_float(factor_row.get("S1"), 0.0)
    s1_conf = _safe_float(factor_row.get("S1_conf"), 0.0)
    sign = int(_safe_float(factor_row.get("impact_sign"), 1.0))

    top_factor = str(sobol_result.get("top_factor", ""))
    top_s1 = _safe_float(sobol_result.get("top_s1"), 0.0)
    y_var = _safe_float(sobol_result.get("y_variance"))
    explained_var = s1 * y_var if np.isfinite(y_var) else float("nan")

    mechanism_hint = {
        "oil": "влияет через экспортную выручку, бюджетные ожидания и оценку сырьевого сегмента",
        "key_rate": "влияет через стоимость капитала, дисконтирование и кредитную активность",
        "usd_rub": "влияет через валютную переоценку, импортные издержки и инфляционный канал",
        "inflation": "влияет через реальные ставки, потребительский спрос и маржинальность компаний",
    }.get(factor, "влияет через оценочную функцию модели для данного набора данных")

    direction_text = "позитивное" if sign > 0 else "негативное"
    confidence_text = (
        "оценка устойчива"
        if s1_conf <= 0.05
        else ("оценка умеренно устойчива" if s1_conf <= 0.12 else "оценка чувствительна к объему выборки")
    )

    headline = (
        f"Фактор **{factor_label}** классифицируется как **{_s1_bucket(s1)}** "
        f"с вкладом **S1={s1:.3f}** ({direction_text} влияние)."
    )
    summary = (
        f"На фоне текущего разложения главным драйвером остается **{top_factor} (S1={top_s1:.3f})**. "
        f"Для выбранного фактора оценка имеет доверительный шум **±{s1_conf:.3f}**, {confidence_text}."
    )
    bullets = [
        f"Интерпретация канала: {factor_label} {mechanism_hint}.",
        (
            f"Абсолютный вклад в дисперсию прогноза: {explained_var:.2f} единиц Var(Y)."
            if np.isfinite(explained_var)
            else "Абсолютный вклад в дисперсию не рассчитан (недостаточно данных о Var(Y))."
        ),
        "Чем выше S1, тем сильнее приоритет фактора в сценарном мониторинге и стресс-тестах."
    ]
    return {"headline": headline, "summary": summary, "bullets": bullets}

