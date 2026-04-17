from __future__ import annotations

from streamlit.testing.v1 import AppTest


APP_FILE = "app.py"


def _run_mc(at: AppTest) -> None:
    mc_idx = next(i for i, b in enumerate(at.button) if b.label == "Запустить симуляцию Монте-Карло")
    at.button[mc_idx].click().run(timeout=300)


def _run_sobol(at: AppTest) -> None:
    sobol_idx = next(i for i, b in enumerate(at.button) if b.label == "Рассчитать чувствительность Sobol")
    at.button[sobol_idx].click().run(timeout=300)


def _first_info_with_text(at: AppTest, needle: str) -> str:
    for info in at.info:
        if needle in info.value:
            return info.value
    return ""


def _first_success_with_text(at: AppTest, needle: str) -> str:
    for succ in at.success:
        if needle in succ.value:
            return succ.value
    return ""


def test_monte_carlo_builds_metrics_and_details() -> None:
    at = AppTest.from_file(APP_FILE)
    at.run(timeout=300)
    _run_mc(at)

    metric_labels = [m.label for m in at.metric]
    assert "P5" in metric_labels
    assert "P95" in metric_labels
    assert any("Детализация столбца Monte Carlo" in m.value for m in at.markdown)
    assert len(at.error) == 0


def test_sobol_builds_tornado_and_details() -> None:
    at = AppTest.from_file(APP_FILE)
    at.run(timeout=300)
    _run_sobol(at)

    assert any("Детализация выбранного столбца Sobol" in m.value for m in at.markdown)
    assert _first_info_with_text(at, "Наибольшее влияние оказывает")
    assert len(at.error) == 0


def test_interpreter_changes_when_selected_bin_changes() -> None:
    at = AppTest.from_file(APP_FILE)
    at.run(timeout=300)
    _run_mc(at)

    bin_select = next(s for s in at.selectbox if s.label == "Диапазон прогноза (бин)")
    selected_before = _first_success_with_text(at, "Выбран столбец Monte Carlo")
    original_value = bin_select.value

    # Меняем выбор на другой бин и проверяем, что интерпретация обновилась.
    fallback = next(v for v in bin_select.options if v != original_value)
    bin_select.set_value(fallback).run(timeout=300)

    selected_after = _first_success_with_text(at, "Выбран столбец Monte Carlo")
    assert selected_before != ""
    assert selected_after != ""
    assert selected_before != selected_after
    assert len(at.error) == 0
