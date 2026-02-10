"""
Примеры использования пакета ts_analysis.

Разделы:
    1. Тесты на тренд (ts_analysis.trend)
    2. Тесты на mean reversion (ts_analysis.mean_reversion)
    3. Режимная диагностика (ts_analysis.regime)
    4. Тест Йохансена на коинтеграцию (ts_analysis.cointegration)
"""

import numpy as np
import pandas as pd
from pprint import pprint

# Импорт сводных функций через пакет
from ts_analysis import (
    trend_diagnostics,
    mean_reversion_summary,
    regime_diagnostics,
    johansen_test,
)

# Импорт отдельных функций при необходимости
from ts_analysis.trend import mann_kendall_test, spearman_trend
from ts_analysis.mean_reversion import adf_test, hurst_exponent, half_life
from ts_analysis.regime import variance_ratio, ar1_mean_reversion


def generate_test_data(n: int = 500, seed: int = 42):
    """Генерация синтетических данных для демонстрации."""
    rng = np.random.default_rng(seed)

    # 1. Стационарный AR(1) — mean reverting
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.7 * ar1[i - 1] + rng.normal()

    # 2. Random walk
    rw = np.cumsum(rng.normal(size=n))

    # 3. Линейный тренд + шум
    trend = 0.05 * np.arange(n) + rng.normal(0, 1, n)

    # 4. Два коинтегрированных ряда
    x = np.cumsum(rng.normal(size=n))
    y = 2 * x + rng.normal(size=n) * 0.1

    return {
        "ar1": ar1,
        "random_walk": rw,
        "trend": trend,
        "coint_x": x,
        "coint_y": y,
    }


# ──────────────────────────────────────────────────────────────────
# 1. ТЕСТЫ НА ТРЕНД
# ──────────────────────────────────────────────────────────────────
def example_trend():
    """Пример: диагностика тренда."""
    print("=" * 70)
    print("1. ТЕСТЫ НА ТРЕНД")
    print("=" * 70)

    data = generate_test_data()

    # --- Сводная диагностика ---
    print("\n--- Ряд с линейным трендом: trend_diagnostics() ---")
    result = trend_diagnostics(data["trend"])
    pprint(result)

    # --- Отдельные функции ---
    print("\n--- Random walk: отдельные тесты ---")
    print("Mann-Kendall:", mann_kendall_test(data["random_walk"]))
    print("Spearman:    ", spearman_trend(data["random_walk"]))


# ──────────────────────────────────────────────────────────────────
# 2. ТЕСТЫ НА MEAN REVERSION
# ──────────────────────────────────────────────────────────────────
def example_mean_reversion():
    """Пример: диагностика mean reversion."""
    print("\n" + "=" * 70)
    print("2. ТЕСТЫ НА MEAN REVERSION")
    print("=" * 70)

    data = generate_test_data()

    # --- Сводная диагностика ---
    print("\n--- AR(1) φ=0.7: mean_reversion_summary() ---")
    result = mean_reversion_summary(data["ar1"])
    # Выведем только ключи верхнего уровня и итоговые оценки
    print(f"  Scores:         {result['scores']}")
    print(f"  Total score:    {result['total_score']}/{result['max_score']}")
    print(f"  Final score:    {result['final_score']:.0%}")
    print(f"  Assessment:     {result['assessment']}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Hurst:          {result['hurst']['hurst_value']:.4f} "
          f"({result['hurst']['interpretation']})")
    print(f"  Half-life:      {result['half_life']['half_life']:.2f} periods")

    # --- Отдельные функции ---
    print("\n--- Random walk: отдельные тесты ---")
    print("ADF:  ", adf_test(data["random_walk"]))
    print("Hurst:", hurst_exponent(data["random_walk"]))
    print("HL:   ", half_life(data["random_walk"]))


# ──────────────────────────────────────────────────────────────────
# 3. РЕЖИМНАЯ ДИАГНОСТИКА
# ──────────────────────────────────────────────────────────────────
def example_regime():
    """Пример: режимная диагностика (тренд vs mean reversion)."""
    print("\n" + "=" * 70)
    print("3. РЕЖИМНАЯ ДИАГНОСТИКА")
    print("=" * 70)

    data = generate_test_data()

    # --- Сводная диагностика ---
    print("\n--- AR(1): regime_diagnostics() ---")
    result = regime_diagnostics(data["ar1"])
    print("Hints:")
    for h in result["hints"]:
        print(f"  - {h}")
    print(f"  AR(1) phi:       {result['ar1']['phi']:.4f}")
    print(f"  Hurst H:         {result['hurst_rs']['H']:.4f}")
    print(f"  ZA break index:  {result['zivot_andrews_ct']['break_index']}")
    print(f"  VR (k=5):        {result['variance_ratio'][1]['vr']:.4f}")

    # --- Отдельные функции ---
    print("\n--- Trend series: отдельные тесты ---")
    print("VR(k=10):", variance_ratio(data["trend"], k=10))
    print("AR(1):   ", ar1_mean_reversion(data["trend"]))


# ──────────────────────────────────────────────────────────────────
# 4. ТЕСТ ЙОХАНСЕНА НА КОИНТЕГРАЦИЮ
# ──────────────────────────────────────────────────────────────────
def example_cointegration():
    """Пример: тест Йохансена."""
    print("\n" + "=" * 70)
    print("4. ТЕСТ ЙОХАНСЕНА НА КОИНТЕГРАЦИЮ")
    print("=" * 70)

    data = generate_test_data()

    # --- Коинтегрированная пара ---
    print("\n--- Коинтегрированная пара (X, Y=2X+noise) ---")
    pair = pd.DataFrame({"X": data["coint_x"], "Y": data["coint_y"]})
    result = johansen_test(pair, det_order=0, k_ar_diff=1)
    print(f"  Eigenvalues:   {result['eigenvalues']}")
    print(f"  Cointegrated:  {result['conclusion']['is_cointegrated']}")
    print(f"  Vectors (rec): {result['conclusion']['recommended']}")
    print(f"  Assessment:    {result['conclusion']['assessment']}")
    print(f"  Vectors:       {result['cointegration_vectors']}")

    # --- Независимые ряды ---
    print("\n--- Независимые random walks ---")
    rng = np.random.default_rng(99)
    indep = pd.DataFrame({
        "A": np.cumsum(rng.normal(size=300)),
        "B": np.cumsum(rng.normal(size=300)),
    })
    result2 = johansen_test(indep, det_order=0)
    print(f"  Cointegrated:  {result2['conclusion']['is_cointegrated']}")
    print(f"  Assessment:    {result2['conclusion']['assessment']}")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_trend()
    example_mean_reversion()
    example_regime()
    example_cointegration()
