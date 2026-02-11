"""
Примеры использования пакета ts_analysis.

Все примеры построены на случайном блуждании как базовом процессе:
  - Раздел 1: ряды с явным трендом (дрифт, детерминированный тренд, нелинейный)
  - Раздел 2: ряды с возвращением к среднему (Ornstein-Uhlenbeck, быстрый/медленный AR)
  - Раздел 3: режимная диагностика — сравнение трендовых и mean-reverting рядов
  - Раздел 4: коинтеграция (пара с общим трендом, тройка, независимые ряды)
"""

import sys
import io
import numpy as np
import pandas as pd

# Корректный вывод UTF-8 на Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from ts_analysis import (
    trend_diagnostics,
    mean_reversion_summary,
    regime_diagnostics,
    johansen_test,
)


# ══════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ДАННЫХ
# ══════════════════════════════════════════════════════════════════════

def generate_random_walk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Чистое случайное блуждание (базовый процесс)."""
    return np.cumsum(rng.normal(size=n))


def generate_rw_with_drift(n: int, rng: np.random.Generator,
                           drift: float = 0.05) -> np.ndarray:
    """Случайное блуждание с дрифтом: X_t = X_{t-1} + μ + ε_t.

    Моделирует стохастический тренд — типичная модель для цен акций.
    """
    increments = drift + rng.normal(size=n)
    return np.cumsum(increments)


def generate_deterministic_trend_rw(n: int, rng: np.random.Generator,
                                    slope: float = 0.1) -> np.ndarray:
    """Детерминированный линейный тренд + случайное блуждание.

    Сочетание: y_t = slope * t + RW_t.  Оба компонента нестационарны,
    но тренд здесь детерминированный.
    """
    rw = np.cumsum(rng.normal(size=n))
    return slope * np.arange(n) + rw


def generate_quadratic_trend_rw(n: int, rng: np.random.Generator,
                                a: float = 0.0002,
                                b: float = 0.02) -> np.ndarray:
    """Нелинейный (квадратичный) тренд + случайное блуждание.

    y_t = a * t² + b * t + RW_t.  Тесты на линейный тренд могут пропускать
    ускоряющийся рост — хороший стресс-тест для пакета.
    """
    t = np.arange(n)
    rw = np.cumsum(rng.normal(size=n))
    return a * t ** 2 + b * t + rw


def generate_ornstein_uhlenbeck(n: int, rng: np.random.Generator,
                                theta: float = 0.15, mu: float = 0.0,
                                sigma: float = 1.0) -> np.ndarray:
    """Процесс Орнштейна-Уленбека (дискретная аппроксимация).

    dX = θ(μ - X)dt + σdW.  Классический mean-reverting процесс.
    theta — скорость притяжения к среднему mu.
    """
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.normal()
    return x


def generate_ar1(n: int, rng: np.random.Generator,
                 phi: float = 0.5, c: float = 0.0) -> np.ndarray:
    """AR(1) процесс: X_t = c + φ * X_{t-1} + ε_t.

    |φ| < 1 → стационарный (mean-reverting).
    Чем меньше φ, тем быстрее возвращение к среднему.
    """
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = c + phi * x[i - 1] + rng.normal()
    return x


def generate_cointegrated_pair(n: int, rng: np.random.Generator,
                               beta: float = 1.5,
                               spread_std: float = 0.5) -> tuple:
    """Два коинтегрированных ряда с общим стохастическим трендом.

    X_t — random walk (общий фактор).
    Y_t = β * X_t + ε_t, где ε — стационарный шум (спред).
    Спред Y - β*X стационарен → коинтеграция.
    """
    common_trend = np.cumsum(rng.normal(size=n))
    spread = generate_ar1(n, rng, phi=0.7, c=0.0) * spread_std
    x = common_trend + rng.normal(size=n) * 0.3
    y = beta * common_trend + spread
    return x, y


def generate_cointegrated_triple(n: int, rng: np.random.Generator) -> tuple:
    """Три ряда с 1 коинтеграционным соотношением.

    Общий фактор F_t — random walk.
    X1 = F + шум, X2 = 2F + шум, X3 = 3F + шум.
    Соотношение: 3*X1 - X3 ≈ стационарный.
    """
    factor = np.cumsum(rng.normal(size=n))
    x1 = factor + rng.normal(size=n) * 0.4
    x2 = 2 * factor + rng.normal(size=n) * 0.4
    x3 = 3 * factor + rng.normal(size=n) * 0.4
    return x1, x2, x3


# ══════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ВЫВОДА
# ══════════════════════════════════════════════════════════════════════

def section(title: str, level: int = 1):
    """Красивый заголовок секции."""
    if level == 1:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    else:
        print(f"\n--- {title} ---")


# ══════════════════════════════════════════════════════════════════════
# 1. ТЕСТЫ НА ТРЕНД
# ══════════════════════════════════════════════════════════════════════

def example_trend():
    """Пример: диагностика тренда на 4 рядах.

    Ожидания:
    - RW с дрифтом: тренд + unit root (стохастический тренд)
    - Детерминированный тренд + RW: тренд обнаружен
    - Квадратичный тренд + RW: линейные тесты ловят частично
    - Чистый RW (контроль): нет значимого тренда
    """
    section("1. ТЕСТЫ НА ТРЕНД")
    rng = np.random.default_rng(42)
    n = 500

    series = {
        "RW с дрифтом (μ=0.05)":      generate_rw_with_drift(n, rng, drift=0.05),
        "Детерм. тренд + RW":         generate_deterministic_trend_rw(n, rng, slope=0.1),
        "Квадратичный тренд + RW":    generate_quadratic_trend_rw(n, rng),
        "Чистый RW (контроль)":       generate_random_walk(n, rng),
    }

    for name, y in series.items():
        section(name, level=2)
        result = trend_diagnostics(y)

        print(f"  ADF  p-value:   {result['adf_ct']['pvalue']:.4f}")
        print(f"  KPSS p-value:   {result['kpss_ct']['pvalue']:.4f}")
        print(f"  Mann-Kendall:   {result['mann_kendall']['trend']} "
              f"(p={result['mann_kendall']['pvalue']:.4f})")
        print(f"  Spearman:       {result['spearman']['trend']} "
              f"(rho={result['spearman']['rho']:.4f})")
        print(f"  Cox-Stuart:     {result['cox_stuart']['trend']} "
              f"(p={result['cox_stuart']['pvalue']:.4f})")
        print(f"  Slope (HAC):    {result['linear_trend_hac']['slope']:.6f} "
              f"(p={result['linear_trend_hac']['pvalue']:.4f})")
        print(f"  >> Интерпретация: {result['interpretation_adf_kpss']}")


# ══════════════════════════════════════════════════════════════════════
# 2. ТЕСТЫ НА MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════

def example_mean_reversion():
    """Пример: диагностика mean reversion на 4 рядах.

    Ожидания:
    - OU (θ=0.15): сильное возвращение к среднему, высокий score
    - AR(1) φ=0.5: быстрое возвращение, высокий score
    - AR(1) φ=0.9: медленное возвращение, пограничный случай
    - Чистый RW (контроль): score низкий, UNLIKELY
    """
    section("2. ТЕСТЫ НА MEAN REVERSION")
    rng = np.random.default_rng(42)
    n = 500

    series = {
        "Ornstein-Uhlenbeck (θ=0.15)": generate_ornstein_uhlenbeck(n, rng, theta=0.15),
        "AR(1) φ=0.5 (быстрый MR)":    generate_ar1(n, rng, phi=0.5),
        "AR(1) φ=0.9 (медленный MR)":  generate_ar1(n, rng, phi=0.9),
        "Чистый RW (контроль)":         generate_random_walk(n, rng),
    }

    for name, y in series.items():
        section(name, level=2)
        result = mean_reversion_summary(y)

        print(f"  Score:          {result['final_score']:.0%} "
              f"({result['total_score']}/{result['max_score']})")
        print(f"  Assessment:     {result['assessment']}")
        print(f"  Hurst:          {result['hurst']['hurst_value']:.4f} "
              f"({result['hurst']['interpretation']})")
        print(f"  Half-life:      {result['half_life']['half_life']:.2f} periods")
        print(f"  ADF stationary: {result['adf']['is_stationary']}")
        print(f"  Scores:         {result['scores']}")
        print(f"  Recommendation: {result['recommendation']}")


# ══════════════════════════════════════════════════════════════════════
# 3. РЕЖИМНАЯ ДИАГНОСТИКА
# ══════════════════════════════════════════════════════════════════════

def example_regime():
    """Пример: режимная диагностика — сравнение трендового и MR рядов.

    Variance ratio, Hurst, AR(1), Zivot-Andrews, CUSUM.
    Показывает, как один и тот же набор тестов различает режимы.
    """
    section("3. РЕЖИМНАЯ ДИАГНОСТИКА")
    rng = np.random.default_rng(42)
    n = 500

    series = {
        "RW с дрифтом (тренд)":         generate_rw_with_drift(n, rng, drift=0.05),
        "OU (mean reversion)":           generate_ornstein_uhlenbeck(n, rng, theta=0.15),
        "AR(1) φ=0.9 (пограничный)":    generate_ar1(n, rng, phi=0.9),
    }

    for name, y in series.items():
        section(name, level=2)
        result = regime_diagnostics(y)

        print(f"  AR(1) φ:       {result['ar1']['phi']:.4f} "
              f"(p={result['ar1']['pvalue_phi_eq_1']:.4f})")
        print(f"  Hurst H:       {result['hurst_rs']['H']:.4f} "
              f"({result['hurst_rs']['hint']})")
        print(f"  ZA break:      index={result['zivot_andrews_ct']['break_index']} "
              f"(p={result['zivot_andrews_ct']['pvalue']:.4f})")

        vrs = result['variance_ratio']
        vr_str = ", ".join(f"VR({v['k']})={v['vr']:.3f}" for v in vrs)
        print(f"  VR:            {vr_str}")

        print("  Hints:")
        for h in result["hints"]:
            print(f"    • {h}")


# ══════════════════════════════════════════════════════════════════════
# 4. ТЕСТ ЙОХАНСЕНА НА КОИНТЕГРАЦИЮ
# ══════════════════════════════════════════════════════════════════════

def example_cointegration():
    """Пример: тест Йохансена на 3 сценариях.

    - Коинтегрированная пара (Y ≈ 1.5*X + стационарный шум)
    - Три ряда с общим фактором (1 коинтеграционный вектор)
    - Независимые random walks (нет коинтеграции)
    """
    section("4. ТЕСТ ЙОХАНСЕНА НА КОИНТЕГРАЦИЮ")
    rng = np.random.default_rng(42)
    n = 500

    # --- Коинтегрированная пара ---
    section("Коинтегрированная пара (Y ≈ 1.5·X + spread)", level=2)
    x, y = generate_cointegrated_pair(n, rng, beta=1.5, spread_std=0.5)
    pair = pd.DataFrame({"X": x, "Y": y})
    result = johansen_test(pair, det_order=0, k_ar_diff=1)

    print(f"  Коинтегрирован: {result['conclusion']['is_cointegrated']}")
    print(f"  Векторов:       {result['conclusion']['recommended']}")
    print(f"  Assessment:     {result['conclusion']['assessment']}")
    print(f"  Eigenvalues:    {np.round(result['eigenvalues'], 4)}")
    print(f"  Векторы:")
    for i, vec in enumerate(result['cointegration_vectors']):
        print(f"    v{i+1}: {np.round(vec, 4)}")

    # --- Тройка с общим фактором ---
    section("Тройка с общим фактором (1 коинтегр. вектор)", level=2)
    x1, x2, x3 = generate_cointegrated_triple(n, rng)
    triple = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})
    result2 = johansen_test(triple, det_order=0, k_ar_diff=1)

    print(f"  Коинтегрирован: {result2['conclusion']['is_cointegrated']}")
    print(f"  Векторов:       {result2['conclusion']['recommended']}")
    print(f"  Assessment:     {result2['conclusion']['assessment']}")
    print(f"  Eigenvalues:    {np.round(result2['eigenvalues'], 4)}")

    # --- Независимые random walks (контроль) ---
    section("Независимые random walks (контроль)", level=2)
    indep = pd.DataFrame({
        "A": generate_random_walk(n, rng),
        "B": generate_random_walk(n, rng),
        "C": generate_random_walk(n, rng),
    })
    result3 = johansen_test(indep, det_order=0, k_ar_diff=1)

    print(f"  Коинтегрирован: {result3['conclusion']['is_cointegrated']}")
    print(f"  Assessment:     {result3['conclusion']['assessment']}")


# ══════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    example_trend()
    example_mean_reversion()
    example_regime()
    example_cointegration()
