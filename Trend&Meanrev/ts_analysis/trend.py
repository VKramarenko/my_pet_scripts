"""
Модуль тестов на наличие тренда во временном ряде.

Функции:
    linear_trend_hac  — OLS с HAC-ошибками (Newey-West)
    mann_kendall_test  — непараметрический тест Манна-Кендалла
    spearman_trend     — ранговая корреляция Спирмена с временным индексом
    cox_stuart_test    — тест Кокса-Стюарта (знаковый тест на парах)
    adf_test           — ADF-тест (regression='ct', H0: unit root)
    kpss_test          — KPSS-тест (regression='ct', H0: тренд-стационарность)
    trend_diagnostics  — запускает все тесты и даёт интерпретацию
"""

import numpy as np
from typing import Dict, Any, Optional

from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss


def _to_1d_array(y):
    """Приведение входных данных к чистому 1-D numpy-массиву без NaN."""
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[~np.isnan(y)]
    if y.size < 8:
        raise ValueError("Слишком мало наблюдений после удаления NaN (нужно хотя бы ~8-10).")
    return y


def linear_trend_hac(y, lags: Optional[int] = None) -> Dict[str, Any]:
    """
    OLS y ~ 1 + t с HAC(Newey-West) стандартными ошибками.

    Возвращает:
        slope   — оценка наклона
        pvalue  — p-value для H0: slope=0
        hac_lags — количество лагов HAC
        r2      — R²
    """
    y = _to_1d_array(y)
    t = np.arange(len(y), dtype=float)
    X = sm.add_constant(t)
    model = sm.OLS(y, X).fit()

    if lags is None:
        n = len(y)
        lags = max(1, int(np.floor(4 * (n / 100.0) ** (2 / 9))))

    robust = model.get_robustcov_results(cov_type="HAC", maxlags=lags)
    return {
        "slope": float(robust.params[1]),
        "pvalue": float(robust.pvalues[1]),
        "hac_lags": int(lags),
        "r2": float(model.rsquared),
    }


def mann_kendall_test(y) -> Dict[str, Any]:
    """
    Mann–Kendall тест на монотонный тренд.

    Возвращает:
        S      — статистика S
        Z      — нормальная аппроксимация
        pvalue — двусторонний p-value
        trend  — "increasing" / "decreasing" / "no trend"
    """
    y = _to_1d_array(y)
    n = len(y)

    S = 0
    for i in range(n - 1):
        S += np.sum(np.sign(y[i + 1:] - y[i]))

    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    varS = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if varS == 0:
        return {"S": int(S), "Z": 0.0, "pvalue": 1.0, "trend": "no trend"}

    if S > 0:
        Z = (S - 1) / np.sqrt(varS)
    elif S < 0:
        Z = (S + 1) / np.sqrt(varS)
    else:
        Z = 0.0

    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    trend = (
        "increasing" if (p < 0.05 and Z > 0) else
        "decreasing" if (p < 0.05 and Z < 0) else
        "no trend"
    )
    return {"S": int(S), "Z": float(Z), "pvalue": float(p), "trend": trend}


def spearman_trend(y) -> Dict[str, Any]:
    """
    Ранговая корреляция Спирмена между рядом и временным индексом.

    Возвращает:
        rho    — коэффициент корреляции
        pvalue — p-value
        trend  — "increasing" / "decreasing" / "no trend"
    """
    y = _to_1d_array(y)
    t = np.arange(len(y))
    rho, p = stats.spearmanr(t, y)
    direction = (
        "increasing" if (p < 0.05 and rho > 0) else
        "decreasing" if (p < 0.05 and rho < 0) else
        "no trend"
    )
    return {"rho": float(rho), "pvalue": float(p), "trend": direction}


def cox_stuart_test(y) -> Dict[str, Any]:
    """
    Cox–Stuart тест на тренд (знаковый тест на парных половинах).

    Возвращает:
        pos    — количество положительных разностей
        neg    — количество отрицательных разностей
        pvalue — двусторонний p-value
        trend  — "increasing" / "decreasing" / "no trend"
    """
    y = _to_1d_array(y)
    n = len(y)
    half = n // 2
    d = y[-half:] - y[:half]

    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    m = pos + neg
    if m == 0:
        return {"pos": pos, "neg": neg, "pvalue": 1.0, "trend": "no trend"}

    k = min(pos, neg)
    p = min(1.0, float(2 * stats.binom.cdf(k, m, 0.5)))

    trend = ("increasing" if pos > neg else "decreasing") if p < 0.05 else "no trend"
    return {"pos": pos, "neg": neg, "pvalue": p, "trend": trend}


def adf_test(y, autolag: str = "AIC") -> Dict[str, Any]:
    """
    ADF-тест с константой и трендом (regression='ct').
    H0: unit root (стохастический тренд).

    Возвращает:
        stat, pvalue, usedlag, nobs, critical_values, icbest
    """
    y = _to_1d_array(y)
    res = adfuller(y, regression="ct", autolag=autolag)
    stat, pvalue, usedlag, nobs, crit, icbest = res
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "usedlag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "icbest": float(icbest) if icbest is not None else None,
    }


def kpss_test(y, nlags: str = "auto") -> Dict[str, Any]:
    """
    KPSS-тест на тренд-стационарность (regression='ct').
    H0: тренд-стационарность.

    Возвращает:
        stat, pvalue, lags, critical_values
    """
    y = _to_1d_array(y)
    stat, pvalue, lags, crit = kpss(y, regression="ct", nlags=nlags)
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "lags": int(lags),
        "critical_values": {k: float(v) for k, v in crit.items()},
    }


def trend_diagnostics(y, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Запускает полный набор тестов на тренд и даёт интерпретацию.

    Возвращает dict с ключами:
        linear_trend_hac, mann_kendall, spearman, cox_stuart,
        adf_ct, kpss_ct, interpretation_adf_kpss
    """
    out = {
        "linear_trend_hac": linear_trend_hac(y),
        "mann_kendall": mann_kendall_test(y),
        "spearman": spearman_trend(y),
        "cox_stuart": cox_stuart_test(y),
        "adf_ct": adf_test(y),
        "kpss_ct": kpss_test(y),
    }

    adf_p = out["adf_ct"]["pvalue"]
    kpss_p = out["kpss_ct"]["pvalue"]

    if adf_p < alpha and kpss_p >= alpha:
        interpretation = ("Похоже на (тренд-)стационарный ряд: unit root отвергается, "
                          "KPSS тренд-стационарность не отвергает.")
    elif adf_p >= alpha and kpss_p < alpha:
        interpretation = ("Похоже на unit root / стохастический тренд: ADF не отвергает "
                          "unit root, KPSS отвергает тренд-стационарность.")
    elif adf_p < alpha and kpss_p < alpha:
        interpretation = ("Смешанный сигнал: ADF отвергает unit root, но KPSS тоже "
                          "отвергает тренд-стационарность (возможны структурные "
                          "сдвиги/сезонность/нелинейности).")
    else:
        interpretation = ("Слабый сигнал: ADF не отвергает unit root и KPSS не отвергает "
                          "тренд-стационарность (возможна малая мощность тестов или "
                          "короткий ряд).")

    out["interpretation_adf_kpss"] = interpretation
    return out
