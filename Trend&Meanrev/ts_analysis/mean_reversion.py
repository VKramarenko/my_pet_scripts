"""
Модуль тестов на mean reversion (возврат к среднему).

Функции:
    adf_test                — ADF-тест на стационарность (regression='c')
    kpss_test               — KPSS-тест на стационарность
    hurst_exponent          — экспонента Херста (R/S метод)
    half_life               — период полураспада через AR(1)
    autocorrelation_analysis — ACF/PACF + Ljung-Box + ARCH
    mean_reversion_summary  — сводная диагностика со скорами
"""

import numpy as np
import warnings
from typing import Dict, Any

from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss as _kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")


def _to_1d_array(y):
    """Приведение входных данных к чистому 1-D numpy-массиву без NaN."""
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[~np.isnan(y)]
    if y.size < 8:
        raise ValueError("Слишком мало наблюдений после удаления NaN (нужно хотя бы ~8-10).")
    return y


def adf_test(series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    ADF-тест на стационарность (regression='c', H0: нестационарный).

    Возвращает:
        test_statistic, p_value, used_lags, n_obs,
        critical_values, is_stationary, interpretation
    """
    series = _to_1d_array(series)
    result = adfuller(series, autolag="AIC")

    return {
        "test_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lags": int(result[2]),
        "n_obs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": result[1] < alpha,
        "interpretation": "Стационарный" if result[1] < alpha else "Нестационарный",
    }


def kpss_test(series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    KPSS-тест на стационарность (regression='c', H0: стационарный).

    Возвращает dict с ключами 'c' и 'ct' — результаты для двух
    спецификаций (константа / константа+тренд).
    """
    series = _to_1d_array(series)
    output = {}

    for regression in ("c", "ct"):
        try:
            result = _kpss(series, regression=regression, nlags="auto")
            output[regression] = {
                "test_statistic": float(result[0]),
                "p_value": float(result[1]),
                "used_lags": int(result[2]),
                "critical_values": {k: float(v) for k, v in result[3].items()},
                "is_stationary": result[1] >= alpha,
                "interpretation": (
                    "Стационарный" if result[1] >= alpha else "Нестационарный"
                ),
            }
        except Exception as e:
            output[regression] = {"error": str(e)}

    return output


def hurst_exponent(series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Экспонента Херста (R/S метод).

    H < 0.5 → mean reversion, H ≈ 0.5 → random walk, H > 0.5 → trending.

    Возвращает:
        hurst_value, confidence_interval, r_squared, p_value,
        interpretation, is_mean_reverting, n_lags_used
    """
    series = _to_1d_array(series)
    n = len(series)

    def _calculate_rs(data, lag):
        k = n // lag
        rs_values = []
        for i in range(k):
            block = data[i * lag : (i + 1) * lag]
            if len(block) < 2:
                continue
            centered = block - np.mean(block)
            cumulative = np.cumsum(centered)
            R = np.max(cumulative) - np.min(cumulative)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_values.append(R / S)
        return np.mean(rs_values) if rs_values else np.nan

    min_lag = 10
    lags_seq = np.unique(
        np.logspace(np.log10(min_lag), np.log10(n // 2), 30).astype(int)
    )

    lags, rs_values = [], []
    for lag in lags_seq:
        if lag >= min_lag:
            rs = _calculate_rs(series, lag)
            if not np.isnan(rs):
                lags.append(lag)
                rs_values.append(rs)

    if len(lags) < 5:
        raise ValueError("Недостаточно данных для расчёта экспоненты Херста.")

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    slope, _, r_value, p_value, std_err = stats.linregress(log_lags, log_rs)
    hurst = slope

    t_critical = stats.t.ppf(1 - alpha / 2, len(lags) - 2)
    ci = (hurst - t_critical * std_err, hurst + t_critical * std_err)

    if hurst < 0.4:
        interpretation = "Strong Mean Reversion"
    elif hurst < 0.5:
        interpretation = "Mean Reverting"
    elif abs(hurst - 0.5) < 1e-6:
        interpretation = "Random Walk"
    elif hurst < 0.6:
        interpretation = "Weak Persistence"
    else:
        interpretation = "Strong Persistence"

    return {
        "hurst_value": float(hurst),
        "confidence_interval": (float(ci[0]), float(ci[1])),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "interpretation": interpretation,
        "is_mean_reverting": hurst < 0.5,
        "n_lags_used": len(lags),
    }


def half_life(series) -> Dict[str, Any]:
    """
    Период полураспада через AR(1): x_t = c + phi * x_{t-1} + e_t.
    half_life = -ln(2) / ln(phi).

    Возвращает:
        phi, phi_std_error, half_life, interpretation,
        is_stationary, residual_variance
    """
    series = _to_1d_array(series)
    y = series[1:]
    y_lag = series[:-1]
    X = np.column_stack([np.ones_like(y_lag), y_lag])

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = beta[1]

        if phi >= 1:
            hl = float("inf")
            interp = "Non-stationary"
        elif phi <= 0:
            hl = 0.0
            interp = "Instant mean reversion"
        else:
            hl = -np.log(2) / np.log(phi)
            interp = f"Half-life: {hl:.2f} periods"

        residuals = y - X @ beta
        sigma2 = float(np.sum(residuals ** 2) / (len(y) - 2))
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        phi_se = float(np.sqrt(var_beta[1, 1]))

        return {
            "phi": float(phi),
            "phi_std_error": phi_se,
            "half_life": float(hl),
            "interpretation": interp,
            "is_stationary": phi < 1,
            "residual_variance": sigma2,
        }
    except Exception as e:
        return {
            "phi": float("nan"),
            "phi_std_error": float("nan"),
            "half_life": float("nan"),
            "interpretation": f"Error: {e}",
            "is_stationary": False,
            "residual_variance": float("nan"),
        }


def autocorrelation_analysis(series, max_lags: int = 40,
                              alpha: float = 0.05) -> Dict[str, Any]:
    """
    Анализ автокорреляции: ACF, PACF, Ljung-Box, ARCH-тест.

    Возвращает:
        acf_values, pacf_values, acf_significant_lags, pacf_significant_lags,
        acf_first_lag, pacf_first_lag, ljung_box_pvalues,
        arch_pvalue, has_autocorrelation, has_arch_effects
    """
    series = _to_1d_array(series)
    n = len(series)
    max_lags = min(max_lags, n // 2)

    acf_vals, acf_ci = acf(series, nlags=max_lags, alpha=alpha, fft=True)
    pacf_vals, pacf_ci = pacf(series, nlags=max_lags, alpha=alpha)

    acf_sig, pacf_sig = [], []
    for lag in range(1, min(20, max_lags) + 1):
        acf_sig.append(not (acf_ci[lag][0] <= 0 <= acf_ci[lag][1]))
        pacf_sig.append(not (pacf_ci[lag][0] <= 0 <= pacf_ci[lag][1]))

    lb = acorr_ljungbox(series, lags=[5, 10, 20], return_df=True)

    try:
        arch_p = float(het_arch(series)[1])
    except Exception:
        arch_p = None

    return {
        "acf_values": acf_vals[:21].tolist(),
        "pacf_values": pacf_vals[:21].tolist(),
        "acf_significant_lags": [i for i, s in enumerate(acf_sig, 1) if s],
        "pacf_significant_lags": [i for i, s in enumerate(pacf_sig, 1) if s],
        "acf_first_lag": float(acf_vals[1]) if len(acf_vals) > 1 else None,
        "pacf_first_lag": float(pacf_vals[1]) if len(pacf_vals) > 1 else None,
        "ljung_box_pvalues": lb["lb_pvalue"].tolist(),
        "arch_pvalue": arch_p,
        "has_autocorrelation": any(lb["lb_pvalue"] < alpha),
        "has_arch_effects": arch_p is not None and arch_p < alpha,
    }


def mean_reversion_summary(series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Сводная диагностика mean reversion: запускает все тесты,
    подсчитывает баллы и даёт итоговую оценку.

    Возвращает dict с ключами:
        adf, kpss, hurst, half_life, autocorrelation,
        scores, total_score, max_score, final_score,
        assessment, recommendation
    """
    results = {
        "adf": adf_test(series, alpha),
        "kpss": kpss_test(series, alpha),
        "hurst": hurst_exponent(series, alpha),
        "half_life": half_life(series),
        "autocorrelation": autocorrelation_analysis(series, alpha=alpha),
    }

    # Подсчёт баллов
    scores = {}
    scores["adf"] = 1 if results["adf"]["is_stationary"] else 0

    kpss_c = results["kpss"].get("c", {})
    scores["kpss"] = 1 if kpss_c.get("is_stationary", False) else 0

    scores["hurst"] = 1 if results["hurst"]["is_mean_reverting"] else 0

    acf1 = results["autocorrelation"]["acf_first_lag"]
    scores["acf"] = 1 if (acf1 is not None and acf1 < 0) else 0

    hl = results["half_life"]["half_life"]
    scores["half_life"] = 1 if (0 < hl < 100) else 0

    total = sum(scores.values())
    max_score = len(scores)
    final = total / max_score

    if final >= 0.8:
        assessment = "HIGH probability of mean reversion"
        recommendation = "Suitable for mean reversion strategies"
    elif final >= 0.6:
        assessment = "MODERATE probability of mean reversion"
        recommendation = "Consider mean reversion strategies with caution"
    elif final >= 0.4:
        assessment = "LOW probability of mean reversion"
        recommendation = "Not ideal for mean reversion strategies"
    else:
        assessment = "UNLIKELY to have mean reversion"
        recommendation = "Avoid mean reversion strategies"

    results["scores"] = scores
    results["total_score"] = total
    results["max_score"] = max_score
    results["final_score"] = final
    results["assessment"] = assessment
    results["recommendation"] = recommendation

    return results
