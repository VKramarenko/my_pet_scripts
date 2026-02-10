"""
Модуль режимной диагностики: тренд vs mean reversion.

Функции:
    variance_ratio       — Variance Ratio тест (Lo–MacKinlay)
    ar1_mean_reversion   — AR(1) с half-life и тестом phi=1
    hurst_rs             — экспонента Херста (R/S)
    zivot_andrews_break  — Zivot–Andrews unit root тест со структурным сдвигом
    cusum_break_on_trend — CUSUM-тест стабильности параметров тренда
    markov_switching_ar1 — Markov Switching AR(1) с переключением режимов
    regime_diagnostics   — сводная диагностика
"""

import numpy as np
from typing import Dict, Any, Sequence

from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import zivot_andrews
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _to_1d_array(y):
    """Приведение входных данных к чистому 1-D numpy-массиву без NaN."""
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[~np.isnan(y)]
    if y.size < 30:
        raise ValueError("Слишком мало наблюдений (желательно 30+ для этих проверок).")
    return y


def variance_ratio(y, k: int = 5, log: bool = False) -> Dict[str, Any]:
    """
    Variance ratio test: VR(k) = Var(delta_k y) / (k * Var(delta_1 y)).

    VR ~ 1 → random walk, VR > 1 → momentum/trending, VR < 1 → mean reversion.

    Возвращает:
        k, vr, z_approx, pvalue_approx, hint
    """
    y = _to_1d_array(y)
    if log:
        y = np.log(y)

    dy1 = np.diff(y, 1)
    dyk = y[k:] - y[:-k]

    v1 = np.var(dy1, ddof=1)
    vk = np.var(dyk, ddof=1)
    vr = vk / (k * v1) if v1 > 0 else np.nan

    z = (
        (vr - 1.0) / np.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k * len(dy1)))
        if np.isfinite(vr)
        else np.nan
    )
    p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

    hint = "random_walk_like"
    if np.isfinite(vr):
        if vr < 1:
            hint = "mean_reversion_like"
        elif vr > 1:
            hint = "momentum_trend_like"

    return {
        "k": int(k),
        "vr": float(vr),
        "z_approx": float(z),
        "pvalue_approx": float(p),
        "hint": hint,
    }


def ar1_mean_reversion(y) -> Dict[str, Any]:
    """
    AR(1): x_t = c + phi * x_{t-1} + e_t.
    Тест H0: phi = 1 (random walk).

    Возвращает:
        phi, phi_se, t_phi_eq_1, pvalue_phi_eq_1, half_life_steps, r2
    """
    y = _to_1d_array(y)

    x = y[1:]
    x_lag = y[:-1]
    X = sm.add_constant(x_lag)
    res = sm.OLS(x, X).fit()

    phi = res.params[1]
    se_phi = res.bse[1]

    t_stat = (phi - 1.0) / se_phi if se_phi > 0 else np.nan
    p = (
        2 * (1 - stats.t.cdf(abs(t_stat), df=res.df_resid))
        if np.isfinite(t_stat)
        else np.nan
    )

    hl = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan

    return {
        "phi": float(phi),
        "phi_se": float(se_phi),
        "t_phi_eq_1": float(t_stat),
        "pvalue_phi_eq_1": float(p),
        "half_life_steps": float(hl),
        "r2": float(res.rsquared),
    }


def hurst_rs(y) -> Dict[str, Any]:
    """
    R/S-оценка экспоненты Херста (индикативная).

    H < 0.5 → mean reversion, H ≈ 0.5 → random walk, H > 0.5 → trending.

    Возвращает:
        H, r2, pvalue, hint
    """
    y = _to_1d_array(y)
    y = y - np.mean(y)
    n = len(y)

    scales = np.unique(np.logspace(np.log10(10), np.log10(n // 2), 12).astype(int))
    rs = []

    for s in scales:
        m = n // s
        if m < 2:
            continue
        chunks = y[: m * s].reshape(m, s)
        r_s = []
        for c in chunks:
            z = np.cumsum(c - np.mean(c))
            R = np.max(z) - np.min(z)
            S = np.std(c, ddof=1)
            if S > 0:
                r_s.append(R / S)
        if r_s:
            rs.append((s, np.mean(r_s)))

    if len(rs) < 5:
        return {"H": float("nan"), "note": "Недостаточно данных/масштабов."}

    s_vals = np.array([a for a, _ in rs], dtype=float)
    rs_vals = np.array([b for _, b in rs], dtype=float)

    slope, _, r, p, _ = stats.linregress(np.log(s_vals), np.log(rs_vals))

    hint = "random_walk_like"
    if np.isfinite(slope):
        if slope < 0.5:
            hint = "mean_reversion_like"
        elif slope > 0.5:
            hint = "trend_like"

    return {"H": float(slope), "r2": float(r ** 2), "pvalue": float(p), "hint": hint}


def zivot_andrews_break(y, regression: str = "ct") -> Dict[str, Any]:
    """
    Zivot–Andrews unit root тест с одним структурным сдвигом.
    H0: unit root, H1: стационарный с одним break.

    Возвращает:
        stat, pvalue, break_index, critical_values, regression
    """
    y = _to_1d_array(y)
    za_result = zivot_andrews(y, regression=regression, autolag="AIC")
    za_stat, za_p, za_crit = za_result[0], za_result[1], za_result[2]
    bp = za_result[-1]  # break point — последний элемент
    return {
        "stat": float(za_stat),
        "pvalue": float(za_p),
        "break_index": int(bp),
        "critical_values": {k: float(v) for k, v in za_crit.items()},
        "regression": regression,
    }


def cusum_break_on_trend(y) -> Dict[str, Any]:
    """
    CUSUM-тест стабильности параметров линейного тренда (OLS y ~ 1 + t).

    Низкий p-value → параметры нестабильны → возможна смена режима.

    Возвращает:
        stat, pvalue, critical_values
    """
    y = _to_1d_array(y)
    t = np.arange(len(y), dtype=float)
    X = sm.add_constant(t)
    fit = sm.OLS(y, X).fit()
    stat, pvalue, crit = breaks_cusumolsresid(
        fit.resid, ddof=int(fit.df_model) + 1
    )
    return {
        "stat": float(stat),
        "pvalue": float(pvalue),
        "critical_values": {str(k): float(v) for k, v in crit},
    }


def markov_switching_ar1(y, k_regimes: int = 2,
                          switching_variance: bool = True) -> Dict[str, Any]:
    """
    Markov Switching AR(1): режимы с разными phi и/или дисперсией.

    Возвращает:
        aic, bic, llf, phi_by_regime, smoothed_probs
    """
    y = _to_1d_array(y)

    model = MarkovRegression(
        y,
        k_regimes=k_regimes,
        trend="c",
        order=1,
        switching_variance=switching_variance,
    )
    res = model.fit(disp=False)

    probs = np.asarray(res.smoothed_marginal_probabilities)

    phi_by_regime = {}
    for r in range(k_regimes):
        candidates = [
            i
            for i, name in enumerate(res.param_names)
            if "ar.L1" in name and f"[{r}]" in name
        ]
        if candidates:
            phi_by_regime[str(r)] = float(res.params[candidates[0]])

    return {
        "aic": float(res.aic),
        "bic": float(res.bic),
        "llf": float(res.llf),
        "phi_by_regime": phi_by_regime,
        "smoothed_probs_shape": list(probs.shape),
    }


def regime_diagnostics(y, alpha: float = 0.05,
                        vr_ks: Sequence[int] = (2, 5, 10, 20)) -> Dict[str, Any]:
    """
    Сводная режимная диагностика: VR (несколько горизонтов), AR(1),
    Hurst, Zivot-Andrews, CUSUM, Markov Switching AR(1).

    Возвращает dict с ключами:
        ar1, hurst_rs, zivot_andrews_ct, cusum_trend,
        variance_ratio, markov_switching_ar1, hints
    """
    out: Dict[str, Any] = {
        "ar1": ar1_mean_reversion(y),
        "hurst_rs": hurst_rs(y),
        "zivot_andrews_ct": zivot_andrews_break(y, regression="ct"),
        "cusum_trend": cusum_break_on_trend(y),
        "variance_ratio": [variance_ratio(y, k=int(k)) for k in vr_ks],
    }

    try:
        out["markov_switching_ar1"] = markov_switching_ar1(y)
    except Exception as e:
        out["markov_switching_ar1"] = {"error": str(e)}

    # Текстовые подсказки
    hints = []

    vrs = [d["vr"] for d in out["variance_ratio"] if np.isfinite(d["vr"])]
    if vrs:
        mvr = float(np.mean(vrs))
        if mvr < 0.98:
            hints.append("VR<1 в среднем: больше похоже на mean-reversion.")
        elif mvr > 1.02:
            hints.append("VR>1 в среднем: больше похоже на momentum/trend.")
        else:
            hints.append("VR≈1: ближе к random walk.")

    phi = out["ar1"]["phi"]
    p_phi1 = out["ar1"]["pvalue_phi_eq_1"]
    if np.isfinite(phi) and np.isfinite(p_phi1):
        if phi < 1 and p_phi1 < alpha:
            hints.append("AR(1): phi значимо < 1 → mean-reversion компонент.")
        elif abs(phi - 1) < 0.02 and p_phi1 >= alpha:
            hints.append("AR(1): phi ≈ 1 → random walk / trend-like.")

    if np.isfinite(out["cusum_trend"]["pvalue"]) and out["cusum_trend"]["pvalue"] < alpha:
        hints.append("CUSUM: нестабильность параметров тренда → возможная смена режима.")

    za_p = out["zivot_andrews_ct"]["pvalue"]
    if np.isfinite(za_p) and za_p < alpha:
        hints.append("Zivot–Andrews: отвергает unit root с 1 разрывом → стационарный с break.")

    out["hints"] = hints
    return out
