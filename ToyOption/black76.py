"""Black-76 (undiscounted) pricing and implied volatility inversion.

All functions work with forward prices directly (no discount factor).

    Call = F * N(d1) - K * N(d2)
    Put  = K * N(-d2) - F * N(-d1)

    d1 = [ln(F/K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black76_price(F: float, K: float, sigma: float, T: float, flag: str) -> float:
    """Undiscounted Black-76 option price.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike price.
    sigma : float
        Volatility (annualized).
    T : float
        Time to expiry in years.
    flag : str
        ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Option price (undiscounted).
    """
    if sigma <= 0 or T <= 0:
        if flag == "call":
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if flag == "call":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    return K * norm.cdf(-d2) - F * norm.cdf(-d1)


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    flag: str,
    tol: float = 1e-12,
) -> float:
    """Compute implied volatility via Brent's method.

    Parameters
    ----------
    price : float
        Observed (undiscounted) option price.
    F, K, T, flag
        As in :func:`black76_price`.
    tol : float
        Absolute tolerance for the root finder.

    Returns
    -------
    float
        Implied volatility, or ``NaN`` if inversion fails.
    """
    intrinsic = max(F - K, 0.0) if flag == "call" else max(K - F, 0.0)

    if price <= intrinsic + tol:
        return np.nan
    if T <= 0:
        return np.nan

    def objective(sigma: float) -> float:
        return black76_price(F, K, sigma, T, flag) - price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=tol)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vols_from_prices(
    prices: np.ndarray,
    strikes: np.ndarray,
    F: float,
    T: float,
    flag: str,
) -> np.ndarray:
    """Vectorized implied volatility computation.

    Parameters
    ----------
    prices : np.ndarray
        Array of option prices.
    strikes : np.ndarray
        Array of strike prices (same length as *prices*).
    F, T, flag
        As in :func:`black76_price`.

    Returns
    -------
    np.ndarray
        Array of implied volatilities (NaN where inversion fails).
    """
    return np.array([
        implied_vol(float(p), F, float(k), T, flag)
        for p, k in zip(prices, strikes)
    ])
