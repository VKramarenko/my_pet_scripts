"""Post-fit diagnostics and no-arbitrage checks."""

from __future__ import annotations
import numpy as np
from .models.base import ModelPlugin


def evaluate_grid(
    model: ModelPlugin,
    params: np.ndarray,
    F: float,
    T: float,
    K_grid: np.ndarray,
) -> dict:
    """Compute model call/put prices on a strike grid."""
    return {
        "K": K_grid,
        "call": model.vectorized_price("call", K_grid, F, T, params),
        "put": model.vectorized_price("put", K_grid, F, T, params),
    }


def check_noarb(curves: dict, F: float, T: float) -> list[dict]:
    """Run no-arbitrage checks on evaluated curves.

    Returns a list of {name, ok, detail} dicts.
    """
    K = curves["K"]
    call = curves["call"]
    put = curves["put"]
    results = []

    # 1. Put-call parity: C - P = F - K
    parity_err = np.abs((call - put) - (F - K))
    max_parity = float(np.max(parity_err))
    results.append({
        "name": "Put-call parity",
        "ok": max_parity < 1e-6,
        "detail": f"max |C-P-(F-K)| = {max_parity:.6f}",
    })

    # 2. Call monotonicity (should decrease in K)
    call_diff = np.diff(call)
    n_violations = int(np.sum(call_diff > 1e-10))
    results.append({
        "name": "Call monotonicity",
        "ok": n_violations == 0,
        "detail": f"{n_violations} violations out of {len(call_diff)} intervals",
    })

    # 3. Put monotonicity (should increase in K)
    put_diff = np.diff(put)
    n_violations = int(np.sum(put_diff < -1e-10))
    results.append({
        "name": "Put monotonicity",
        "ok": n_violations == 0,
        "detail": f"{n_violations} violations out of {len(put_diff)} intervals",
    })

    # 4. Call convexity (second derivative >= 0)
    if len(K) >= 3:
        call_dd = np.diff(call, 2)
        n_convex = int(np.sum(call_dd < -1e-10))
        results.append({
            "name": "Call convexity",
            "ok": n_convex == 0,
            "detail": f"{n_convex} violations out of {len(call_dd)} points",
        })

    # 5. Prices >= intrinsic
    call_intrinsic = np.maximum(F - K, 0.0)
    put_intrinsic = np.maximum(K - F, 0.0)
    call_below = int(np.sum(call < call_intrinsic - 1e-10))
    put_below = int(np.sum(put < put_intrinsic - 1e-10))
    results.append({
        "name": "Price >= intrinsic",
        "ok": (call_below + put_below) == 0,
        "detail": f"call: {call_below}, put: {put_below} violations",
    })

    return results


def summary_metrics(residuals: np.ndarray) -> dict:
    """Compute summary error metrics from residuals."""
    return {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_error": float(np.max(np.abs(residuals))) if len(residuals) else 0.0,
    }
