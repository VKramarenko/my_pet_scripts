"""Universal calibrator — fits any ModelPlugin to a CanonicalQuoteSet."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import least_squares

from .data import CanonicalQuoteSet
from .models.base import ModelPlugin


@dataclass
class FitResult:
    params: np.ndarray
    residuals: np.ndarray
    metrics: dict
    success: bool
    message: str


class Calibrator:
    """Fit model parameters to market quotes.

    Parameters
    ----------
    model : ModelPlugin
    loss_type : str
        Loss function for scipy least_squares: 'linear', 'huber', 'soft_l1'.
    penalty_weight : float
        Multiplier for soft penalty terms (monotonicity, convexity, TV >= 0).
    """

    def __init__(
        self,
        model: ModelPlugin,
        loss_type: str = "huber",
        penalty_weight: float = 0.1,
    ):
        self.model = model
        self.loss_type = loss_type
        self.penalty_weight = penalty_weight

    def fit(
        self,
        quote_set: CanonicalQuoteSet,
        x0: Optional[np.ndarray] = None,
    ) -> FitResult:
        model = self.model
        F, T = quote_set.F, quote_set.T

        # Collect market data vectors
        call_K = quote_set.call_strikes()
        call_P = quote_set.call_prices()
        call_W = quote_set.call_weights()
        put_K = quote_set.put_strikes()
        put_P = quote_set.put_prices()
        put_W = quote_set.put_weights()

        def residuals(params: np.ndarray) -> np.ndarray:
            res = []
            # Call residuals
            if len(call_K):
                model_c = model.vectorized_price("call", call_K, F, T, params)
                res.append((model_c - call_P) * call_W)
            # Put residuals
            if len(put_K):
                model_p = model.vectorized_price("put", put_K, F, T, params)
                res.append((model_p - put_P) * put_W)
            # Soft penalties
            pen = self._penalties(params, F, T)
            if pen:
                res.append(np.array(pen) * self.penalty_weight)
            return np.concatenate(res)

        bounds_lo = [b[0] for b in model.bounds()]
        bounds_hi = [b[1] for b in model.bounds()]
        x_start = x0 if x0 is not None else model.default_params()

        result = least_squares(
            residuals,
            x_start,
            bounds=(bounds_lo, bounds_hi),
            loss=self.loss_type,
            method="trf",
            max_nfev=2000,
        )

        # Compute metrics on data residuals only (exclude penalty terms)
        n_data = len(call_K) + len(put_K)
        data_res = result.fun[:n_data]
        metrics = _compute_metrics(data_res)

        return FitResult(
            params=result.x,
            residuals=data_res,
            metrics=metrics,
            success=result.success,
            message=result.message,
        )

    def _penalties(
        self, params: np.ndarray, F: float, T: float
    ) -> list[float]:
        """Soft penalty terms to encourage well-behaved curves."""
        model = self.model
        pen = []
        # Evaluate on a small grid around F
        K_grid = F * np.exp(np.linspace(-0.5, 0.5, 20))
        call_prices = model.vectorized_price("call", K_grid, F, T, params)
        put_prices = model.vectorized_price("put", K_grid, F, T, params)

        # Monotonicity: call should decrease in K
        call_diff = np.diff(call_prices)
        pen.extend(np.maximum(call_diff, 0.0))  # penalise increases

        # Monotonicity: put should increase in K
        put_diff = np.diff(put_prices)
        pen.extend(np.maximum(-put_diff, 0.0))  # penalise decreases

        # Convexity: second derivative >= 0 for calls
        call_dd = np.diff(call_prices, 2)
        pen.extend(np.maximum(-call_dd, 0.0))

        return pen


def _compute_metrics(residuals: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_error": float(np.max(np.abs(residuals))),
    }
