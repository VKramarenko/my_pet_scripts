"""Bid/Ask spread models.

A spread model computes the full bid-ask spread (ask - bid) for an option.
Given a fair price P:
    Bid = P - spread / 2
    Ask = P + spread / 2

To add a new spread model:
1. Subclass SpreadModel and implement spread() (and optionally vectorized_spread()).
2. Register it in SPREAD_MODELS.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class SpreadModel(ABC):
    """Abstract base class for bid/ask spread models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name (used as registry key)."""

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of the model parameters."""

    @abstractmethod
    def default_params(self) -> np.ndarray:
        """Default parameter values."""

    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]:
        """(lo, hi) bounds for each parameter."""

    @abstractmethod
    def spread(
        self,
        option_type: str,
        K: float,
        F: float,
        T: float,
        fair_price: float,
        params: np.ndarray,
    ) -> float:
        """Return the full bid-ask spread (ask − bid) for a single option.

        Parameters
        ----------
        option_type : str
            ``"call"`` or ``"put"``.
        K : float
            Strike price.
        F : float
            Forward price.
        T : float
            Time to expiry in years.
        fair_price : float
            Model fair price for this option.
        params : np.ndarray
            Spread model parameters.

        Returns
        -------
        float
            Non-negative full spread value.
        """

    def vectorized_spread(
        self,
        option_type: str,
        K_array: np.ndarray,
        F: float,
        T: float,
        fair_prices: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Vectorised version of spread().  Override for performance.

        Default implementation loops over each strike.
        """
        return np.array([
            self.spread(option_type, float(K), F, T, float(p), params)
            for K, p in zip(K_array, fair_prices)
        ])

    def bid_ask(
        self,
        option_type: str,
        K_array: np.ndarray,
        F: float,
        T: float,
        fair_prices: np.ndarray,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (bid_prices, ask_prices) arrays."""
        s = self.vectorized_spread(option_type, K_array, F, T, fair_prices, params)
        return fair_prices - s / 2, fair_prices + s / 2


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class ConstantSpreadModel(SpreadModel):
    """Constant spread regardless of strike, moneyness, or price level.

    Parameters
    ----------
    spread : float
        Full bid-ask spread (ask − bid).  Default: 0.5.
    """

    @property
    def name(self) -> str:
        return "Constant"

    @property
    def param_names(self) -> list[str]:
        return ["spread"]

    def default_params(self) -> np.ndarray:
        return np.array([0.5])

    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 100.0)]

    def spread(
        self,
        option_type: str,
        K: float,
        F: float,
        T: float,
        fair_price: float,
        params: np.ndarray,
    ) -> float:
        return float(params[0])

    def vectorized_spread(
        self,
        option_type: str,
        K_array: np.ndarray,
        F: float,
        T: float,
        fair_prices: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        return np.full(len(K_array), float(params[0]))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SPREAD_MODELS: dict[str, type[SpreadModel]] = {
    "Constant": ConstantSpreadModel,
}
