"""Abstract model plugin interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class ModelPlugin(ABC):
    """Base class for all pricing models.

    Every model must implement: name, param_names, default_params,
    bounds, and price.  vectorized_price has a default fallback.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def param_names(self) -> list[str]: ...

    @abstractmethod
    def default_params(self) -> np.ndarray:
        """Starting guess for calibration."""
        ...

    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]:
        """(lower, upper) for each parameter."""
        ...

    @abstractmethod
    def price(
        self,
        option_type: str,
        K: float,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> float:
        """Single-strike price. option_type in {'call', 'put'}."""
        ...

    def vectorized_price(
        self,
        option_type: str,
        K_array: np.ndarray,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> np.ndarray:
        """Vectorised pricing over an array of strikes.

        Models may override for speed; default uses a loop.
        """
        return np.array(
            [self.price(option_type, float(k), F, T, params) for k in K_array]
        )
