"""Abstract model plugin interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..data import CanonicalQuoteSet


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

    def apply_reaction(
        self,
        params: np.ndarray,
        side: str,
        atm_component: float,
        wing_component: float,
        strikes: np.ndarray,
        F: float,
    ) -> np.ndarray:
        """Analytically adjust model params for a trade reaction.

        Parameters
        ----------
        params : np.ndarray
            Current model parameters.
        side : str
            ``"call"`` or ``"put"``.
        atm_component : float
            Weighted ATM shift to apply (shift_atm_eff * weight_atm).
        wing_component : float
            Weighted wing shift to apply (shift_wing_eff * weight_wing).
        strikes : np.ndarray
            Strike array for the traded side.
        F : float
            Forward price.

        Returns
        -------
        np.ndarray
            New model parameters.
        """
        raise NotImplementedError(
            f"{self.name} does not implement apply_reaction"
        )

    def apply_parity_correction(
        self,
        params: np.ndarray,
        traded_side: str,
        quote_set: CanonicalQuoteSet,
    ) -> np.ndarray:
        """Restore put-call parity on the non-traded side after a reaction.

        Default implementation is a no-op (models with a shared TV function
        satisfy parity automatically).  Override in models that have
        independent call/put TV functions.

        Parameters
        ----------
        params : np.ndarray
            Model parameters after apply_reaction.
        traded_side : str
            ``"call"`` or ``"put"`` — the side that was just traded.
        quote_set : CanonicalQuoteSet
            Current quote set (used to find common call/put strikes).

        Returns
        -------
        np.ndarray
            Potentially adjusted model parameters.
        """
        return params
