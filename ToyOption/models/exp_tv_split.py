"""Split exponential time-value model — independent TV for call and put.

C(K) = max(F - K, 0) + TV_call(x; ac, bc, cc)
P(K) = max(K - F, 0) + TV_put(x; ap, bp, cp)

where  x = ln(K / F)
       TV(x; a, b, c) = a * exp(-b * |x|) + c

Parameters (6 total)
--------------------
ac, bc, cc : amplitude, decay, base level for calls
ap, bp, cp : amplitude, decay, base level for puts

Note: put-call parity is NOT built in — call and put TV are
independent, so parity violations are possible and will show
up in diagnostics.
"""

from __future__ import annotations
import numpy as np
from .base import ModelPlugin


class ExpTimeValueSplitModel(ModelPlugin):

    @property
    def name(self) -> str:
        return "ExpTimeValueSplit"

    @property
    def param_names(self) -> list[str]:
        return ["ac", "bc", "cc", "ap", "bp", "cp"]

    def default_params(self) -> np.ndarray:
        return np.array([5.0, 3.0, 0.5, 5.0, 3.0, 0.5])

    def bounds(self) -> list[tuple[float, float]]:
        return [
            (1e-4, 200.0),  # ac
            (0.1, 50.0),    # bc
            (0.0, 50.0),    # cc
            (1e-4, 200.0),  # ap
            (0.1, 50.0),    # bp
            (0.0, 50.0),    # cp
        ]

    @staticmethod
    def _tv(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-b * np.abs(x)) + c

    def price(
        self,
        option_type: str,
        K: float,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> float:
        ac, bc, cc, ap, bp, cp = params
        x = np.log(K / F)
        if option_type == "call":
            tv = float(self._tv(np.array([x]), ac, bc, cc)[0])
            return max(F - K, 0.0) + tv
        else:
            tv = float(self._tv(np.array([x]), ap, bp, cp)[0])
            return max(K - F, 0.0) + tv

    def vectorized_price(
        self,
        option_type: str,
        K_array: np.ndarray,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> np.ndarray:
        ac, bc, cc, ap, bp, cp = params
        x = np.log(K_array / F)
        if option_type == "call":
            tv = self._tv(x, ac, bc, cc)
            return np.maximum(F - K_array, 0.0) + tv
        else:
            tv = self._tv(x, ap, bp, cp)
            return np.maximum(K_array - F, 0.0) + tv
