"""Exponential time-value model (Variant A).

C(K) = max(F - K, 0) + TV(x; a, b, c, d)
P(K) = max(K - F, 0) + TV(x; a, b, c, d)

where  x = ln(K / F)
       TV(x; a, b, c, d) = a * exp(-b * |x - d|) + c

Parameters
----------
a : amplitude of the TV peak (> 0)
b : decay rate            (> 0)
c : base level of TV      (>= 0)
d : center shift in log-strike space (small, around 0)

Put-call parity is automatically satisfied because the same TV
function is added to both call and put intrinsic values.
"""

from __future__ import annotations
import numpy as np
from .base import ModelPlugin


class ExpTimeValueModel(ModelPlugin):

    @property
    def name(self) -> str:
        return "ExpTimeValue"

    @property
    def param_names(self) -> list[str]:
        return ["a", "b", "c", "d"]

    def default_params(self) -> np.ndarray:
        return np.array([5.0, 3.0, 0.5, 0.0])

    def bounds(self) -> list[tuple[float, float]]:
        return [
            (1e-4, 200.0),   # a
            (0.1, 50.0),     # b
            (0.0, 50.0),     # c
            (-1.0, 1.0),     # d
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _tv(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return a * np.exp(-b * np.abs(x - d)) + c

    def price(
        self,
        option_type: str,
        K: float,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> float:
        a, b, c, d = params
        x = np.log(K / F)
        tv = float(self._tv(np.array([x]), a, b, c, d)[0])
        if option_type == "call":
            return max(F - K, 0.0) + tv
        else:
            return max(K - F, 0.0) + tv

    def vectorized_price(
        self,
        option_type: str,
        K_array: np.ndarray,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> np.ndarray:
        a, b, c, d = params
        x = np.log(K_array / F)
        tv = self._tv(x, a, b, c, d)
        if option_type == "call":
            return np.maximum(F - K_array, 0.0) + tv
        else:
            return np.maximum(K_array - F, 0.0) + tv
