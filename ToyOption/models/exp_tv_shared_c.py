"""Exponential time-value model with a shared floor parameter.

C(K) = max(F - K, 0) + ac * exp(-bc * |x|) + c
P(K) = max(K - F, 0) + ap * exp(-bp * |x|) + c

where  x = ln(K / F)
       c  — shared floor for both call and put

Parameters
----------
ac : call TV amplitude  (> 0)
bc : call decay rate    (> 0)
ap : put  TV amplitude  (> 0)
bp : put  decay rate    (> 0)
c  : shared floor level (>= 0)

Trade reaction
--------------
ATM component  → shifts c (raises/lowers both call and put uniformly)
Wing component → shifts bc for call trades, bp for put trades (shape only)
"""

from __future__ import annotations
import numpy as np
from .base import ModelPlugin


class ExpTimeValueSharedCModel(ModelPlugin):

    @property
    def name(self) -> str:
        return "ExpTimeValueSharedC"

    @property
    def param_names(self) -> list[str]:
        return ["ac", "bc", "ap", "bp", "c"]

    def default_params(self) -> np.ndarray:
        return np.array([5.0, 3.0, 5.0, 3.0, 0.5])

    def bounds(self) -> list[tuple[float, float]]:
        return [
            (1e-4, 200.0),  # ac
            (0.1,  50.0),   # bc
            (1e-4, 200.0),  # ap
            (0.1,  50.0),   # bp
            (0.0,  50.0),   # c  (shared floor, non-negative)
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def _tv(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Time value: a * exp(-b * |x|) + c."""
        return a * np.exp(-b * np.abs(x)) + c

    def apply_reaction(
        self,
        params: np.ndarray,
        side: str,
        atm_component: float,
        wing_component: float,
        strikes: np.ndarray,
        F: float,
    ) -> np.ndarray:
        ac, bc, ap, bp, c = params

        # ATM component always shifts the shared floor (affects both sides)
        c_new = c + atm_component

        # Wing component adjusts only the decay of the traded side
        if side == "call":
            a, b = ac, bc
            wing_K = float(strikes.max())
        else:
            a, b = ap, bp
            wing_K = float(strikes.min())

        x_wing = abs(np.log(wing_K / F))
        b_new = b
        if x_wing > 1e-12 and abs(wing_component) > 1e-12:
            val = np.exp(-b * x_wing) + wing_component / a
            if val > 1e-8:
                val_safe = min(val, 1.0 - 1e-9)  # saturate for large buys → b → b_lo
                b_new = -np.log(val_safe) / x_wing
            b_idx = 1 if side == "call" else 3
            b_lo, b_hi = self.bounds()[b_idx]
            b_new = float(np.clip(b_new, b_lo, b_hi))

        if side == "call":
            return np.array([ac, b_new, ap, bp, c_new])
        else:
            return np.array([ac, bc, ap, b_new, c_new])

    def price(
        self,
        option_type: str,
        K: float,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> float:
        ac, bc, ap, bp, c = params
        x = np.log(K / F)
        if option_type == "call":
            tv = float(self._tv(np.array([x]), ac, bc, c)[0])
            return max(F - K, 0.0) + tv
        else:
            tv = float(self._tv(np.array([x]), ap, bp, c)[0])
            return max(K - F, 0.0) + tv

    def vectorized_price(
        self,
        option_type: str,
        K_array: np.ndarray,
        F: float,
        T: float,
        params: np.ndarray,
    ) -> np.ndarray:
        ac, bc, ap, bp, c = params
        x = np.log(K_array / F)
        if option_type == "call":
            tv = self._tv(x, ac, bc, c)
            return np.maximum(F - K_array, 0.0) + tv
        else:
            tv = self._tv(x, ap, bp, c)
            return np.maximum(K_array - F, 0.0) + tv
