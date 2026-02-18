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

    def apply_reaction(
        self,
        params: np.ndarray,
        side: str,
        atm_component: float,
        wing_component: float,
        strikes: np.ndarray,
        F: float,
    ) -> np.ndarray:
        new_params = params.copy()
        ac, bc, cc, ap, bp, cp = new_params

        if side == "call":
            a, b, c = ac, bc, cc
            wing_K = float(strikes.max())
        else:
            a, b, c = ap, bp, cp
            wing_K = float(strikes.min())

        # Step 1: ATM shift — adjust c
        c_new = c + atm_component

        # Step 2: Wing shift — find b analytically
        x_wing = abs(np.log(wing_K / F))
        b_new = b
        if x_wing > 1e-12 and abs(wing_component) > 1e-12:
            val = np.exp(-b * x_wing) + wing_component / a
            if val > 1e-8 and val < 1.0:
                b_new = -np.log(val) / x_wing
            # Clamp to bounds
            bounds = self.bounds()
            b_idx = 1 if side == "call" else 4
            b_new = np.clip(b_new, bounds[b_idx][0], bounds[b_idx][1])

        if side == "call":
            new_params = np.array([a, b_new, c_new, ap, bp, cp])
        else:
            new_params = np.array([ac, bc, cc, a, b_new, c_new])
        return new_params

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
