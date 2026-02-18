"""Strike Reaction Emulator — shifts model surface in response to trades.

Both movements (ATM parallel shift and wing shift) are always computed.
A CombinationModel plugin determines how to blend them for a given trade strike:
  - ATM and ITM: only shift_atm applies (weight_atm=1, weight_wing=0)
  - Wing (extreme OTM): only shift_wing applies (weight_atm=0, weight_wing=1)
  - Intermediate OTM: interpolated by the combination model

The model itself adjusts its parameters analytically via ``apply_reaction``.
Trades are applied cumulatively on top of the current model state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from .models.base import ModelPlugin
from .data import CanonicalQuoteSet


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class Trade:
    """A single trade to react to.

    Attributes
    ----------
    side : str
        ``"call"`` or ``"put"``.
    strike : float
        Strike price of the trade.
    volume : float
        Trade volume.
    direction : str
        ``"buy"`` (prices shift up) or ``"sell"`` (prices shift down).
    """
    side: str
    strike: float
    volume: float
    direction: str

    def __post_init__(self) -> None:
        if self.side not in ("call", "put"):
            raise ValueError(f"side must be 'call' or 'put', got {self.side!r}")
        if self.direction not in ("buy", "sell"):
            raise ValueError(f"direction must be 'buy' or 'sell', got {self.direction!r}")
        if self.volume <= 0:
            raise ValueError(f"volume must be positive, got {self.volume}")


@dataclass
class ReactionConfig:
    """UI-controlled parameters for the strike reaction emulator.

    Attributes
    ----------
    shift_atm : float
        Baseline price shift at ATM for ``volume_ref`` volume.
    shift_wing : float
        Target price shift at the wing strike for ``volume_ref`` volume.
    volume_ref : float
        Reference volume that corresponds to the full shift amounts.
    combination_model : str
        Name of the combination model (key in ``COMBINATION_MODELS``).
    """
    shift_atm: float = 1.0
    shift_wing: float = 1.0
    volume_ref: float = 100.0
    combination_model: str = "Linear"

    def __post_init__(self) -> None:
        if self.combination_model not in COMBINATION_MODELS:
            raise ValueError(
                f"combination_model must be one of {list(COMBINATION_MODELS)}, "
                f"got {self.combination_model!r}"
            )
        if self.volume_ref <= 0:
            raise ValueError(f"volume_ref must be positive, got {self.volume_ref}")


# -----------------------------------------------------------------------
# Combination models (plugin system)
# -----------------------------------------------------------------------

class CombinationModel(ABC):
    """Base class for strike-weight combination models.

    A combination model computes an interpolation parameter ``t(K)`` that
    goes from 0 at ATM/ITM to 1 at the wing (extreme OTM).

    For a trade at strike K:
        weight_atm  = 1 - t(K)
        weight_wing = t(K)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""

    @abstractmethod
    def interpolation_param(
        self,
        strikes: np.ndarray,
        F: float,
        side: str,
    ) -> np.ndarray:
        """Compute per-strike interpolation parameter t(K).

        Parameters
        ----------
        strikes : np.ndarray
            Ordered strike array for the given side.
        F : float
            Forward price (used as ATM proxy).
        side : str
            ``"call"`` or ``"put"``.

        Returns
        -------
        t : np.ndarray
            Interpolation parameter, 0 at ATM/ITM, 1 at wing.
        """


class LinearCombination(CombinationModel):
    """Linear interpolation in strike space.

    ``t(K)`` increases linearly from 0 at ATM to 1 at the wing strike.
    ITM strikes get ``t = 0``.
    """

    @property
    def name(self) -> str:
        return "Linear"

    def interpolation_param(
        self,
        strikes: np.ndarray,
        F: float,
        side: str,
    ) -> np.ndarray:
        atm = F
        if side == "call":
            wing = float(strikes.max())
        else:
            wing = float(strikes.min())

        span = wing - atm
        if abs(span) < 1e-12:
            return np.zeros_like(strikes, dtype=float)

        if side == "call":
            return np.clip((strikes - atm) / span, 0.0, 1.0)
        else:
            return np.clip((atm - strikes) / (atm - wing), 0.0, 1.0)


class LogMoneynessCombination(CombinationModel):
    """Linear interpolation in log-moneyness space.

    ``t(K)`` increases linearly in ``log(K/F)`` from 0 at ATM to 1
    at the wing.  ITM strikes get ``t = 0``.
    """

    @property
    def name(self) -> str:
        return "LogMoneyness"

    def interpolation_param(
        self,
        strikes: np.ndarray,
        F: float,
        side: str,
    ) -> np.ndarray:
        atm = F
        if side == "call":
            wing = float(strikes.max())
        else:
            wing = float(strikes.min())

        log_wing = np.log(wing / atm)
        if abs(log_wing) < 1e-12:
            return np.zeros_like(strikes, dtype=float)

        if side == "call":
            log_k = np.log(strikes / atm)
            return np.clip(log_k / log_wing, 0.0, 1.0)
        else:
            log_k = np.log(atm / strikes)
            log_span = np.log(atm / wing)
            return np.clip(log_k / log_span, 0.0, 1.0)


COMBINATION_MODELS: dict[str, type[CombinationModel]] = {
    "Linear": LinearCombination,
    "LogMoneyness": LogMoneynessCombination,
}


# -----------------------------------------------------------------------
# Emulator
# -----------------------------------------------------------------------

class StrikeReactionEmulator:
    """Applies analytical parameter adjustments in response to trades.

    For each trade:
    1. Compute shift_atm_eff and shift_wing_eff (scaled by volume).
    2. Evaluate combination model at trade.strike to get (weight_atm, weight_wing).
    3. Compute atm_component and wing_component.
    4. Call model.apply_reaction() for analytical parameter adjustment.

    Parameters
    ----------
    model : ModelPlugin
        The pricing model (same instance used by ModelService).
    base_params : np.ndarray
        Model parameters *before* any trades.
    config : ReactionConfig
        Shift magnitudes and combination model name.
    """

    def __init__(
        self,
        model: ModelPlugin,
        base_params: np.ndarray,
        config: ReactionConfig,
    ) -> None:
        self.model = model
        self.base_params = base_params.copy()
        self.current_params = base_params.copy()
        self.config = config
        self.combination: CombinationModel = COMBINATION_MODELS[config.combination_model]()
        self.trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_trade(
        self,
        trade: Trade,
        quote_set: CanonicalQuoteSet,
    ) -> np.ndarray:
        """Apply a single trade and return the updated model parameters.

        Parameters
        ----------
        trade : Trade
            The trade to react to.
        quote_set : CanonicalQuoteSet
            Current market quotes (used for strike structure).

        Returns
        -------
        np.ndarray
            New model parameters after reaction.
        """
        sign = 1.0 if trade.direction == "buy" else -1.0
        s = trade.volume / self.config.volume_ref
        shift_atm_eff = sign * self.config.shift_atm * s
        shift_wing_eff = sign * self.config.shift_wing * s

        F = quote_set.F
        side = trade.side

        # Get strikes for the traded side
        if side == "call":
            strikes = quote_set.call_strikes()
        else:
            strikes = quote_set.put_strikes()

        if len(strikes) == 0:
            return self.current_params.copy()

        # Evaluate combination model on full strike array, then extract t
        # at trade.strike (must use full array so wing is determined correctly)
        t_all = self.combination.interpolation_param(strikes, F, side)
        idx = np.argmin(np.abs(strikes - trade.strike))
        t = float(t_all[idx])
        weight_atm = 1.0 - t
        weight_wing = t

        atm_component = shift_atm_eff * weight_atm
        wing_component = shift_wing_eff * weight_wing

        # Analytical parameter adjustment
        new_params = self.model.apply_reaction(
            self.current_params, side,
            atm_component, wing_component,
            strikes, F,
        )

        self.current_params = new_params.copy()
        self.trade_log.append({
            "side": side,
            "strike": trade.strike,
            "volume": trade.volume,
            "direction": trade.direction,
            "combination": self.combination.name,
            "weight_atm": weight_atm,
            "weight_wing": weight_wing,
        })
        return new_params

    def reset(self) -> None:
        """Reset to base (pre-trade) parameters."""
        self.current_params = self.base_params.copy()
        self.trade_log.clear()
