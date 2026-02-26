"""ModelService — single entry point bridging UI and computation."""

from __future__ import annotations
from typing import Optional
import numpy as np

from .data import CanonicalQuoteSet
from .models.base import ModelPlugin
from .models import MODELS, ExpTimeValueSharedCModel
from .calibrator import Calibrator, FitResult
from .emulator import StrikeReactionEmulator, ReactionConfig, Trade
from .spread import SPREAD_MODELS, ConstantSpreadModel, SpreadModel
from .chain import TradableChain
from . import analyzer


class ModelService:
    """Facade used by the UI layer.

    Holds current state (data, model, params) and exposes
    high-level actions: calibrate, evaluate, diagnose.
    """

    def __init__(self):
        self.quote_set: Optional[CanonicalQuoteSet] = None
        self.model: ModelPlugin = ExpTimeValueSharedCModel()
        self.params: np.ndarray = self.model.default_params()
        self.last_fit: Optional[FitResult] = None
        self.loss_type: str = "huber"
        self.penalty_weight: float = 0.1
        self.emulator: Optional[StrikeReactionEmulator] = None
        self.base_params: Optional[np.ndarray] = None
        self.spread_model: SpreadModel = ConstantSpreadModel()
        self.spread_params: np.ndarray = self.spread_model.default_params()
        # (call_start, call_end, put_start, put_end, step)
        self.chain_config: Optional[tuple[float, float, float, float, float]] = None

    # ------------------------------------------------------------------
    # State setters
    # ------------------------------------------------------------------
    def set_data(self, quote_set: CanonicalQuoteSet) -> None:
        self.quote_set = quote_set

    def set_model(self, model_name: str) -> None:
        cls = MODELS[model_name]
        self.model = cls()
        self.params = self.model.default_params()
        self.last_fit = None
        self.emulator = None    # clear stale emulator from previous model
        self.base_params = None # clear stale base_params from previous model

    def set_params(self, params: np.ndarray) -> None:
        self.params = params.copy()

    def available_models(self) -> list[str]:
        return list(MODELS.keys())

    def set_spread_model(self, model_name: str, params: Optional[np.ndarray] = None) -> None:
        cls = SPREAD_MODELS[model_name]
        self.spread_model = cls()
        self.spread_params = params if params is not None else self.spread_model.default_params()

    def available_spread_models(self) -> list[str]:
        return list(SPREAD_MODELS.keys())

    def set_chain(self, call_start: float, call_end: float,
                  put_start: float, put_end: float, step: float) -> None:
        self.chain_config = (call_start, call_end, put_start, put_end, step)

    def clear_chain(self) -> None:
        self.chain_config = None

    def _build_chain(self, call_start: float, call_end: float,
                     put_start: float, put_end: float, step: float) -> TradableChain:
        call_strikes = np.arange(call_start, call_end + step * 0.5, step)
        put_strikes = np.arange(put_start, put_end + step * 0.5, step)
        F, T = self.quote_set.F, self.quote_set.T
        call_fair = self.model.vectorized_price("call", call_strikes, F, T, self.params)
        put_fair = self.model.vectorized_price("put", put_strikes, F, T, self.params)
        cb_raw, ca_raw = self.spread_model.bid_ask(
            "call", call_strikes, F, T, call_fair, self.spread_params
        )
        pb_raw, pa_raw = self.spread_model.bid_ask(
            "put", put_strikes, F, T, put_fair, self.spread_params
        )
        chain = TradableChain(
            call_strikes=call_strikes,
            put_strikes=put_strikes,
            call_fair=call_fair,
            put_fair=put_fair,
            call_bid_raw=cb_raw,
            call_ask_raw=ca_raw,
            put_bid_raw=pb_raw,
            put_ask_raw=pa_raw,
            call_bid=cb_raw.copy(),
            call_ask=ca_raw.copy(),
            put_bid=pb_raw.copy(),
            put_ask=pa_raw.copy(),
        )
        self._apply_parity_to_chain(chain, step)
        return chain

    def _apply_parity_to_chain(self, chain: TradableChain, step: float) -> None:
        """In-place parity correction на общих страйках цепочки.

        Для каждого страйка, присутствующего в обоих рядах (call и put):
        - вычисляет ошибку паритета: err = C_mid_raw - P_mid_raw - (F - K)
        - распределяет ошибку поровну между call и put
        - сохраняет исходную ширину спреда
        Результат записывается в call_bid/ask, put_bid/ask (не в *_raw).
        """
        F = self.quote_set.F
        # Находим общие страйки через округление (оба грида имеют одинаковый шаг)
        call_key = {round(k / step): i for i, k in enumerate(chain.call_strikes)}
        put_key  = {round(k / step): j for j, k in enumerate(chain.put_strikes)}
        for key in set(call_key) & set(put_key):
            i = call_key[key]
            j = put_key[key]
            K = float(chain.call_strikes[i])
            c_mid  = (chain.call_bid_raw[i] + chain.call_ask_raw[i]) / 2
            p_mid  = (chain.put_bid_raw[j]  + chain.put_ask_raw[j])  / 2
            err    = c_mid - p_mid - (F - K)
            c_half = (chain.call_ask_raw[i] - chain.call_bid_raw[i]) / 2
            p_half = (chain.put_ask_raw[j]  - chain.put_bid_raw[j])  / 2
            chain.call_bid[i] = c_mid - err / 2 - c_half
            chain.call_ask[i] = c_mid - err / 2 + c_half
            chain.put_bid[j]  = p_mid + err / 2 - p_half
            chain.put_ask[j]  = p_mid + err / 2 + p_half

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def calibrate(self) -> FitResult:
        if self.quote_set is None:
            raise ValueError("No data loaded")
        cal = Calibrator(
            self.model,
            loss_type=self.loss_type,
            penalty_weight=self.penalty_weight,
        )
        result = cal.fit(self.quote_set, x0=self.params)
        self.params = result.params
        self.last_fit = result
        return result

    def evaluate_for_plots(self) -> dict:
        """Return everything the UI needs to draw charts.

        Returns dict with keys: K_grid, call_curve, put_curve,
        market_calls, market_puts, residuals, noarb, metrics.
        """
        if self.quote_set is None:
            return {}

        qs = self.quote_set
        F, T = qs.F, qs.T

        # Build strike grid covering all market strikes with margin
        all_K = qs.all_strikes()
        lo = min(all_K.min(), F) * 0.8
        hi = max(all_K.max(), F) * 1.2
        K_grid = np.linspace(lo, hi, 200)

        curves = analyzer.evaluate_grid(self.model, self.params, F, T, K_grid)
        noarb = analyzer.check_noarb(curves, F, T)

        # Residuals on market points
        res_call = res_put = np.array([])
        if len(qs.calls):
            model_c = self.model.vectorized_price(
                "call", qs.call_strikes(), F, T, self.params
            )
            res_call = model_c - qs.call_prices()
        if len(qs.puts):
            model_p = self.model.vectorized_price(
                "put", qs.put_strikes(), F, T, self.params
            )
            res_put = model_p - qs.put_prices()

        all_res = np.concatenate([res_call, res_put]) if (len(res_call) or len(res_put)) else np.array([])
        metrics = analyzer.summary_metrics(all_res) if len(all_res) else {}

        call_bid, call_ask = self.spread_model.bid_ask(
            "call", K_grid, F, T, curves["call"], self.spread_params
        )
        put_bid, put_ask = self.spread_model.bid_ask(
            "put", K_grid, F, T, curves["put"], self.spread_params
        )

        result = {
            "K_grid": K_grid,
            "call_curve": curves["call"],
            "put_curve": curves["put"],
            "call_bid": call_bid,
            "call_ask": call_ask,
            "put_bid": put_bid,
            "put_ask": put_ask,
            "market_calls": {"K": qs.call_strikes(), "P": qs.call_prices()},
            "market_puts": {"K": qs.put_strikes(), "P": qs.put_prices()},
            "res_call": res_call,
            "res_put": res_put,
            "noarb": noarb,
            "metrics": metrics,
            "F": F,
        }

        if self.chain_config is not None:
            result["chain"] = self._build_chain(*self.chain_config)

        return result

    # ------------------------------------------------------------------
    # Emulator
    # ------------------------------------------------------------------
    def init_emulator(self, config: ReactionConfig) -> None:
        """Create a reaction emulator with current params as base.

        Must be called after calibration or manual param setup.
        """
        self.base_params = self.params.copy()
        self.emulator = StrikeReactionEmulator(
            model=self.model,
            base_params=self.base_params,
            config=config,
        )

    def apply_trade(self, trade: Trade) -> np.ndarray:
        """Apply a trade via the emulator and update current params.

        Parameters
        ----------
        trade : Trade
            The trade to react to.

        Returns
        -------
        np.ndarray
            Updated model parameters.

        Raises
        ------
        RuntimeError
            If emulator has not been initialised or no data is loaded.
        """
        if self.emulator is None:
            raise RuntimeError("Emulator not initialised; call init_emulator first")
        if self.quote_set is None:
            raise RuntimeError("No data loaded")
        new_params = self.emulator.apply_trade(trade, self.quote_set)
        self.params = new_params.copy()
        return new_params

    def reset_emulator(self) -> None:
        """Reset params to base (pre-trade) state."""
        if self.emulator is not None:
            self.emulator.reset()
        if self.base_params is not None:
            self.params = self.base_params.copy()

    def evaluate_base_curves(self) -> Optional[dict]:
        """Return base (pre-trade) curves for before/after visualisation.

        Returns None if no emulator is active.
        """
        if self.base_params is None or self.quote_set is None:
            return None
        qs = self.quote_set
        F, T = qs.F, qs.T
        all_K = qs.all_strikes()
        lo = min(all_K.min(), F) * 0.8
        hi = max(all_K.max(), F) * 1.2
        K_grid = np.linspace(lo, hi, 200)
        return {
            "K_grid": K_grid,
            "call_curve": self.model.vectorized_price("call", K_grid, F, T, self.base_params),
            "put_curve": self.model.vectorized_price("put", K_grid, F, T, self.base_params),
        }

    def get_diagnostics(self) -> dict:
        payload = self.evaluate_for_plots()
        return {
            "noarb": payload.get("noarb", []),
            "metrics": payload.get("metrics", {}),
            "fit_success": self.last_fit.success if self.last_fit else None,
            "fit_message": self.last_fit.message if self.last_fit else "",
        }
