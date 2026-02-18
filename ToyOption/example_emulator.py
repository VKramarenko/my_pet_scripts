"""Demo script for the Strike Reaction Emulator with analytical calibration.

Run:  python -m ToyOption.example_emulator

Shows how the two combination models (Linear, LogMoneyness) produce
different parameter adjustments for the same trade, and how weights
at trade.strike determine the blend of ATM and wing movements.
"""

from __future__ import annotations
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_root = _pkg_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from ToyOption.data import CanonicalQuoteSet
from ToyOption.service import ModelService
from ToyOption.emulator import Trade, ReactionConfig, COMBINATION_MODELS


def main() -> None:
    svc = ModelService()

    # Load example data
    csv_path = _pkg_dir / "example_data.csv"
    qs = CanonicalQuoteSet.from_csv(csv_path)
    svc.set_data(qs)

    # Calibrate first
    result = svc.calibrate()
    print("=== Initial calibration ===")
    print(f"Model: {svc.model.name}")
    print(f"Params: {dict(zip(svc.model.param_names, result.params.round(4)))}")
    print(f"RMSE: {result.metrics['rmse']:.4f}")
    print()

    # Prices before trades
    F, T = qs.F, qs.T
    call_K = qs.call_strikes()
    call_pre = svc.model.vectorized_price("call", call_K, F, T, svc.params)
    base_params = svc.params.copy()

    # Trade at wing strike (K=120 for calls)
    trade_wing = Trade(side="call", strike=120.0, volume=100.0, direction="buy")

    for combo_name in COMBINATION_MODELS:
        print(f"=== Combination: {combo_name} ===")
        print(f"Trade: buy call K=120 v=100  (shift_atm=0.0, shift_wing=1.0)")

        svc.set_params(base_params)
        config = ReactionConfig(
            shift_atm=0.0,
            shift_wing=1.0,
            volume_ref=100.0,
            combination_model=combo_name,
        )
        svc.init_emulator(config)

        # Show weights at trade strike (must use full strike array for correct wing)
        combo = COMBINATION_MODELS[combo_name]()
        t_all = combo.interpolation_param(call_K, F, "call")
        t_trade = float(t_all[np.argmin(np.abs(call_K - 120.0))])
        print(f"  t(K=120) = {t_trade:.3f}  ->  weight_atm={1 - t_trade:.3f}, weight_wing={t_trade:.3f}")

        svc.apply_trade(trade_wing)

        print(f"  Params before: {dict(zip(svc.model.param_names, base_params.round(4)))}")
        print(f"  Params after:  {dict(zip(svc.model.param_names, svc.params.round(4)))}")

        call_post = svc.model.vectorized_price("call", call_K, F, T, svc.params)
        print(f"  {'Strike':>8} {'Before':>10} {'After':>10} {'Diff':>10}")
        for k, pre, post in zip(call_K, call_pre, call_post):
            print(f"  {k:8.1f} {pre:10.4f} {post:10.4f} {post - pre:+10.4f}")
        print()

    # Trade at intermediate strike
    print("=== Trade at intermediate strike (K=110, Linear) ===")
    trade_mid = Trade(side="call", strike=110.0, volume=100.0, direction="buy")
    svc.set_params(base_params)
    config = ReactionConfig(shift_atm=0.5, shift_wing=1.0, volume_ref=100.0,
                            combination_model="Linear")
    svc.init_emulator(config)

    combo = COMBINATION_MODELS["Linear"]()
    t_all_mid = combo.interpolation_param(call_K, F, "call")
    t_mid = float(t_all_mid[np.argmin(np.abs(call_K - 110.0))])
    print(f"Trade: buy call K=110 v=100  (shift_atm=0.5, shift_wing=1.0)")
    print(f"  t(K=110) = {t_mid:.3f}  ->  weight_atm={1 - t_mid:.3f}, weight_wing={t_mid:.3f}")
    print(f"  atm_component = 0.5 * {1 - t_mid:.3f} = {0.5 * (1 - t_mid):.3f}")
    print(f"  wing_component = 1.0 * {t_mid:.3f} = {1.0 * t_mid:.3f}")

    svc.apply_trade(trade_mid)
    print(f"  Params before: {dict(zip(svc.model.param_names, base_params.round(4)))}")
    print(f"  Params after:  {dict(zip(svc.model.param_names, svc.params.round(4)))}")

    call_post = svc.model.vectorized_price("call", call_K, F, T, svc.params)
    print(f"  {'Strike':>8} {'Before':>10} {'After':>10} {'Diff':>10}")
    for k, pre, post in zip(call_K, call_pre, call_post):
        print(f"  {k:8.1f} {pre:10.4f} {post:10.4f} {post - pre:+10.4f}")
    print()

    # Cumulative test
    print("=== Cumulative trades (Linear) ===")
    svc.set_params(base_params)
    config = ReactionConfig(shift_atm=0.0, shift_wing=1.0, volume_ref=100.0,
                            combination_model="Linear")
    svc.init_emulator(config)

    trades = [
        Trade(side="call", strike=120.0, volume=50.0, direction="buy"),
        Trade(side="call", strike=120.0, volume=50.0, direction="buy"),
    ]
    for i, tr in enumerate(trades, 1):
        svc.apply_trade(tr)
        call_after = svc.model.vectorized_price("call", call_K, F, T, svc.params)
        print(f"After trade #{i} ({tr.direction} {tr.side} K={tr.strike} v={tr.volume}):")
        print(f"  Params: {dict(zip(svc.model.param_names, svc.params.round(4)))}")
        for k, pre, post in zip(call_K, call_pre, call_after):
            print(f"  K={k:6.1f}  base={pre:.4f}  now={post:.4f}  diff={post - pre:+.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
