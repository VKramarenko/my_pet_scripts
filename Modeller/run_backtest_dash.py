from __future__ import annotations

import argparse
from typing import Any

from run_backtest import _build_strategy, run
from sim.visualization import create_dash_app


def _run_from_form(form: dict[str, Any]):
    strategy_args = argparse.Namespace(
        strategy=form["strategy"],
        mm_spread=form["mm_spread"],
        mm_quote_size=form["mm_quote_size"],
        taker_window=form["taker_window"],
        taker_std_mult=form["taker_std_mult"],
        taker_qty=form["taker_qty"],
        taker_cooldown=form["taker_cooldown"],
        taker_max_position=form["taker_max_position"],
    )
    strategy = _build_strategy(strategy_args)
    return run(
        form["l2"],
        form.get("trades"),
        loader=form["loader"],
        strategy=strategy,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run backtest and visualize the result with Dash."
    )
    parser.add_argument("--l2", required=True, help="Path to L2 snapshots")
    parser.add_argument("--trades", help="Path to trades (optional)")
    parser.add_argument(
        "--loader",
        default="default",
        choices=["default", "bybit", "test_data"],
        help="Input loader format",
    )
    parser.add_argument(
        "--strategy",
        default="mm",
        choices=["mm", "taker"],
        help="Strategy to run",
    )
    parser.add_argument("--mm-spread", type=float, default=1.0, help="MM quote spread")
    parser.add_argument("--mm-quote-size", type=float, default=1.0, help="MM quote size")
    parser.add_argument("--taker-window", type=int, default=20, help="Bollinger window size")
    parser.add_argument("--taker-std-mult", type=float, default=2.0, help="Bollinger stdev multiplier")
    parser.add_argument("--taker-qty", type=float, default=1.0, help="Taker market order quantity")
    parser.add_argument(
        "--taker-cooldown",
        type=float,
        default=0.0,
        help="Cooldown in seconds between new taker entries from flat",
    )
    parser.add_argument(
        "--taker-max-position",
        type=float,
        default=1.0,
        help="Absolute max position for taker strategy",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    initial_form = {
        "l2": args.l2,
        "trades": args.trades,
        "loader": args.loader,
        "strategy": args.strategy,
        "mm_spread": args.mm_spread,
        "mm_quote_size": args.mm_quote_size,
        "taker_window": args.taker_window,
        "taker_std_mult": args.taker_std_mult,
        "taker_qty": args.taker_qty,
        "taker_cooldown": args.taker_cooldown,
        "taker_max_position": args.taker_max_position,
    }
    metrics = _run_from_form(initial_form)

    app = create_dash_app(metrics, _run_from_form, initial_form)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
