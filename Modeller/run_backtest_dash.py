from __future__ import annotations

import argparse
import datetime
from typing import Any

from run_backtest import _build_strategy, run
from sim.data.loaders import load_l2_binance
from sim.visualization import create_dash_app


def _run_from_form(form: dict[str, Any]):
    strategy_args = argparse.Namespace(
        strategy=form["strategy"],
        # MM
        mm_spread=form["mm_spread"],
        mm_quote_size=form["mm_quote_size"],
        # Taker Bollinger
        taker_window=form["taker_window"],
        taker_std_mult=form["taker_std_mult"],
        taker_qty=form["taker_qty"],
        taker_cooldown=form["taker_cooldown"],
        taker_max_position=form["taker_max_position"],
        # EMA Crossover
        ema_fast=form["ema_fast"],
        ema_slow=form["ema_slow"],
        ema_qty=form["ema_qty"],
        ema_max_position=form["ema_max_position"],
        ema_offset=form["ema_offset"],
        # Orderbook Imbalance
        imb_depth=form["imb_depth"],
        imb_threshold=form["imb_threshold"],
        imb_smoothing=form["imb_smoothing"],
        imb_qty=form["imb_qty"],
        imb_max_position=form["imb_max_position"],
    )
    strategy = _build_strategy(strategy_args)

    # Если Binance-снапшоты уже загружены — переиспользуем их (загрузка медленная)
    pre_loaded = form.get("snapshots")
    if pre_loaded is not None:
        return run(snapshots=pre_loaded, strategy=strategy)

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

    # Data source
    parser.add_argument("--l2", help="Path to L2 snapshots (for default/bybit/test_data loaders)")
    parser.add_argument("--trades", help="Path to trades (optional)")
    parser.add_argument(
        "--loader",
        default="default",
        choices=["default", "bybit", "test_data", "binance"],
        help="Input loader format",
    )

    # Binance-specific options
    binance_group = parser.add_argument_group("Binance data source (--loader binance)")
    binance_group.add_argument(
        "--binance-dir",
        help="Directory with Binance data (containing snapshots/ and diff/ subdirs)",
    )
    binance_group.add_argument(
        "--binance-symbol",
        default="BTCUSDT",
        help="Binance symbol, e.g. BTCUSDT (default: BTCUSDT)",
    )
    binance_group.add_argument(
        "--binance-date",
        help="Date to load in YYYY-MM-DD format (loads full UTC day). "
             "If omitted, loads all available data.",
    )
    binance_group.add_argument(
        "--depth",
        type=int,
        default=25,
        help="Orderbook depth (levels per side) for Binance loader (default: 25)",
    )

    # Strategy
    parser.add_argument(
        "--strategy",
        default="mm",
        choices=["mm", "taker", "ema", "imbalance"],
        help="Strategy to run",
    )
    parser.add_argument("--mm-spread", type=float, default=1.0, help="MM quote spread")
    parser.add_argument("--mm-quote-size", type=float, default=1.0, help="MM quote size")
    parser.add_argument("--taker-window", type=int, default=20, help="Bollinger window size")
    parser.add_argument("--taker-std-mult", type=float, default=2.0, help="Bollinger stdev multiplier")
    parser.add_argument("--taker-qty", type=float, default=1.0, help="Taker market order quantity")
    parser.add_argument("--taker-cooldown", type=float, default=0.0,
                        help="Cooldown in seconds between new taker entries from flat")
    parser.add_argument("--taker-max-position", type=float, default=1.0,
                        help="Absolute max position for taker strategy")

    ema_group = parser.add_argument_group("EMA Crossover strategy (--strategy ema)")
    ema_group.add_argument("--ema-fast", type=int, default=10, help="Fast EMA window (default: 10)")
    ema_group.add_argument("--ema-slow", type=int, default=30, help="Slow EMA window (default: 30)")
    ema_group.add_argument("--ema-qty", type=float, default=1.0, help="Order quantity (default: 1.0)")
    ema_group.add_argument("--ema-max-position", type=float, default=1.0, help="Max position (default: 1.0)")
    ema_group.add_argument("--ema-offset", type=float, default=0.0,
                           help="Limit price offset from best bid/ask (default: 0.0)")

    imb_group = parser.add_argument_group("Orderbook Imbalance strategy (--strategy imbalance)")
    imb_group.add_argument("--imb-depth", type=int, default=5, help="Orderbook levels to consider (default: 5)")
    imb_group.add_argument("--imb-threshold", type=float, default=0.3,
                           help="Imbalance threshold for entry (default: 0.3)")
    imb_group.add_argument("--imb-smoothing", type=int, default=3,
                           help="Smoothing window for imbalance signal (default: 3)")
    imb_group.add_argument("--imb-qty", type=float, default=1.0, help="Order quantity (default: 1.0)")
    imb_group.add_argument("--imb-max-position", type=float, default=1.0, help="Max position (default: 1.0)")

    # Dash server
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8050, help="Dash port")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")

    args = parser.parse_args()

    # Загрузка данных
    pre_loaded_snapshots = None
    if args.loader == "binance":
        if not args.binance_dir:
            parser.error("--binance-dir is required when --loader binance")

        start_ts: int | None = None
        end_ts: int | None = None
        if args.binance_date:
            day = datetime.date.fromisoformat(args.binance_date)
            day_start = datetime.datetime(day.year, day.month, day.day, tzinfo=datetime.timezone.utc)
            day_end = day_start + datetime.timedelta(days=1) - datetime.timedelta(milliseconds=1)
            start_ts = int(day_start.timestamp() * 1000)
            end_ts = int(day_end.timestamp() * 1000)

        print(
            f"Loading Binance data: dir={args.binance_dir}, "
            f"symbol={args.binance_symbol}, date={args.binance_date or 'all'}"
        )
        pre_loaded_snapshots = list(load_l2_binance(
            data_dir=args.binance_dir,
            symbol=args.binance_symbol,
            depth=args.depth,
            start_ts=start_ts,
            end_ts=end_ts,
        ))
        print(f"Loaded {len(pre_loaded_snapshots)} snapshots")
    elif not args.l2:
        parser.error("--l2 is required when not using --loader binance")

    initial_form: dict[str, Any] = {
        "l2": args.l2,
        "trades": args.trades,
        "loader": args.loader,
        "snapshots": pre_loaded_snapshots,
        "strategy": args.strategy,
        # MM
        "mm_spread": args.mm_spread,
        "mm_quote_size": args.mm_quote_size,
        # Taker
        "taker_window": args.taker_window,
        "taker_std_mult": args.taker_std_mult,
        "taker_qty": args.taker_qty,
        "taker_cooldown": args.taker_cooldown,
        "taker_max_position": args.taker_max_position,
        # EMA
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "ema_qty": args.ema_qty,
        "ema_max_position": args.ema_max_position,
        "ema_offset": args.ema_offset,
        # Imbalance
        "imb_depth": args.imb_depth,
        "imb_threshold": args.imb_threshold,
        "imb_smoothing": args.imb_smoothing,
        "imb_qty": args.imb_qty,
        "imb_max_position": args.imb_max_position,
    }

    metrics = _run_from_form(initial_form)
    print(f"fills={metrics.num_fills}")
    if metrics.equity_curve:
        print(f"last_equity={metrics.equity_curve[-1][1]:.6f}")

    app = create_dash_app(metrics, _run_from_form, initial_form)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
