from __future__ import annotations

import argparse
from typing import Any

from run_backtest import run_from_config
from sim.config import (
    BacktestConfig,
    DashboardConfig,
    DataConfig,
    EmaConfig,
    ImbalanceConfig,
    MMConfig,
    StrategyConfig,
    TakerConfig,
    load_backtest_config,
)
from sim.visualization import create_dash_app


def _config_to_form(config: BacktestConfig) -> dict[str, Any]:
    return {
        "l2": config.data.l2_path,
        "trades": config.data.trades_path,
        "symbol": config.data.symbol,
        "strategy": config.strategy.name,
        "mm_spread": config.strategy.mm.spread,
        "mm_quote_size": config.strategy.mm.quote_size,
        "taker_window": config.strategy.taker.window,
        "taker_std_mult": config.strategy.taker.std_mult,
        "taker_qty": config.strategy.taker.qty,
        "taker_cooldown": config.strategy.taker.cooldown,
        "taker_max_position": config.strategy.taker.max_position,
        "ema_fast": config.strategy.ema.fast,
        "ema_slow": config.strategy.ema.slow,
        "ema_qty": config.strategy.ema.qty,
        "ema_max_position": config.strategy.ema.max_position,
        "ema_offset": config.strategy.ema.offset,
        "imb_depth": config.strategy.imbalance.depth,
        "imb_threshold": config.strategy.imbalance.threshold,
        "imb_smoothing": config.strategy.imbalance.smoothing,
        "imb_qty": config.strategy.imbalance.qty,
        "imb_max_position": config.strategy.imbalance.max_position,
    }


def _form_to_config(form: dict[str, Any], dashboard: DashboardConfig) -> BacktestConfig:
    return BacktestConfig(
        data=DataConfig(
            l2_path=str(form["l2"]),
            trades_path=str(form["trades"]) if form.get("trades") else None,
            symbol=str(form["symbol"]) if form.get("symbol") else None,
        ),
        strategy=StrategyConfig(
            name=str(form["strategy"]),
            mm=MMConfig(
                spread=float(form["mm_spread"]),
                quote_size=float(form["mm_quote_size"]),
            ),
            taker=TakerConfig(
                window=int(form["taker_window"]),
                std_mult=float(form["taker_std_mult"]),
                qty=float(form["taker_qty"]),
                cooldown=float(form["taker_cooldown"]),
                max_position=float(form["taker_max_position"]),
            ),
            ema=EmaConfig(
                fast=int(form["ema_fast"]),
                slow=int(form["ema_slow"]),
                qty=float(form["ema_qty"]),
                max_position=float(form["ema_max_position"]),
                offset=float(form["ema_offset"]),
            ),
            imbalance=ImbalanceConfig(
                depth=int(form["imb_depth"]),
                threshold=float(form["imb_threshold"]),
                smoothing=int(form["imb_smoothing"]),
                qty=float(form["imb_qty"]),
                max_position=float(form["imb_max_position"]),
            ),
        ),
        dashboard=dashboard,
    )


def _run_from_form(form: dict[str, Any], dashboard: DashboardConfig):
    return run_from_config(_form_to_config(form, dashboard))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a backtest dashboard from a JSON config."
    )
    parser.add_argument(
        "--config",
        help="Path to JSON config. Defaults to ./backtest_config.json when it exists.",
    )
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    initial_form = _config_to_form(config)

    metrics = run_from_config(config)
    print(f"fills={metrics.num_fills}")
    if metrics.equity_curve:
        print(f"last_equity={metrics.equity_curve[-1][1]:.6f}")

    app = create_dash_app(
        metrics,
        lambda form: _run_from_form(form, config.dashboard),
        initial_form,
    )
    app.run(
        host=config.dashboard.host,
        port=config.dashboard.port,
        debug=config.dashboard.debug,
    )


if __name__ == "__main__":
    main()
