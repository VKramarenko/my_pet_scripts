from __future__ import annotations

import argparse

from sim.core.events import MarketSnapshot, MarketTrade
from sim.data.loaders import (
    load_bybit_l2_snapshots,
    load_bybit_trades,
    load_l2_binance,
    load_l2_snapshots,
    load_test_data_l2_snapshots,
    load_test_data_trades,
    load_trades,
)
from sim.data.normalizers import merge_streams
from sim.execution.fill_model_touch import TouchFillModel
from sim.exchange.exchange_sim import ExchangeSim, PlaceOrderRequest
from sim.market.orderbook_l2 import OrderBookL2
from sim.portfolio.metrics import MetricsCollector
from sim.portfolio.portfolio import Portfolio
from sim.strategy.ema_crossover import EmaCrossoverStrategy
from sim.strategy.mm_basic import BasicMMStrategy
from sim.strategy.ob_imbalance import ObImbalanceStrategy
from sim.strategy.strategy_base import CancelRequest, StrategyBase
from sim.strategy.taker_bollinger import TakerBollingerStrategy


def _build_strategy(args: argparse.Namespace) -> StrategyBase:
    if args.strategy == "taker":
        return TakerBollingerStrategy(
            window=args.taker_window,
            std_mult=args.taker_std_mult,
            order_qty=args.taker_qty,
            cooldown=args.taker_cooldown,
            max_position=args.taker_max_position,
        )
    if args.strategy == "ema":
        return EmaCrossoverStrategy(
            fast_window=args.ema_fast,
            slow_window=args.ema_slow,
            order_qty=args.ema_qty,
            max_position=args.ema_max_position,
            limit_offset=args.ema_offset,
        )
    if args.strategy == "imbalance":
        return ObImbalanceStrategy(
            depth=args.imb_depth,
            threshold=args.imb_threshold,
            smoothing=args.imb_smoothing,
            order_qty=args.imb_qty,
            max_position=args.imb_max_position,
        )
    return BasicMMStrategy(spread=args.mm_spread, quote_size=args.mm_quote_size)


def run(
    l2_path: str | None = None,
    trades_path: str | None = None,
    loader: str = "default",
    strategy: StrategyBase | None = None,
    snapshots: list[MarketSnapshot] | None = None,
    trades: list[MarketTrade] | None = None,
) -> MetricsCollector:
    if snapshots is None:
        if loader == "bybit":
            snapshots = list(load_bybit_l2_snapshots(l2_path))
            trades = list(load_bybit_trades(trades_path)) if trades_path else []
        elif loader == "test_data":
            snapshots = list(load_test_data_l2_snapshots(l2_path))
            trades = list(load_test_data_trades(trades_path)) if trades_path else []
        else:
            snapshots = list(load_l2_snapshots(l2_path))
            trades = list(load_trades(trades_path)) if trades_path else []
    if trades is None:
        trades = []
    events = merge_streams(snapshots, trades)
    book = OrderBookL2(depth=10)
    exchange = ExchangeSim(book=book, fill_model=TouchFillModel())
    portfolio = Portfolio()
    metrics = MetricsCollector()
    strategy = strategy or BasicMMStrategy(spread=1.0, quote_size=1.0)

    for event in events:
        if isinstance(event, MarketSnapshot):
            actions = strategy.on_snapshot(event.ts, book, exchange.active_orders(event.ts))
            for action in actions:
                if isinstance(action, PlaceOrderRequest):
                    exchange.submit_place(action, event.ts)
                elif isinstance(action, CancelRequest):
                    exchange.submit_cancel(action.order_id, event.ts)
        fills = []
        if isinstance(event, (MarketSnapshot, MarketTrade)):
            fills = exchange.on_market_event(event)
        if isinstance(event, MarketSnapshot):
            book.update_from_snapshot(event)
            mid = book.mid()
            if mid is not None:
                metrics.on_tob(event.ts, mid)
        if isinstance(event, MarketTrade):
            metrics.on_market_trade(
                event.ts, event.price, event.size, event.side
            )
        for fill in fills:
            order = exchange.order(fill.order_id)
            if order is None:
                continue
            portfolio.apply_fill(fill, order.side)
            strategy.on_fill(fill)
            metrics.on_fill(fill, order.side)
        eq = portfolio.equity(book.mid())
        if eq is not None:
            metrics.on_equity(event.ts, eq)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple backtest.")

    # Data source
    parser.add_argument("--l2", help="Path to L2 snapshots (for default/bybit/test_data loaders)")
    parser.add_argument("--trades", help="Path to trades (optional, for default/bybit/test_data loaders)")
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

    ema_group = parser.add_argument_group("EMA Crossover strategy (--strategy ema)")
    ema_group.add_argument("--ema-fast", type=int, default=10, help="Fast EMA window (default: 10)")
    ema_group.add_argument("--ema-slow", type=int, default=30, help="Slow EMA window (default: 30)")
    ema_group.add_argument("--ema-qty", type=float, default=1.0, help="Order quantity (default: 1.0)")
    ema_group.add_argument("--ema-max-position", type=float, default=1.0, help="Max position (default: 1.0)")
    ema_group.add_argument("--ema-offset", type=float, default=0.0, help="Limit price offset from best bid/ask (default: 0.0)")

    imb_group = parser.add_argument_group("Orderbook Imbalance strategy (--strategy imbalance)")
    imb_group.add_argument("--imb-depth", type=int, default=5, help="Orderbook levels to consider (default: 5)")
    imb_group.add_argument("--imb-threshold", type=float, default=0.3, help="Imbalance threshold for entry (default: 0.3)")
    imb_group.add_argument("--imb-smoothing", type=int, default=3, help="Smoothing window for imbalance signal (default: 3)")
    imb_group.add_argument("--imb-qty", type=float, default=1.0, help="Order quantity (default: 1.0)")
    imb_group.add_argument("--imb-max-position", type=float, default=1.0, help="Max position (default: 1.0)")
    args = parser.parse_args()

    strategy = _build_strategy(args)

    if args.loader == "binance":
        if not args.binance_dir:
            parser.error("--binance-dir is required when --loader binance")

        start_ts: int | None = None
        end_ts: int | None = None
        if args.binance_date:
            import datetime
            day = datetime.date.fromisoformat(args.binance_date)
            day_start = datetime.datetime(day.year, day.month, day.day, tzinfo=datetime.timezone.utc)
            day_end = day_start + datetime.timedelta(days=1) - datetime.timedelta(milliseconds=1)
            start_ts = int(day_start.timestamp() * 1000)
            end_ts = int(day_end.timestamp() * 1000)

        print(
            f"Loading Binance data: dir={args.binance_dir}, "
            f"symbol={args.binance_symbol}, date={args.binance_date or 'all'}"
        )
        snapshots = list(load_l2_binance(
            data_dir=args.binance_dir,
            symbol=args.binance_symbol,
            depth=args.depth,
            start_ts=start_ts,
            end_ts=end_ts,
        ))
        print(f"Loaded {len(snapshots)} snapshots")
        metrics = run(snapshots=snapshots, strategy=strategy)
    else:
        if not args.l2:
            parser.error("--l2 is required when not using --loader binance")
        metrics = run(args.l2, args.trades, loader=args.loader, strategy=strategy)

    print(f"fills={metrics.num_fills}")
    if metrics.equity_curve:
        print(f"last_equity={metrics.equity_curve[-1][1]:.6f}")


if __name__ == "__main__":
    main()
