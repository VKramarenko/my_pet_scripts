from __future__ import annotations

import argparse

from sim.core.events import MarketSnapshot, MarketTrade
from sim.data.loaders import (
    load_bybit_l2_snapshots,
    load_bybit_trades,
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
from sim.strategy.mm_basic import BasicMMStrategy
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
    return BasicMMStrategy(spread=args.mm_spread, quote_size=args.mm_quote_size)


def run(
    l2_path: str,
    trades_path: str | None = None,
    loader: str = "default",
    strategy: StrategyBase | None = None,
) -> MetricsCollector:
    if loader == "bybit":
        snapshots = list(load_bybit_l2_snapshots(l2_path))
        trades = list(load_bybit_trades(trades_path)) if trades_path else []
    elif loader == "test_data":
        snapshots = list(load_test_data_l2_snapshots(l2_path))
        trades = list(load_test_data_trades(trades_path)) if trades_path else []
    else:
        snapshots = list(load_l2_snapshots(l2_path))
        trades = list(load_trades(trades_path)) if trades_path else []
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
    args = parser.parse_args()
    strategy = _build_strategy(args)
    metrics = run(args.l2, args.trades, loader=args.loader, strategy=strategy)
    print(f"fills={metrics.num_fills}")
    if metrics.equity_curve:
        print(f"last_equity={metrics.equity_curve[-1][1]:.6f}")


if __name__ == "__main__":
    main()
