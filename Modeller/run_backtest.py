from __future__ import annotations

import argparse

from sim.config import BacktestConfig, StrategyConfig, load_backtest_config
from sim.core.events import MarketSnapshot, MarketTrade
from sim.data.loaders import load_trades, load_wide_l2_snapshots
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


def _format_levels(levels: list[tuple[float, float]], depth: int) -> str:
    clipped = levels[:depth]
    if not clipped:
        return "-"
    return ", ".join(f"{price:.6f}@{size:.6f}" for price, size in clipped)


def _snapshot_book_view(snapshot: MarketSnapshot, depth: int) -> str:
    bids = sorted(snapshot.bids, key=lambda x: x[0], reverse=True)
    asks = sorted(snapshot.asks, key=lambda x: x[0])
    return (
        f"best_bid={bids[0][0]:.6f}" if bids else "best_bid=-"
    ) + " | " + (
        f"best_ask={asks[0][0]:.6f}" if asks else "best_ask=-"
    ) + " | " + (
        f"bids[{depth}]={_format_levels(bids, depth)} | "
        f"asks[{depth}]={_format_levels(asks, depth)}"
    )


def _book_view(book: OrderBookL2, depth: int) -> str:
    bids = book.top_n("BUY", depth)
    asks = book.top_n("SELL", depth)
    return (
        f"best_bid={bids[0][0]:.6f}" if bids else "best_bid=-"
    ) + " | " + (
        f"best_ask={asks[0][0]:.6f}" if asks else "best_ask=-"
    ) + " | " + (
        f"bids[{depth}]={_format_levels(bids, depth)} | "
        f"asks[{depth}]={_format_levels(asks, depth)}"
    )


def _log_strategy_action(
    action: PlaceOrderRequest | CancelRequest,
    snapshot: MarketSnapshot,
    book_levels: int,
) -> None:
    prefix = f"[strategy ts={snapshot.ts:.6f}]"
    if isinstance(action, PlaceOrderRequest):
        price_text = "MKT" if action.price is None else f"{action.price:.6f}"
        print(
            f"{prefix} place id={action.order_id} side={action.side} "
            f"type={action.type} qty={action.qty:.6f} price={price_text} | "
            f"{_snapshot_book_view(snapshot, book_levels)}"
        )
        print()
        return
    print(
        f"{prefix} cancel id={action.order_id} | "
        f"{_snapshot_book_view(snapshot, book_levels)}"
    )
    print()


def _log_fill(
    fill,
    order_side: str,
    portfolio: Portfolio,
    mark_price: float | None,
    book: OrderBookL2,
    book_levels: int,
) -> None:
    equity = portfolio.equity(mark_price)
    equity_text = "-" if equity is None else f"{equity:.6f}"
    print(
        f"[fill ts={fill.ts:.6f}] order_id={fill.order_id} side={order_side} "
        f"price={fill.price:.6f} qty={fill.qty:.6f} fee={fill.fee:.6f} "
        f"liquidity={fill.liquidity} position={portfolio.position:.6f} "
        f"cash={portfolio.cash:.6f} equity={equity_text} | "
        f"{_book_view(book, book_levels)}"
    )
    print()


def _build_strategy(config: StrategyConfig | None = None) -> StrategyBase:
    strategy_config = config or StrategyConfig()
    if strategy_config.name == "taker":
        return TakerBollingerStrategy(
            window=strategy_config.taker.window,
            std_mult=strategy_config.taker.std_mult,
            order_qty=strategy_config.taker.qty,
            cooldown=strategy_config.taker.cooldown,
            max_position=strategy_config.taker.max_position,
        )
    if strategy_config.name == "ema":
        return EmaCrossoverStrategy(
            fast_window=strategy_config.ema.fast,
            slow_window=strategy_config.ema.slow,
            order_qty=strategy_config.ema.qty,
            max_position=strategy_config.ema.max_position,
            limit_offset=strategy_config.ema.offset,
        )
    if strategy_config.name == "imbalance":
        return ObImbalanceStrategy(
            depth=strategy_config.imbalance.depth,
            threshold=strategy_config.imbalance.threshold,
            smoothing=strategy_config.imbalance.smoothing,
            order_qty=strategy_config.imbalance.qty,
            max_position=strategy_config.imbalance.max_position,
        )
    return BasicMMStrategy(
        spread=strategy_config.mm.spread,
        quote_size=strategy_config.mm.quote_size,
    )


def run(
    l2_path: str | None = None,
    trades_path: str | None = None,
    strategy: StrategyBase | None = None,
    symbol: str | None = None,
    snapshots: list[MarketSnapshot] | None = None,
    trades: list[MarketTrade] | None = None,
    console_level: int = 1,
    console_book_levels: int = 1,
) -> MetricsCollector:
    if snapshots is None:
        if not l2_path:
            raise ValueError("l2_path is required when snapshots are not passed explicitly")
        snapshots = list(load_wide_l2_snapshots(l2_path, symbol_filter=symbol))
        trades = list(load_trades(trades_path)) if trades_path else []
    if trades is None:
        trades = []

    events = merge_streams(snapshots, trades)
    book = OrderBookL2(depth=10)
    exchange = ExchangeSim(book=book, fill_model=TouchFillModel())
    portfolio = Portfolio()
    metrics = MetricsCollector()
    strategy = strategy or BasicMMStrategy()

    for event in events:
        if isinstance(event, MarketSnapshot):
            actions = strategy.on_snapshot(event.ts, book, exchange.active_orders(event.ts))
            for action in actions:
                if console_level >= 2:
                    _log_strategy_action(action, event, console_book_levels)
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
            metrics.on_market_trade(event.ts, event.price, event.size, event.side)
        for fill in fills:
            order = exchange.order(fill.order_id)
            if order is None:
                continue
            portfolio.apply_fill(fill, order.side)
            strategy.on_fill(fill)
            metrics.on_fill(fill, order.side)
            if console_level >= 3:
                _log_fill(
                    fill,
                    order.side,
                    portfolio,
                    book.mid(),
                    book,
                    console_book_levels,
                )
        eq = portfolio.equity(book.mid())
        if eq is not None:
            metrics.on_equity(event.ts, eq)
    return metrics


def run_from_config(
    config: BacktestConfig,
    snapshots: list[MarketSnapshot] | None = None,
    trades: list[MarketTrade] | None = None,
) -> MetricsCollector:
    return run(
        l2_path=config.data.l2_path,
        trades_path=config.data.trades_path,
        strategy=_build_strategy(config.strategy),
        symbol=config.data.symbol,
        snapshots=snapshots,
        trades=trades,
        console_level=config.console.level,
        console_book_levels=config.console.book_levels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest from wide CSV files.")
    parser.add_argument(
        "--config",
        help="Path to JSON config. Defaults to ./backtest_config.json when it exists.",
    )
    args = parser.parse_args()

    config = load_backtest_config(args.config)
    metrics = run_from_config(config)
    print(f"fills={metrics.num_fills}")
    if metrics.equity_curve:
        print(f"last_equity={metrics.equity_curve[-1][1]:.6f}")


if __name__ == "__main__":
    main()
