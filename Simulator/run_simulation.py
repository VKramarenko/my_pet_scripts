from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.commission_models import BpsCommission, FixedPerTradeCommission, NoCommission
from src.data_loader import iter_snapshots_csv, load_snapshots_csv
from src.engine import SimulationEngine
from src.multi_feed import merge_snapshot_feeds
from src.reporting import (
    build_execution_report,
    format_execution_report,
    write_execution_fills_csv,
    write_execution_orders_csv,
    write_execution_report_json,
)
from src.risk_limits import StrategyLimits
from src.slippage_models import FixedBpsSlippage, NoSlippage
from src.strategy.runtime import available_strategy_names, build_strategy
from src.validation import CSVSnapshotLoaderConfig


def infer_csv_depth(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

    if not header:
        raise ValueError(f"CSV file {csv_path} is empty or missing a header row.")

    columns = set(header)
    inferred_depth = 0
    level = 1
    while True:
        required_columns = {
            f"ask_price_{level}",
            f"bid_price_{level}",
            f"ask_size_{level}",
            f"bid_size_{level}",
        }
        if not required_columns.issubset(columns):
            break
        inferred_depth = level
        level += 1

    if inferred_depth == 0:
        raise ValueError(
            f"Could not infer a valid order book depth from CSV header in {csv_path}."
        )

    return inferred_depth


def peek_instrument_id(csv_path: Path) -> str:
    """Read the first data row and return its `symbol` value, or the filename stem."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader, None)
    if first_row is None:
        return csv_path.stem
    symbol = str(first_row.get("symbol", "") or "").strip()
    return symbol if symbol else csv_path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simulator backtest with a selected strategy.")

    # Single-book mode (backward compat)
    parser.add_argument("--csv", help="Path to snapshots CSV file (single-book mode).")

    # Multi-book mode
    parser.add_argument(
        "--trading-csv",
        action="append",
        dest="trading_csvs",
        metavar="PATH",
        help="Trading instrument CSV (repeatable). Instrument ID read from 'symbol' column.",
    )
    parser.add_argument(
        "--info-csv",
        action="append",
        dest="info_csvs",
        metavar="PATH",
        help="Info-only (non-trading) instrument CSV (repeatable).",
    )

    parser.add_argument(
        "--depth",
        type=int,
        help="Book depth in CSV. If omitted, inferred from the maximum complete header depth.",
    )
    parser.add_argument("--strategy", required=True, choices=available_strategy_names(), help="Strategy name.")
    parser.add_argument("--strategy-id", default="strategy-1", help="Strategy identifier.")

    parser.add_argument("--qty", type=float, default=1.0, help="Default order quantity for strategies.")
    parser.add_argument("--price", type=float, default=100.0, help="Passive price for passive strategy.")

    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30.0)
    parser.add_argument("--overbought", type=float, default=70.0)
    parser.add_argument("--order-ttl-seconds", type=float, default=5.0)

    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=5)

    parser.add_argument("--commission-model", choices=["none", "fixed", "bps"], default="none")
    parser.add_argument("--commission-amount", type=float, default=0.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)

    parser.add_argument("--slippage-model", choices=["none", "fixed_bps"], default="none")
    parser.add_argument("--slippage-bps", type=float, default=0.0)

    parser.add_argument("--max-order-qty", type=float, default=None)
    parser.add_argument("--execution-report-json", default=None, help="Optional path to save execution report as JSON.")
    parser.add_argument("--execution-orders-csv", default=None, help="Optional path to save order-level execution report as CSV.")
    parser.add_argument("--execution-fills-csv", default=None, help="Optional path to save fill-level execution report as CSV.")
    return parser


def build_commission_model(args: argparse.Namespace):
    if args.commission_model == "fixed":
        return FixedPerTradeCommission(args.commission_amount)
    if args.commission_model == "bps":
        return BpsCommission(args.commission_bps)
    return NoCommission()


def build_slippage_model(args: argparse.Namespace):
    if args.slippage_model == "fixed_bps":
        return FixedBpsSlippage(args.slippage_bps)
    return NoSlippage()


def build_strategy_config(args: argparse.Namespace, trading_instrument_ids: list[str] | None = None) -> dict:
    config = {
        "qty": args.qty,
        "price": args.price,
        "rsi_period": args.rsi_period,
        "oversold": args.oversold,
        "overbought": args.overbought,
        "order_ttl_seconds": args.order_ttl_seconds,
        "short_window": args.short_window,
        "long_window": args.long_window,
    }
    if trading_instrument_ids is not None:
        config["trading_instrument_ids"] = trading_instrument_ids
    return config


def format_summary(engine: SimulationEngine) -> str:
    if engine.strategy is None:
        raise ValueError("engine strategy must be set")
    strategy = engine.strategy
    lines = [
        f"strategy={strategy.__class__.__name__}",
        f"orders_total={len(strategy.state.orders)}",
        f"trades_total={len(strategy.state.trades)}",
        f"position={strategy.state.position}",
        f"cash={strategy.state.cash:.6f}",
        f"realized_pnl={strategy.state.realized_pnl:.6f}",
        f"unrealized_pnl={strategy.state.unrealized_pnl:.6f}",
        f"equity={strategy.state.equity:.6f}",
    ]
    if strategy.state.positions:
        for iid, pos in sorted(strategy.state.positions.items()):
            lines.append(f"  position[{iid}]={pos}")
    return "\n".join(lines)


def _run_single_book(args: argparse.Namespace) -> SimulationEngine:
    csv_path = Path(args.csv)
    depth = args.depth if args.depth is not None else infer_csv_depth(csv_path)
    loader_config = CSVSnapshotLoaderConfig(depth=depth)
    snapshots = load_snapshots_csv(csv_path, loader_config)

    # Use the actual instrument_id from the data so orders are not rejected
    trading_id = snapshots[0].instrument_id if snapshots else "default"

    strategy = build_strategy(args.strategy, args.strategy_id, build_strategy_config(args))
    engine = SimulationEngine(
        strategy=strategy,
        commission_model=build_commission_model(args),
        slippage_model=build_slippage_model(args),
        strategy_limits=StrategyLimits(max_order_qty=args.max_order_qty),
        trading_instrument_ids=frozenset({trading_id}),
    )
    engine.run(snapshots)
    return engine


def _run_multi_book(args: argparse.Namespace) -> SimulationEngine:
    trading_paths = [Path(p) for p in (args.trading_csvs or [])]
    info_paths = [Path(p) for p in (args.info_csvs or [])]
    all_paths = trading_paths + info_paths

    # Infer shared depth from first file (all files must share the same depth)
    reference_path = all_paths[0]
    depth = args.depth if args.depth is not None else infer_csv_depth(reference_path)
    loader_config = CSVSnapshotLoaderConfig(depth=depth)

    # Build feeds dict: {instrument_id -> snapshot iterable}
    feeds: dict[str, object] = {}
    trading_ids: list[str] = []

    for path in trading_paths:
        iid = peek_instrument_id(path)
        feeds[iid] = iter_snapshots_csv(path, loader_config)
        trading_ids.append(iid)

    for path in info_paths:
        iid = peek_instrument_id(path)
        feeds[iid] = iter_snapshots_csv(path, loader_config)

    merged = merge_snapshot_feeds(feeds)

    strategy = build_strategy(
        args.strategy,
        args.strategy_id,
        build_strategy_config(args, trading_instrument_ids=trading_ids),
    )
    engine = SimulationEngine(
        strategy=strategy,
        commission_model=build_commission_model(args),
        slippage_model=build_slippage_model(args),
        strategy_limits=StrategyLimits(max_order_qty=args.max_order_qty),
        trading_instrument_ids=frozenset(trading_ids),
    )
    engine.run_events(merged)
    return engine


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    has_multi = bool(args.trading_csvs or args.info_csvs)
    has_single = bool(args.csv)

    if has_multi and has_single:
        parser.error("Use either --csv (single-book) or --trading-csv/--info-csv (multi-book), not both.")
    if not has_multi and not has_single:
        parser.error("Provide --csv or at least one --trading-csv.")

    engine = _run_multi_book(args) if has_multi else _run_single_book(args)

    report = build_execution_report(engine, strategy_id=engine.strategy.strategy_id)
    print(format_summary(engine))
    print()
    print(format_execution_report(report))

    if args.execution_report_json:
        write_execution_report_json(report, args.execution_report_json)
    if args.execution_orders_csv:
        write_execution_orders_csv(report, args.execution_orders_csv)
    if args.execution_fills_csv:
        write_execution_fills_csv(report, args.execution_fills_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
