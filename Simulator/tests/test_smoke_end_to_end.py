from __future__ import annotations

from src.commission_models import FixedPerTradeCommission
from src.data_loader import CSVSnapshotLoaderConfig, load_snapshots_csv
from src.engine import SimulationEngine
from src.risk_limits import StrategyLimits
from src.slippage_models import FixedBpsSlippage
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy


def test_small_csv_with_passive_strategy_runs_without_errors(tmp_path) -> None:
    csv_path = tmp_path / "book.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,101,100,1,1\n"
        "2024-01-01T10:00:01,100,99,1,1\n"
        "2024-01-01T10:00:02,102,101,1,1\n",
        encoding="utf-8",
    )
    snapshots = load_snapshots_csv(csv_path, CSVSnapshotLoaderConfig(depth=1))
    engine = SimulationEngine(strategy=PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0}))
    state = engine.run(snapshots)
    assert state.event_log
    assert engine.strategy.state.metrics is not None


def test_limit_resting_fill_with_costs_completes_full_run(tmp_path) -> None:
    csv_path = tmp_path / "book.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,101,100,1,1\n"
        "2024-01-01T10:00:01,100,99,1,1\n",
        encoding="utf-8",
    )
    snapshots = load_snapshots_csv(csv_path, CSVSnapshotLoaderConfig(depth=1))
    engine = SimulationEngine(
        strategy=PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0}),
        commission_model=FixedPerTradeCommission(1.0),
        slippage_model=FixedBpsSlippage(10.0),
        strategy_limits=StrategyLimits(max_order_qty=2.0),
    )
    state = engine.run(snapshots)
    assert state.trades
    assert engine.strategy.state.cash < 0
    assert engine.strategy.state.equity != 0
