from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.commission_models import FixedPerTradeCommission
from src.engine import SimulationEngine
from src.models import Level, Snapshot
from src.risk_limits import StrategyLimits
from src.slippage_models import FixedBpsSlippage
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy


def make_snapshot(ts: datetime, asks: list[tuple[float, float]], bids: list[tuple[float, float]]) -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(price, qty) for price, qty in asks],
        bids=[Level(price, qty) for price, qty in bids],
    )


def test_single_step_trade_flows_through_costs_and_accounting() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0})
    engine = SimulationEngine(
        strategy=strategy,
        commission_model=FixedPerTradeCommission(1.0),
        slippage_model=FixedBpsSlippage(10.0),
    )
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(make_snapshot(ts, [(100.0, 1.0)], [(99.0, 1.0)]))
    assert strategy.state.position == 1.0
    assert strategy.state.cash < -100.0
    assert strategy.state.metrics.trade_count == 1


def test_after_step_equity_uses_mark_to_mid() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0})
    engine = SimulationEngine(strategy=strategy)
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(make_snapshot(ts, [(100.0, 1.0)], [(99.0, 1.0)]))
    engine.process_snapshot(make_snapshot(ts + timedelta(seconds=1), [(102.0, 1.0)], [(100.0, 1.0)]))
    assert strategy.state.equity > strategy.state.cash


def test_identical_input_is_deterministic() -> None:
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    snapshots = [
        make_snapshot(ts, [(100.0, 1.0)], [(99.0, 1.0)]),
        make_snapshot(ts + timedelta(seconds=1), [(102.0, 1.0)], [(100.0, 1.0)]),
    ]
    engine1 = SimulationEngine(strategy=PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0}))
    engine2 = SimulationEngine(strategy=PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0}))
    engine1.run(snapshots)
    engine2.run(snapshots)
    assert engine1.strategy.state.equity == engine2.strategy.state.equity


def test_rejected_order_by_limits_does_not_enter_active_orders() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 10.0})
    engine = SimulationEngine(strategy=strategy, strategy_limits=StrategyLimits(max_order_qty=1.0))
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)
    engine.process_snapshot(make_snapshot(ts, [(101.0, 1.0)], [(100.0, 1.0)]))
    assert engine.state.active_orders == {}
