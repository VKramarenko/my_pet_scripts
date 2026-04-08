from __future__ import annotations

from datetime import timedelta

from src.commission_models import FixedPerTradeCommission, NoCommission
from src.engine import SimulationEngine
from src.risk_limits import StrategyLimits
from src.slippage_models import FixedBpsSlippage, NoSlippage
from tests.helpers.snapshots import make_snapshot
from tests.helpers.strategies import AlwaysPlaceBuyLimitStrategy, CancelStaleOrderStrategy, PassiveBuyOnceStrategy


def test_passive_strategy_limit_gets_filled_and_updates_local_state(base_time) -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0})
    engine = SimulationEngine(strategy=strategy)
    snapshots = [
        make_snapshot(base_time, best_ask=101.0, best_bid=100.0),
        make_snapshot(base_time + timedelta(seconds=1), best_ask=100.0, best_bid=99.0),
    ]
    engine.run(snapshots)
    assert strategy.state.trades
    assert strategy.state.position == 1.0
    assert strategy.state.orders


def test_immediate_fill_strategy_receives_resulting_state_same_run(base_time) -> None:
    strategy = AlwaysPlaceBuyLimitStrategy("s-1", price=101.0, qty=1.0)
    engine = SimulationEngine(strategy=strategy)
    engine.run([make_snapshot(base_time, best_ask=101.0, best_bid=100.0)])
    assert strategy.state.trades
    assert strategy.state.position == 1.0


def test_cancel_logic_strategy_clears_local_active_orders(base_time) -> None:
    strategy = CancelStaleOrderStrategy("s-1", stale_after_steps=1, price=99.0, qty=1.0)
    engine = SimulationEngine(strategy=strategy)
    snapshots = [
        make_snapshot(base_time, best_ask=101.0, best_bid=100.0),
        make_snapshot(base_time + timedelta(seconds=1), best_ask=101.0, best_bid=100.0),
    ]
    engine.run(snapshots)
    assert strategy.state.active_orders == {}


def test_costs_make_result_worse_than_run_without_costs(base_time) -> None:
    snapshots = [
        make_snapshot(base_time, best_ask=101.0, best_bid=100.0),
        make_snapshot(base_time + timedelta(seconds=1), best_ask=100.0, best_bid=99.0),
        make_snapshot(base_time + timedelta(seconds=2), best_ask=102.0, best_bid=101.0),
    ]
    no_cost_strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0})
    cost_strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0})
    no_cost_engine = SimulationEngine(strategy=no_cost_strategy, commission_model=NoCommission(), slippage_model=NoSlippage())
    cost_engine = SimulationEngine(strategy=cost_strategy, commission_model=FixedPerTradeCommission(1.0), slippage_model=FixedBpsSlippage(10.0))
    no_cost_engine.run(snapshots)
    cost_engine.run(snapshots)
    assert cost_strategy.state.equity < no_cost_strategy.state.equity


def test_strategy_under_limits_receives_rejection_and_stays_consistent(base_time) -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 10.0})
    engine = SimulationEngine(strategy=strategy, strategy_limits=StrategyLimits(max_order_qty=1.0))
    engine.run([make_snapshot(base_time, best_ask=101.0, best_bid=100.0)])
    assert strategy.state.active_orders == {}
    assert strategy.state.orders
    assert all(order.status.name == "REJECTED" for order in strategy.state.orders.values())


def test_engine_and_strategy_state_are_semantically_consistent(base_time) -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0})
    engine = SimulationEngine(strategy=strategy)
    snapshots = [
        make_snapshot(base_time, best_ask=101.0, best_bid=100.0),
        make_snapshot(base_time + timedelta(seconds=1), best_ask=100.0, best_bid=99.0),
    ]
    engine.run(snapshots)
    engine_trades = [trade for trade in engine.state.trades if trade.strategy_id == strategy.strategy_id]
    engine_active = {oid: order for oid, order in engine.state.active_orders.items() if order.strategy_id == strategy.strategy_id}
    assert len(strategy.state.trades) == len(engine_trades)
    assert strategy.state.active_orders.keys() == engine_active.keys()
