from __future__ import annotations

from src.engine import SimulationEngine
from tests.helpers.snapshots import make_snapshot
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy


def test_identical_runs_produce_identical_events_and_metrics(base_time) -> None:
    snapshots = [
        make_snapshot(base_time, best_ask=101.0, best_bid=100.0),
        make_snapshot(base_time.replace(second=1), best_ask=100.0, best_bid=99.0),
    ]
    engine1 = SimulationEngine(strategy=PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0}))
    engine2 = SimulationEngine(strategy=PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0}))

    state1 = engine1.run(snapshots)
    state2 = engine2.run(snapshots)

    assert [(trade.trade_id, trade.order_id, trade.price, trade.qty) for trade in state1.trades] == [
        (trade.trade_id, trade.order_id, trade.price, trade.qty) for trade in state2.trades
    ]
    assert [(event.event_type.value, event.timestamp) for event in state1.event_log] == [
        (event.event_type.value, event.timestamp) for event in state2.event_log
    ]
    assert engine1.strategy.state.metrics.turnover == engine2.strategy.state.metrics.turnover
    assert state1.order_sequence == state2.order_sequence
    assert state1.trade_sequence == state2.trade_sequence


def test_deterministic_id_generation(base_time) -> None:
    engine = SimulationEngine(strategy=PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 1.0}))
    engine.run(
        [
            make_snapshot(base_time, best_ask=101.0, best_bid=100.0),
            make_snapshot(base_time.replace(second=1), best_ask=100.0, best_bid=99.0),
        ]
    )
    assert list(engine.state.completed_orders) == ["ORD-000001"]
    assert [trade.trade_id for trade in engine.state.trades] == ["TRD-000001"]

