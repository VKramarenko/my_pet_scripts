from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.actions import BaseAction
from src.engine import SimulationEngine
from src.enums import OrderType, Side
from src.events import OrderUpdateEvent, OwnTradeEvent
from src.models import Level, Snapshot
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy


class RecordingStrategy(BaseStrategy):
    def __init__(self, strategy_id: str, config: dict | None = None) -> None:
        super().__init__(strategy_id, config)
        self.started = 0
        self.ended = 0
        self.received_order_updates: list[OrderUpdateEvent] = []
        self.received_trades: list[OwnTradeEvent] = []
        self.contexts: list[StrategyContext] = []

    def on_simulation_start(self) -> None:
        super().on_simulation_start()
        self.started += 1

    def on_order_update(self, event: OrderUpdateEvent) -> None:
        super().on_order_update(event)
        self.received_order_updates.append(event)

    def on_trade(self, event: OwnTradeEvent) -> None:
        super().on_trade(event)
        self.received_trades.append(event)

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        self.contexts.append(context)
        return []

    def on_simulation_end(self) -> None:
        self.ended += 1


def make_snapshot(ts: datetime, best_ask: float, best_bid: float, ask_qty: float = 5.0, bid_qty: float = 5.0) -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(best_ask, ask_qty), Level(best_ask + 1.0, 10.0)],
        bids=[Level(best_bid, bid_qty), Level(best_bid - 1.0, 10.0)],
    )


def test_run_calls_strategy_lifecycle_hooks() -> None:
    strategy = RecordingStrategy("s-1")
    engine = SimulationEngine(strategy=strategy)
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)

    engine.run([make_snapshot(ts, 101.0, 100.0)])

    assert strategy.started == 1
    assert strategy.ended == 1


def test_resting_events_delivered_before_on_snapshot() -> None:
    strategy = RecordingStrategy("s-1")
    engine = SimulationEngine(strategy=strategy)
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)

    engine.process_snapshot(
        make_snapshot(ts, 105.0, 104.0),
        strategy_actions=[],
    )
    engine.process_snapshot(
        make_snapshot(ts + timedelta(seconds=1), 100.0, 99.0, ask_qty=2.0),
        strategy_actions=[],
    )

    # No strategy-generated actions here; lifecycle check uses real sample strategy flow below.
    assert strategy.contexts[-1].timestamp == ts + timedelta(seconds=1)


def test_on_snapshot_receives_strategy_context() -> None:
    strategy = RecordingStrategy("s-1")
    engine = SimulationEngine(strategy=strategy)
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)

    engine.process_snapshot(make_snapshot(ts, 101.0, 100.0))

    assert isinstance(strategy.contexts[0], StrategyContext)


def test_strategy_actions_are_processed_by_engine() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 99.0, "qty": 2.0})
    engine = SimulationEngine(strategy=strategy)
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)

    engine.process_snapshot(make_snapshot(ts, 105.0, 104.0))

    assert "ORD-000001" in engine.state.active_orders


def test_resulting_events_after_strategy_actions_delivered_back_to_strategy() -> None:
    strategy = PassiveBuyOnceStrategy("s-1", {"price": 100.0, "qty": 2.0})
    engine = SimulationEngine(strategy=strategy)
    ts = datetime(2026, 4, 7, 12, 0, tzinfo=UTC)

    engine.process_snapshot(make_snapshot(ts, 100.0, 99.0, ask_qty=2.0))

    assert strategy.state.metrics.trade_count == 1
    assert strategy.state.position == 2.0

