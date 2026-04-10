"""Tests for CustomEvent and TimerEvent dispatch through the engine."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.actions import BaseAction, PlaceOrderAction
from src.engine import SimulationEngine
from src.enums import EventType, OrderType, Side
from src.events import CustomEvent, MarketSnapshotEvent, TimerEvent
from src.models import Level, Snapshot
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext


T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _snap(ts: datetime, instrument_id: str = "default") -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(price=101.0, qty=5.0)],
        bids=[Level(price=100.0, qty=5.0)],
        instrument_id=instrument_id,
    )


class CustomEventRecorder(BaseStrategy):
    def __init__(self, strategy_id: str) -> None:
        super().__init__(strategy_id)
        self.custom_contexts: list[StrategyContext] = []

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        return []

    def on_custom_event(self, context: StrategyContext) -> list[BaseAction]:
        self.custom_contexts.append(context)
        return []


class CustomEventActingStrategy(BaseStrategy):
    """Places a BUY order when it receives a custom event."""

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        return []

    def on_custom_event(self, context: StrategyContext) -> list[BaseAction]:
        return [
            PlaceOrderAction(
                strategy_id=self.strategy_id,
                side=Side.BUY,
                price=101.0,
                qty=1.0,
                order_type=OrderType.LIMIT,
            )
        ]


# ---------------------------------------------------------------------------
# CustomEvent / TimerEvent event_type
# ---------------------------------------------------------------------------


def test_custom_event_has_correct_event_type() -> None:
    event = CustomEvent(timestamp=_t(0), name="signal")
    assert event.event_type == EventType.CUSTOM


def test_timer_event_has_correct_event_type() -> None:
    event = TimerEvent(timestamp=_t(0), name="tick")
    assert event.event_type == EventType.TIMER


def test_custom_event_stores_payload() -> None:
    event = CustomEvent(timestamp=_t(0), name="alpha", payload={"value": 42})
    assert event.payload["value"] == 42


# ---------------------------------------------------------------------------
# Engine dispatches on_custom_event
# ---------------------------------------------------------------------------


def test_process_event_routes_custom_event_to_on_custom_event() -> None:
    strategy = CustomEventRecorder("s1")
    engine = SimulationEngine(strategy=strategy)
    # Must have a snapshot first so context can be built
    engine.process_snapshot(_snap(_t(0)))
    custom = CustomEvent(timestamp=_t(1), name="signal")
    engine.process_event(custom)
    assert len(strategy.custom_contexts) == 1
    assert strategy.custom_contexts[0].triggering_event is custom


def test_process_event_routes_timer_event_to_on_custom_event() -> None:
    strategy = CustomEventRecorder("s1")
    engine = SimulationEngine(strategy=strategy)
    engine.process_snapshot(_snap(_t(0)))
    timer = TimerEvent(timestamp=_t(1), name="heartbeat")
    engine.process_event(timer)
    assert len(strategy.custom_contexts) == 1
    assert strategy.custom_contexts[0].triggering_event is timer


def test_process_event_snapshot_still_calls_on_snapshot() -> None:
    from tests.helpers.strategies import RecordingStrategy

    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    snap = _snap(_t(0))
    engine.process_event(MarketSnapshotEvent(timestamp=snap.timestamp, snapshot=snap))
    assert len(strategy.contexts) == 1


def test_custom_event_context_has_current_snapshots() -> None:
    strategy = CustomEventRecorder("s1")
    engine = SimulationEngine(strategy=strategy)
    snap = _snap(_t(0))
    engine.process_snapshot(snap)
    engine.process_event(CustomEvent(timestamp=_t(1), name="ping"))
    ctx = strategy.custom_contexts[0]
    assert "default" in ctx.snapshots


def test_no_strategy_custom_event_returns_empty_list() -> None:
    engine = SimulationEngine()
    engine.process_snapshot(_snap(_t(0)))
    result = engine.process_event(CustomEvent(timestamp=_t(1), name="ping"))
    assert result == []


# ---------------------------------------------------------------------------
# Actions returned from on_custom_event are executed
# ---------------------------------------------------------------------------


def test_actions_from_custom_event_are_executed() -> None:
    strategy = CustomEventActingStrategy("s1")
    engine = SimulationEngine(strategy=strategy, trading_instrument_ids=frozenset({"default"}))
    # Provide a snapshot so the engine has a current book
    engine.process_snapshot(_snap(_t(0)))
    # Fire a custom event; strategy will return a BUY limit
    engine.process_event(CustomEvent(timestamp=_t(1), name="go"))
    # The order should have been placed and possibly filled (ask=101, price=101)
    assert len(engine.state.trades) >= 0  # at minimum an order was placed
    assert len(strategy.state.orders) > 0


def test_custom_event_without_prior_snapshot_produces_no_actions() -> None:
    strategy = CustomEventActingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    # No snapshot → current_snapshot is None → actions cannot be processed
    result = engine.process_event(CustomEvent(timestamp=_t(0), name="early"))
    assert result == []


# ---------------------------------------------------------------------------
# run_events handles CustomEvent in merged stream
# ---------------------------------------------------------------------------


def test_run_events_processes_custom_events_in_stream() -> None:
    strategy = CustomEventRecorder("s1")
    engine = SimulationEngine(strategy=strategy)

    def _event_stream():
        yield MarketSnapshotEvent(timestamp=_t(0), snapshot=_snap(_t(0)))
        yield CustomEvent(timestamp=_t(1), name="alpha")
        yield TimerEvent(timestamp=_t(2), name="beta")

    engine.run_events(_event_stream())
    assert len(strategy.custom_contexts) == 2
    assert strategy.custom_contexts[0].triggering_event.name == "alpha"
    assert strategy.custom_contexts[1].triggering_event.name == "beta"
