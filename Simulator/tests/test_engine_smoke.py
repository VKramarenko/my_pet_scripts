from __future__ import annotations

from datetime import UTC, datetime

from src.config import SimulationConfig
from src.engine import SimulationEngine
from src.events import MarketSnapshotEvent
from src.models import Level, Snapshot
from src.state import SimulationState


def make_snapshot() -> Snapshot:
    return Snapshot(
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        asks=[Level(price=101.0, qty=1.0)],
        bids=[Level(price=100.0, qty=1.0)],
    )


def test_engine_is_created_with_config_and_state() -> None:
    config = SimulationConfig()
    state = SimulationState()
    engine = SimulationEngine(config=config, state=state)

    assert engine.config is config
    assert engine.state is state


def test_process_snapshot_updates_current_snapshot_and_event_log() -> None:
    engine = SimulationEngine()
    snapshot = make_snapshot()

    engine.process_snapshot(snapshot)

    assert engine.state.current_snapshot is snapshot
    assert isinstance(engine.state.event_log[-1], MarketSnapshotEvent)


def test_process_resting_orders_can_be_called() -> None:
    engine = SimulationEngine()
    snapshot = make_snapshot()

    events = engine._process_resting_orders(snapshot)

    assert events == []


def test_build_strategy_context_returns_current_snapshot() -> None:
    engine = SimulationEngine()
    snapshot = make_snapshot()
    engine.process_snapshot(snapshot)

    context = engine._build_strategy_context()

    assert context["current_snapshot"] is snapshot
    assert context["active_orders"] == {}
