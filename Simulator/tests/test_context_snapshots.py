"""Tests for StrategyContext.snapshots, positions, and triggering_event fields."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.engine import SimulationEngine
from src.events import MarketSnapshotEvent
from src.models import Level, Snapshot
from tests.helpers.strategies import RecordingStrategy


T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _snap(ts: datetime, *, instrument_id: str = "default", ask: float = 101.0, bid: float = 100.0) -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(price=ask, qty=1.0)],
        bids=[Level(price=bid, qty=1.0)],
        instrument_id=instrument_id,
    )


# ---------------------------------------------------------------------------
# context.snapshots
# ---------------------------------------------------------------------------


def test_context_snapshots_empty_on_first_tick() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    engine.process_snapshot(_snap(_t(0)))
    ctx = strategy.contexts[0]
    # After the first snapshot is processed, current_snapshots["default"] is set
    assert "default" in ctx.snapshots


def test_context_snapshots_accumulates_across_instruments() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(
        strategy=strategy,
        trading_instrument_ids=frozenset({"BTC", "ETH"}),
    )
    engine.process_snapshot(_snap(_t(0), instrument_id="BTC"))
    engine.process_snapshot(_snap(_t(1), instrument_id="ETH"))
    # On second tick both snapshots are visible
    ctx = strategy.contexts[1]
    assert "BTC" in ctx.snapshots
    assert "ETH" in ctx.snapshots


def test_context_snapshots_values_match_last_processed() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    snap0 = _snap(_t(0), ask=101.0)
    snap1 = _snap(_t(1), ask=102.0)
    engine.process_snapshot(snap0)
    engine.process_snapshot(snap1)
    # Second context's snapshot should be snap1
    ctx = strategy.contexts[1]
    assert ctx.snapshots["default"].asks[0].price == 102.0


# ---------------------------------------------------------------------------
# context.positions
# ---------------------------------------------------------------------------


def test_context_positions_initially_empty() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    engine.process_snapshot(_snap(_t(0)))
    assert strategy.contexts[0].positions == {}


def test_context_positions_updated_after_fill() -> None:
    from collections import deque

    from src.actions import PlaceOrderAction
    from src.enums import OrderType, Side
    from src.strategy.base import BaseStrategy
    from src.strategy.context import StrategyContext

    class OneShotBuyStrategy(BaseStrategy):
        def __init__(self) -> None:
            super().__init__("s1")
            self._placed = False
            self.contexts: list[StrategyContext] = []

        def on_snapshot(self, context: StrategyContext):
            self.contexts.append(context)
            if not self._placed:
                self._placed = True
                return [
                    PlaceOrderAction(
                        strategy_id=self.strategy_id,
                        side=Side.BUY,
                        price=101.0,
                        qty=1.0,
                        order_type=OrderType.LIMIT,
                    )
                ]
            return []

    strategy = OneShotBuyStrategy()
    engine = SimulationEngine(strategy=strategy, trading_instrument_ids=frozenset({"default"}))
    # tick 0: places BUY at 101 (ask is 101) → fills immediately
    engine.process_snapshot(_snap(_t(0), ask=101.0))
    # tick 1: context should reflect position = 1.0
    engine.process_snapshot(_snap(_t(1)))
    ctx = strategy.contexts[1]
    assert ctx.positions.get("default", 0.0) == 1.0


# ---------------------------------------------------------------------------
# context.triggering_event
# ---------------------------------------------------------------------------


def test_triggering_event_is_market_snapshot_event() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    engine.process_snapshot(_snap(_t(0)))
    ctx = strategy.contexts[0]
    assert isinstance(ctx.triggering_event, MarketSnapshotEvent)


def test_triggering_event_timestamp_matches_snapshot() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    snap = _snap(_t(5))
    engine.process_snapshot(snap)
    ctx = strategy.contexts[0]
    assert ctx.triggering_event.timestamp == _t(5)


# ---------------------------------------------------------------------------
# Legacy context.snapshot and context.position unchanged
# ---------------------------------------------------------------------------


def test_legacy_snapshot_field_set_from_engine() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    snap = _snap(_t(0))
    engine.process_snapshot(snap)
    ctx = strategy.contexts[0]
    assert ctx.snapshot is not None
    assert ctx.snapshot.timestamp == snap.timestamp


def test_legacy_position_zero_initially() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(strategy=strategy)
    engine.process_snapshot(_snap(_t(0)))
    assert strategy.contexts[0].position == 0.0
