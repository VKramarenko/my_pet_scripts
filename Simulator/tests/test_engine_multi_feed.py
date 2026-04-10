"""Tests for SimulationEngine multi-book behaviour."""
from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta

import pytest

from src.actions import PlaceOrderAction
from src.engine import SimulationEngine
from src.enums import OrderType, Side
from src.events import MarketSnapshotEvent, OwnTradeEvent
from src.models import Level, Snapshot
from src.multi_feed import merge_snapshot_feeds
from tests.helpers.strategies import NoOpStrategy, RecordingStrategy


T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def _snap(ts: datetime, *, instrument_id: str = "default", ask: float = 101.0, bid: float = 100.0) -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(price=ask, qty=10.0)],
        bids=[Level(price=bid, qty=10.0)],
        instrument_id=instrument_id,
    )


# ---------------------------------------------------------------------------
# trading_instrument_ids — basics
# ---------------------------------------------------------------------------


def test_default_trading_instrument_ids() -> None:
    engine = SimulationEngine()
    assert "default" in engine.trading_instrument_ids


def test_process_snapshot_updates_current_snapshots() -> None:
    engine = SimulationEngine()
    snap_btc = _snap(_t(0), instrument_id="BTC")
    snap_eth = _snap(_t(1), instrument_id="ETH")
    engine.process_snapshot(snap_btc)
    engine.process_snapshot(snap_eth)
    assert "BTC" in engine.state.current_snapshots
    assert "ETH" in engine.state.current_snapshots


# ---------------------------------------------------------------------------
# on_snapshot called for ALL instruments (trading AND info)
# ---------------------------------------------------------------------------


def test_on_snapshot_called_for_info_instrument() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(
        strategy=strategy,
        trading_instrument_ids=frozenset({"BTC"}),
    )
    info_snap = _snap(_t(0), instrument_id="INFO")
    engine.process_snapshot(info_snap)
    assert len(strategy.contexts) == 1
    assert strategy.contexts[0].snapshot.instrument_id == "INFO"


def test_on_snapshot_called_for_trading_instrument() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(
        strategy=strategy,
        trading_instrument_ids=frozenset({"BTC"}),
    )
    btc_snap = _snap(_t(0), instrument_id="BTC")
    engine.process_snapshot(btc_snap)
    assert len(strategy.contexts) == 1


# ---------------------------------------------------------------------------
# Resting order matching only for trading instruments
# ---------------------------------------------------------------------------


def test_resting_order_matched_only_for_trading_instrument() -> None:
    """Place a resting BUY limit at ask price; only matches when instrument is trading."""
    from tests.helpers.strategies import ActionReturningStrategy

    snap_trading = _snap(_t(0), instrument_id="BTC", ask=100.0, bid=99.0)

    actions_step0 = [
        PlaceOrderAction(
            strategy_id="s1",
            side=Side.BUY,
            price=100.0,
            qty=1.0,
            order_type=OrderType.LIMIT,
            instrument_id="BTC",
        )
    ]
    strategy = ActionReturningStrategy("s1", [actions_step0])
    engine = SimulationEngine(
        strategy=strategy,
        trading_instrument_ids=frozenset({"BTC"}),
    )
    engine.process_snapshot(snap_trading)
    # Order placed and should fill immediately against snap_trading (ask=100.0, bid=99.0)
    trades = engine.state.trades
    assert len(trades) == 1
    assert trades[0].instrument_id == "BTC"


def test_order_for_info_instrument_ignored_with_warning() -> None:
    from tests.helpers.strategies import ActionReturningStrategy

    snap_info = _snap(_t(0), instrument_id="INFO", ask=101.0, bid=100.0)
    actions_step0 = [
        PlaceOrderAction(
            strategy_id="s1",
            side=Side.BUY,
            price=101.0,
            qty=1.0,
            order_type=OrderType.LIMIT,
            instrument_id="INFO",
        )
    ]
    strategy = ActionReturningStrategy("s1", [actions_step0])
    engine = SimulationEngine(
        strategy=strategy,
        trading_instrument_ids=frozenset({"BTC"}),
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        engine.process_snapshot(snap_info)
    assert len(engine.state.trades) == 0
    assert any("non-trading" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# run_events — merged stream
# ---------------------------------------------------------------------------


def test_run_events_processes_two_books() -> None:
    strategy = RecordingStrategy("s1")
    engine = SimulationEngine(
        strategy=strategy,
        trading_instrument_ids=frozenset({"BTC", "ETH"}),
    )
    btc_snaps = [_snap(_t(i), instrument_id="BTC") for i in range(3)]
    eth_snaps = [_snap(_t(i) + timedelta(milliseconds=500), instrument_id="ETH") for i in range(3)]
    events = merge_snapshot_feeds({"BTC": iter(btc_snaps), "ETH": iter(eth_snaps)})
    engine.run_events(events)
    assert len(strategy.contexts) == 6


def test_run_events_returns_simulation_state() -> None:
    engine = SimulationEngine(trading_instrument_ids=frozenset({"BTC"}))
    snaps = [_snap(_t(i), instrument_id="BTC") for i in range(2)]
    events = (MarketSnapshotEvent(timestamp=s.timestamp, snapshot=s) for s in snaps)
    state = engine.run_events(events)
    assert state is engine.state


# ---------------------------------------------------------------------------
# Per-instrument position tracking
# ---------------------------------------------------------------------------


def test_per_instrument_positions_tracked_independently() -> None:
    from tests.helpers.strategies import ActionReturningStrategy

    btc_snap = _snap(_t(0), instrument_id="BTC", ask=100.0, bid=99.0)
    eth_snap = _snap(_t(1), instrument_id="ETH", ask=200.0, bid=199.0)

    actions_btc = [
        PlaceOrderAction(
            strategy_id="s1",
            side=Side.BUY,
            price=100.0,
            qty=1.0,
            order_type=OrderType.LIMIT,
            instrument_id="BTC",
        )
    ]
    actions_eth = [
        PlaceOrderAction(
            strategy_id="s1",
            side=Side.SELL,
            price=199.0,
            qty=2.0,
            order_type=OrderType.LIMIT,
            instrument_id="ETH",
        )
    ]
    strategy = ActionReturningStrategy("s1", [actions_btc, actions_eth])
    engine = SimulationEngine(
        strategy=strategy,
        trading_instrument_ids=frozenset({"BTC", "ETH"}),
    )
    engine.process_snapshot(btc_snap)
    engine.process_snapshot(eth_snap)

    positions = strategy.state.positions
    assert positions.get("BTC", 0.0) == pytest.approx(1.0)
    assert positions.get("ETH", 0.0) == pytest.approx(-2.0)
