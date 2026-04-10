"""Tests for src/multi_feed.py merge utilities."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.events import BaseEvent, CustomEvent, MarketSnapshotEvent
from src.models import Level, Snapshot
from src.multi_feed import merge_event_feeds, merge_snapshot_feeds


def _snap(ts: datetime, *, instrument_id: str = "default", ask: float = 101.0, bid: float = 100.0) -> Snapshot:
    return Snapshot(
        timestamp=ts,
        asks=[Level(price=ask, qty=1.0)],
        bids=[Level(price=bid, qty=1.0)],
        instrument_id=instrument_id,
    )


T0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# merge_snapshot_feeds
# ---------------------------------------------------------------------------


def test_merge_snapshot_feeds_single_feed_preserves_order() -> None:
    snaps = [_snap(_t(i)) for i in range(3)]
    result = list(merge_snapshot_feeds({"A": iter(snaps)}))
    assert [e.timestamp for e in result] == [_t(0), _t(1), _t(2)]


def test_merge_snapshot_feeds_two_feeds_sorted_by_time() -> None:
    feed_a = [_snap(_t(0)), _snap(_t(2)), _snap(_t(4))]
    feed_b = [_snap(_t(1)), _snap(_t(3)), _snap(_t(5))]
    result = list(merge_snapshot_feeds({"A": iter(feed_a), "B": iter(feed_b)}))
    timestamps = [e.timestamp for e in result]
    assert timestamps == sorted(timestamps)
    assert len(result) == 6


def test_merge_snapshot_feeds_stamps_instrument_id() -> None:
    snap = _snap(_t(0), instrument_id="WRONG")
    result = list(merge_snapshot_feeds({"BTCUSDT": iter([snap])}))
    assert result[0].snapshot.instrument_id == "BTCUSDT"


def test_merge_snapshot_feeds_equal_timestamps_deterministic_order() -> None:
    ts = _t(0)
    feed_a = [_snap(ts, instrument_id="A")]
    feed_b = [_snap(ts, instrument_id="B")]
    feed_c = [_snap(ts, instrument_id="C")]
    result = list(merge_snapshot_feeds({"A": iter(feed_a), "B": iter(feed_b), "C": iter(feed_c)}))
    assert [e.snapshot.instrument_id for e in result] == ["A", "B", "C"]


def test_merge_snapshot_feeds_returns_market_snapshot_events() -> None:
    result = list(merge_snapshot_feeds({"X": iter([_snap(_t(0))])}))
    assert all(isinstance(e, MarketSnapshotEvent) for e in result)


def test_merge_snapshot_feeds_empty_input() -> None:
    assert list(merge_snapshot_feeds({})) == []


def test_merge_snapshot_feeds_one_empty_feed() -> None:
    feed_a = [_snap(_t(0))]
    result = list(merge_snapshot_feeds({"A": iter(feed_a), "B": iter([])}))
    assert len(result) == 1
    assert result[0].snapshot.instrument_id == "A"


# ---------------------------------------------------------------------------
# merge_event_feeds
# ---------------------------------------------------------------------------


def _snap_event(ts: datetime) -> MarketSnapshotEvent:
    return MarketSnapshotEvent(timestamp=ts, snapshot=_snap(ts))


def _custom_event(ts: datetime, name: str = "ping") -> CustomEvent:
    return CustomEvent(timestamp=ts, name=name)


def test_merge_event_feeds_single_feed_order() -> None:
    events = [_snap_event(_t(i)) for i in range(3)]
    result = list(merge_event_feeds(iter(events)))
    assert [e.timestamp for e in result] == [_t(0), _t(1), _t(2)]


def test_merge_event_feeds_two_feeds_sorted() -> None:
    feed_a = [_snap_event(_t(0)), _snap_event(_t(2))]
    feed_b = [_custom_event(_t(1)), _custom_event(_t(3))]
    result = list(merge_event_feeds(iter(feed_a), iter(feed_b)))
    timestamps = [e.timestamp for e in result]
    assert timestamps == sorted(timestamps)
    assert len(result) == 4


def test_merge_event_feeds_equal_timestamps_lower_feed_index_wins() -> None:
    ts = _t(0)
    e0 = _snap_event(ts)
    e1 = _custom_event(ts, "from_feed_1")
    result = list(merge_event_feeds(iter([e0]), iter([e1])))
    assert result[0] is e0
    assert result[1] is e1


def test_merge_event_feeds_empty() -> None:
    assert list(merge_event_feeds()) == []


def test_merge_event_feeds_mixed_types_preserved() -> None:
    feed = [_snap_event(_t(0)), _custom_event(_t(1))]
    result = list(merge_event_feeds(iter(feed)))
    assert isinstance(result[0], MarketSnapshotEvent)
    assert isinstance(result[1], CustomEvent)
