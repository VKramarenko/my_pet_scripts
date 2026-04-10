"""Utilities for merging multiple snapshot/event feeds into one ordered stream."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import replace

from src.events import BaseEvent, MarketSnapshotEvent
from src.models import Snapshot


def merge_snapshot_feeds(
    feeds: dict[str, Iterable[Snapshot]],
) -> Iterator[MarketSnapshotEvent]:
    """
    Merge multiple per-instrument snapshot iterables into one ordered stream.

    Args:
        feeds: mapping of {instrument_id: iterable_of_snapshots}.

    Yields:
        MarketSnapshotEvent instances sorted by timestamp.
        At equal timestamps the order is deterministic: lexicographic by instrument_id.
        Each snapshot gets its instrument_id set from the dict key.
    """
    # heap entries: (timestamp, instrument_id, MarketSnapshotEvent)
    # instrument_id is the tie-breaker for determinism
    heap: list[tuple] = []
    iterators: dict[str, Iterator[Snapshot]] = {
        iid: iter(feed) for iid, feed in feeds.items()
    }

    def _push(iid: str) -> None:
        snap = next(iterators[iid], None)
        if snap is None:
            return
        # Stamp the instrument_id onto the snapshot
        if snap.instrument_id != iid:
            snap = replace(snap, instrument_id=iid)
        event = MarketSnapshotEvent(timestamp=snap.timestamp, snapshot=snap)
        heapq.heappush(heap, (snap.timestamp, iid, event))

    for iid in iterators:
        _push(iid)

    while heap:
        _, iid, event = heapq.heappop(heap)
        yield event
        _push(iid)


def merge_event_feeds(
    *feeds: Iterable[BaseEvent],
) -> Iterator[BaseEvent]:
    """
    Merge multiple BaseEvent iterables into one ordered stream.

    At equal timestamps the original feed index is used as a tie-breaker
    (lower index wins) to ensure determinism.

    Yields:
        BaseEvent instances sorted by timestamp.
    """
    heap: list[tuple] = []
    iterators = [iter(feed) for feed in feeds]

    def _push(feed_idx: int) -> None:
        event = next(iterators[feed_idx], None)
        if event is None:
            return
        # tie-breaker: (timestamp, feed_index, event)
        heapq.heappush(heap, (event.timestamp, feed_idx, event))

    for idx in range(len(iterators)):
        _push(idx)

    while heap:
        _, feed_idx, event = heapq.heappop(heap)
        yield event
        _push(feed_idx)
