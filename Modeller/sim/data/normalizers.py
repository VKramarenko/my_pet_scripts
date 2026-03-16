from __future__ import annotations

from collections.abc import Iterable, Iterator

from sim.core.events import Event, MarketSnapshot, MarketTrade, TimerEvent


def _event_priority(event: Event) -> int:
    if isinstance(event, MarketSnapshot):
        return 0
    if isinstance(event, MarketTrade):
        return 1
    if isinstance(event, TimerEvent):
        return 2
    raise TypeError(f"Unknown event type: {type(event)}")


def merge_streams(
    snapshot_iter: Iterable[MarketSnapshot],
    trade_iter: Iterable[MarketTrade],
    timer_iter: Iterable[TimerEvent] | None = None,
) -> Iterator[Event]:
    enriched: list[tuple[float, int, int, Event]] = []
    seq = 0
    for event in snapshot_iter:
        enriched.append((event.ts, _event_priority(event), seq, event))
        seq += 1
    for event in trade_iter:
        enriched.append((event.ts, _event_priority(event), seq, event))
        seq += 1
    if timer_iter is not None:
        for event in timer_iter:
            enriched.append((event.ts, _event_priority(event), seq, event))
            seq += 1
    for _, _, _, event in sorted(enriched, key=lambda x: (x[0], x[1], x[2])):
        yield event

