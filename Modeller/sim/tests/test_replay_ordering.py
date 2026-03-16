from sim.core.events import MarketSnapshot, MarketTrade, TimerEvent
from sim.data.normalizers import merge_streams


def test_replay_ordering_snapshot_then_trade_then_timer() -> None:
    snapshots = [MarketSnapshot(ts=1.0, bids=[(100, 1)], asks=[(101, 1)])]
    trades = [MarketTrade(ts=1.0, price=101, size=1, side="buyer_initiated")]
    timers = [TimerEvent(ts=1.0)]

    events = list(merge_streams(snapshots, trades, timers))
    assert isinstance(events[0], MarketSnapshot)
    assert isinstance(events[1], MarketTrade)
    assert isinstance(events[2], TimerEvent)

