from __future__ import annotations

from src.actions import PlaceOrderAction
from src.enums import Side
from src.strategy.examples.technical import MovingAverageCrossStrategy, RSIMeanReversionStrategy, compute_rsi
from tests.helpers.contexts import make_empty_strategy_context, make_strategy_context
from tests.helpers.snapshots import make_snapshot


def test_compute_rsi_returns_none_without_enough_prices() -> None:
    assert compute_rsi([100.0, 101.0], period=5) is None


def test_rsi_strategy_buys_when_oversold(base_time) -> None:
    strategy = RSIMeanReversionStrategy(
        "s-1",
        {"rsi_period": 3, "oversold": 35.0, "overbought": 70.0, "qty": 1.0},
    )
    snapshots = [
        make_snapshot(base_time, best_ask=101.0, best_bid=100.0),
        make_snapshot(base_time.replace(second=1), best_ask=100.0, best_bid=99.0),
        make_snapshot(base_time.replace(second=2), best_ask=99.0, best_bid=98.0),
        make_snapshot(base_time.replace(second=3), best_ask=98.0, best_bid=97.0),
    ]

    actions = []
    for snapshot in snapshots:
        actions = strategy.on_snapshot(make_empty_strategy_context(snapshot.timestamp, snapshot))

    assert len(actions) == 1
    assert isinstance(actions[0], PlaceOrderAction)
    assert actions[0].side == Side.BUY


def test_rsi_strategy_sells_when_overbought_and_has_position(base_time) -> None:
    strategy = RSIMeanReversionStrategy(
        "s-1",
        {"rsi_period": 3, "oversold": 30.0, "overbought": 65.0, "qty": 1.0},
    )
    strategy.state.position = 1.0
    snapshots = [
        make_snapshot(base_time, best_ask=100.0, best_bid=99.0),
        make_snapshot(base_time.replace(second=1), best_ask=101.0, best_bid=100.0),
        make_snapshot(base_time.replace(second=2), best_ask=102.0, best_bid=101.0),
        make_snapshot(base_time.replace(second=3), best_ask=103.0, best_bid=102.0),
    ]

    actions = []
    for snapshot in snapshots:
        actions = strategy.on_snapshot(make_strategy_context(snapshot.timestamp, snapshot, position=1.0))

    assert len(actions) == 1
    assert actions[0].side == Side.SELL


def test_moving_average_cross_buys_on_bullish_cross(base_time) -> None:
    strategy = MovingAverageCrossStrategy("s-1", {"short_window": 2, "long_window": 3, "qty": 1.0})
    snapshots = [
        make_snapshot(base_time, best_ask=100.0, best_bid=99.0),
        make_snapshot(base_time.replace(second=1), best_ask=100.0, best_bid=99.0),
        make_snapshot(base_time.replace(second=2), best_ask=103.0, best_bid=102.0),
    ]
    actions = []
    for snapshot in snapshots:
        actions = strategy.on_snapshot(make_empty_strategy_context(snapshot.timestamp, snapshot))
    assert len(actions) == 1
    assert actions[0].side == Side.BUY
