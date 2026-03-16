from __future__ import annotations

from pathlib import Path

from run_backtest import run
from sim.core.events import MarketSnapshot
from sim.exchange.orders import Fill
from sim.market.orderbook_l2 import OrderBookL2
from sim.strategy.taker_bollinger import TakerBollingerStrategy


def _write_l2_csv(path: Path) -> None:
    path.write_text(
        "ts,bids,asks\n"
        '1.0,"[[100.0, 2.0]]","[[101.0, 2.0]]"\n'
        '2.0,"[[101.0, 2.0]]","[[102.0, 2.0]]"\n',
        encoding="utf-8",
    )


def test_run_snapshots_only_smoke(tmp_path: Path) -> None:
    l2_path = tmp_path / "l2.csv"
    _write_l2_csv(l2_path)

    metrics = run(str(l2_path), trades_path=None, loader="default")

    assert metrics.num_fills >= 0
    assert len(metrics.equity_curve) > 0


def test_taker_bollinger_breakout_then_revert() -> None:
    strategy = TakerBollingerStrategy(
        window=3,
        std_mult=1.0,
        order_qty=1.0,
        cooldown=0.0,
        max_position=1.0,
    )
    book = OrderBookL2(depth=5)

    for mid in [100.0, 100.0, 100.0]:
        snapshot = MarketSnapshot(ts=0.0, bids=[(mid - 0.5, 1.0)], asks=[(mid + 0.5, 1.0)])
        book.update_from_snapshot(snapshot)
        assert strategy.on_snapshot(0.0, book, []) == []

    snapshot_breakout = MarketSnapshot(ts=1.0, bids=[(103.5, 1.0)], asks=[(104.5, 1.0)])
    book.update_from_snapshot(snapshot_breakout)
    actions = strategy.on_snapshot(1.0, book, [])
    assert len(actions) == 1
    assert actions[0].type == "MARKET"
    assert actions[0].side == "BUY"
    strategy.on_fill(Fill(actions[0].order_id, 1.0, 104.5, actions[0].qty, 0.0, "TAKER"))

    snapshot_revert = MarketSnapshot(ts=2.0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    book.update_from_snapshot(snapshot_revert)
    reverse_actions = strategy.on_snapshot(2.0, book, [])
    assert len(reverse_actions) == 1
    assert reverse_actions[0].type == "MARKET"
    assert reverse_actions[0].side == "SELL"
    assert reverse_actions[0].qty == 2.0

    strategy.on_fill(Fill(reverse_actions[0].order_id, 2.0, 100.0, reverse_actions[0].qty, 0.0, "TAKER"))

    snapshot_revert_again = MarketSnapshot(ts=3.0, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])
    book.update_from_snapshot(snapshot_revert_again)
    reverse_back_actions = strategy.on_snapshot(3.0, book, [])
    assert len(reverse_back_actions) == 1
    assert reverse_back_actions[0].type == "MARKET"
    assert reverse_back_actions[0].side == "BUY"
    assert reverse_back_actions[0].qty == 2.0
