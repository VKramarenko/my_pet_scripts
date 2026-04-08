from __future__ import annotations

from datetime import UTC, datetime

from src.enums import LiquidityRole, Side
from src.events import OwnTradeEvent
from src.models import Trade
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext


class NoopStrategy(BaseStrategy):
    def on_snapshot(self, context: StrategyContext):
        return []


def make_event(side: Side, price: float, qty: float) -> OwnTradeEvent:
    trade = Trade(
        trade_id="t-1",
        order_id="o-1",
        strategy_id="s-1",
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        side=side,
        price=price,
        qty=qty,
        liquidity_role=LiquidityRole.TAKER,
    )
    return OwnTradeEvent(timestamp=trade.timestamp, trade=trade)


def test_on_trade_adds_trade_to_state() -> None:
    strategy = NoopStrategy("s-1")
    strategy.on_trade(make_event(Side.BUY, 100.0, 2.0))
    assert len(strategy.state.trades) == 1


def test_buy_trade_increases_position() -> None:
    strategy = NoopStrategy("s-1")
    strategy.on_trade(make_event(Side.BUY, 100.0, 2.0))
    assert strategy.state.position == 2.0


def test_sell_trade_decreases_position() -> None:
    strategy = NoopStrategy("s-1")
    strategy.on_trade(make_event(Side.SELL, 100.0, 2.0))
    assert strategy.state.position == -2.0


def test_buy_trade_decreases_cash() -> None:
    strategy = NoopStrategy("s-1")
    strategy.on_trade(make_event(Side.BUY, 100.0, 2.0))
    assert strategy.state.cash == -200.0


def test_sell_trade_increases_cash() -> None:
    strategy = NoopStrategy("s-1")
    strategy.on_trade(make_event(Side.SELL, 100.0, 2.0))
    assert strategy.state.cash == 200.0


def test_trade_count_increases() -> None:
    strategy = NoopStrategy("s-1")
    strategy.on_trade(make_event(Side.BUY, 100.0, 2.0))
    assert strategy.state.metrics.trade_count == 1


def test_turnover_increases_by_notional() -> None:
    strategy = NoopStrategy("s-1")
    strategy.on_trade(make_event(Side.BUY, 100.0, 2.0))
    assert strategy.state.metrics.turnover == 200.0

