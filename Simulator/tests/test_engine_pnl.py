from __future__ import annotations

from datetime import UTC, datetime

from src.accounting import apply_trade_to_strategy_state, mark_strategy_to_market
from src.enums import LiquidityRole, Side
from src.models import Level, Snapshot, Trade
from src.strategy.state import StrategyState


def make_trade(side: Side, price: float, qty: float, commission: float = 0.0) -> Trade:
    return Trade(
        trade_id="t-1",
        order_id="o-1",
        strategy_id="s-1",
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        side=side,
        price=price,
        qty=qty,
        liquidity_role=LiquidityRole.TAKER,
        commission=commission,
    )


def test_buy_trade_increases_position_and_decreases_cash() -> None:
    state = StrategyState()
    apply_trade_to_strategy_state(state, make_trade(Side.BUY, 100.0, 2.0))
    assert state.position == 2.0
    assert state.cash == -200.0


def test_sell_trade_decreases_position_and_increases_cash() -> None:
    state = StrategyState(position=2.0, avg_entry_price=100.0)
    apply_trade_to_strategy_state(state, make_trade(Side.SELL, 110.0, 1.0))
    assert state.position == 1.0
    assert state.cash == 110.0


def test_average_cost_updates_when_increasing_position() -> None:
    state = StrategyState()
    apply_trade_to_strategy_state(state, make_trade(Side.BUY, 100.0, 1.0))
    apply_trade_to_strategy_state(state, make_trade(Side.BUY, 110.0, 1.0))
    assert state.avg_entry_price == 105.0


def test_realized_pnl_computed_for_partial_long_close() -> None:
    state = StrategyState(position=2.0, avg_entry_price=100.0)
    apply_trade_to_strategy_state(state, make_trade(Side.SELL, 110.0, 1.0))
    assert state.realized_pnl == 10.0


def test_realized_pnl_computed_for_short_close() -> None:
    state = StrategyState(position=-2.0, avg_entry_price=100.0)
    apply_trade_to_strategy_state(state, make_trade(Side.BUY, 90.0, 1.0))
    assert state.realized_pnl == 10.0


def test_flip_through_zero_is_handled() -> None:
    state = StrategyState(position=1.0, avg_entry_price=100.0)
    apply_trade_to_strategy_state(state, make_trade(Side.SELL, 110.0, 2.0))
    assert state.position == -1.0
    assert state.avg_entry_price == 110.0


def test_unrealized_pnl_uses_mark_to_mid() -> None:
    state = StrategyState(position=2.0, avg_entry_price=100.0, cash=-200.0)
    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 7, 12, 1, tzinfo=UTC),
        asks=[Level(111.0, 1.0)],
        bids=[Level(109.0, 1.0)],
    )
    mark_strategy_to_market(state, snapshot)
    assert state.unrealized_pnl == 20.0


def test_equity_equals_cash_plus_position_times_mid() -> None:
    state = StrategyState(position=2.0, avg_entry_price=100.0, cash=-200.0)
    snapshot = Snapshot(
        timestamp=datetime(2026, 4, 7, 12, 1, tzinfo=UTC),
        asks=[Level(111.0, 1.0)],
        bids=[Level(109.0, 1.0)],
    )
    mark_strategy_to_market(state, snapshot)
    assert state.equity == 20.0

