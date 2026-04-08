from __future__ import annotations

from datetime import UTC, datetime

from src.commission_models import BpsCommission, FixedPerTradeCommission, NoCommission
from src.enums import LiquidityRole, Side
from src.models import Trade


def make_trade() -> Trade:
    return Trade(
        trade_id="t-1",
        order_id="o-1",
        strategy_id="s-1",
        timestamp=datetime(2026, 4, 7, 12, 0, tzinfo=UTC),
        side=Side.BUY,
        price=100.0,
        qty=2.0,
        liquidity_role=LiquidityRole.TAKER,
    )


def test_no_commission_returns_zero() -> None:
    assert NoCommission().compute(make_trade()) == 0.0


def test_fixed_per_trade_commission_returns_fixed_amount() -> None:
    assert FixedPerTradeCommission(1.5).compute(make_trade()) == 1.5


def test_bps_commission_uses_notional() -> None:
    assert BpsCommission(10.0).compute(make_trade()) == 0.2

