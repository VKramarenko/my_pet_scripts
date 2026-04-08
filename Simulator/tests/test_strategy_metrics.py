from __future__ import annotations

from datetime import UTC, datetime

from src.enums import LiquidityRole, Side
from src.models import Trade
from src.strategy.metrics import StrategyMetrics, update_metrics_after_trade, update_metrics_after_valuation


def make_trade() -> Trade:
    return Trade(
        trade_id="t-1",
        order_id="o-1",
        strategy_id="s-1",
        timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
        side=Side.BUY,
        price=100.0,
        qty=2.0,
        liquidity_role=LiquidityRole.TAKER,
    )


def test_trade_count_increases_after_trade() -> None:
    metrics = StrategyMetrics()
    update_metrics_after_trade(metrics, make_trade())
    assert metrics.trade_count == 1


def test_fill_count_reflected_in_fill_ratio() -> None:
    metrics = StrategyMetrics(submitted_orders=2, filled_orders=1)
    assert metrics.fill_ratio == 0.5


def test_cancel_count_reflected_in_cancel_ratio() -> None:
    metrics = StrategyMetrics(submitted_orders=4, canceled_orders=1)
    assert metrics.cancel_ratio == 0.25


def test_turnover_grows_by_notional_sum() -> None:
    metrics = StrategyMetrics()
    update_metrics_after_trade(metrics, make_trade())
    assert metrics.turnover == 200.0


def test_max_drawdown_is_derived_from_equity_curve() -> None:
    metrics = StrategyMetrics()
    update_metrics_after_valuation(metrics, datetime(2024, 1, 1, 10, 0, tzinfo=UTC), 100.0, 0.0)
    update_metrics_after_valuation(metrics, datetime(2024, 1, 1, 10, 1, tzinfo=UTC), 120.0, 0.0)
    update_metrics_after_valuation(metrics, datetime(2024, 1, 1, 10, 2, tzinfo=UTC), 90.0, 0.0)
    assert metrics.max_drawdown == 0.25


def test_sharpe_equity_returns_is_computed_from_mock_series() -> None:
    metrics = StrategyMetrics()
    metrics.equity_returns = [0.01, 0.02, -0.01]
    assert metrics.sharpe_equity_returns is not None


def test_sharpe_pnl_increments_is_computed_from_mock_series() -> None:
    metrics = StrategyMetrics()
    metrics.pnl_increments = [1.0, 2.0, -1.0]
    assert metrics.sharpe_pnl_increments is not None


def test_empty_metrics_do_not_break_derived_properties() -> None:
    metrics = StrategyMetrics()
    assert metrics.max_drawdown is None
    assert metrics.sharpe_equity_returns is None
    assert metrics.sharpe_pnl_increments is None

