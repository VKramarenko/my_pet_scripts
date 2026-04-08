from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.metrics import compute_cancel_ratio, compute_fill_ratio, compute_max_drawdown, compute_sharpe_ratio
from src.models import Trade


@dataclass(slots=True)
class StrategyMetrics:
    """Lightweight container for strategy metrics and time series."""

    trade_count: int = 0
    turnover: float = 0.0
    equity_curve: list[tuple[datetime, float | None]] = field(default_factory=list)
    pnl_increments: list[float] = field(default_factory=list)
    equity_returns: list[float] = field(default_factory=list)
    submitted_orders: int = 0
    filled_orders: int = 0
    canceled_orders: int = 0

    def record_equity_point(self, timestamp: datetime, equity: float | None) -> None:
        self.equity_curve.append((timestamp, equity))

    @property
    def fill_ratio(self) -> float | None:
        return compute_fill_ratio(self.filled_orders, self.submitted_orders)

    @property
    def cancel_ratio(self) -> float | None:
        return compute_cancel_ratio(self.canceled_orders, self.submitted_orders)

    @property
    def max_drawdown(self) -> float | None:
        values = [value for _, value in self.equity_curve if value is not None]
        return compute_max_drawdown(values)

    @property
    def sharpe_equity_returns(self) -> float | None:
        return compute_sharpe_ratio(self.equity_returns)

    @property
    def sharpe_pnl_increments(self) -> float | None:
        return compute_sharpe_ratio(self.pnl_increments)


def update_metrics_after_trade(metrics: StrategyMetrics, trade: Trade) -> None:
    metrics.trade_count += 1
    metrics.turnover += abs(trade.notional if trade.notional is not None else trade.price * trade.qty)


def update_metrics_after_valuation(
    metrics: StrategyMetrics,
    timestamp: datetime,
    equity: float | None,
    realized_pnl: float,
) -> None:
    previous_equity = metrics.equity_curve[-1][1] if metrics.equity_curve else None
    previous_realized = sum(metrics.pnl_increments) if metrics.pnl_increments else 0.0
    metrics.record_equity_point(timestamp, equity)
    metrics.pnl_increments.append(realized_pnl - previous_realized)
    if previous_equity is not None and previous_equity != 0 and equity is not None:
        metrics.equity_returns.append((equity - previous_equity) / previous_equity)
