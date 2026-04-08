from __future__ import annotations

from src.metrics import compute_cancel_ratio, compute_fill_ratio, compute_max_drawdown, compute_sharpe_ratio


def test_fill_ratio_is_computed() -> None:
    assert compute_fill_ratio(2, 4) == 0.5


def test_cancel_ratio_is_computed() -> None:
    assert compute_cancel_ratio(1, 4) == 0.25


def test_max_drawdown_is_computed() -> None:
    assert compute_max_drawdown([100.0, 120.0, 90.0]) == 0.25


def test_sharpe_equity_returns_is_computed() -> None:
    value = compute_sharpe_ratio([0.01, 0.02, -0.01])
    assert value is not None


def test_sharpe_returns_none_on_zero_dispersion() -> None:
    assert compute_sharpe_ratio([0.01, 0.01, 0.01]) is None

