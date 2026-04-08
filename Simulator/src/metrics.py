from __future__ import annotations

from math import sqrt


def compute_fill_ratio(filled_orders: int, submitted_orders: int) -> float | None:
    if submitted_orders <= 0:
        return None
    return filled_orders / submitted_orders


def compute_cancel_ratio(canceled_orders: int, submitted_orders: int) -> float | None:
    if submitted_orders <= 0:
        return None
    return canceled_orders / submitted_orders


def compute_max_drawdown(equity_values: list[float]) -> float | None:
    if not equity_values:
        return None
    peak = equity_values[0]
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak != 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    return max_drawdown


def compute_sharpe_ratio(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0:
        return None
    return mean / (variance ** 0.5) * sqrt(len(values))

