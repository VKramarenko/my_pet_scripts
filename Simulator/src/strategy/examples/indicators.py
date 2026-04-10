from __future__ import annotations


def compute_rsi(prices: list[float], period: int) -> float | None:
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(prices) < period + 1:
        return None

    window = prices[-(period + 1) :]
    gains = 0.0
    losses = 0.0
    for previous, current in zip(window, window[1:]):
        change = current - previous
        if change > 0:
            gains += change
        elif change < 0:
            losses += abs(change)

    average_gain = gains / period
    average_loss = losses / period
    if average_loss == 0:
        return 100.0
    rs = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + rs))
