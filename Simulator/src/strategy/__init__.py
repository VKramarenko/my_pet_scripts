"""Strategy framework for the simulator.

Core contracts live in this package, while ready-to-run strategy examples
are grouped under ``src.strategy.examples``.
"""

from src.strategy.base import BaseStrategy
from src.strategy.examples import MovingAverageCrossStrategy, PassiveBuyOnceStrategy, RSIMeanReversionStrategy, compute_rsi

__all__ = [
    "BaseStrategy",
    "PassiveBuyOnceStrategy",
    "RSIMeanReversionStrategy",
    "MovingAverageCrossStrategy",
    "compute_rsi",
]

