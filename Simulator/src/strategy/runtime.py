from __future__ import annotations

from dataclasses import dataclass

from src.strategy.base import BaseStrategy
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy
from src.strategy.examples.technical import MovingAverageCrossStrategy, RSIMeanReversionStrategy


@dataclass(frozen=True, slots=True)
class StrategySpec:
    name: str
    description: str
    builder: type[BaseStrategy]


STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    "passive_buy_once": StrategySpec(
        name="passive_buy_once",
        description="Places one passive buy order once.",
        builder=PassiveBuyOnceStrategy,
    ),
    "rsi_mean_reversion": StrategySpec(
        name="rsi_mean_reversion",
        description="Buys on oversold RSI and exits on overbought RSI.",
        builder=RSIMeanReversionStrategy,
    ),
    "moving_average_cross": StrategySpec(
        name="moving_average_cross",
        description="Trades on short/long moving average crossovers.",
        builder=MovingAverageCrossStrategy,
    ),
}


def build_strategy(name: str, strategy_id: str, config: dict) -> BaseStrategy:
    try:
        spec = STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc
    return spec.builder(strategy_id=strategy_id, config=config)


def available_strategy_names() -> list[str]:
    return sorted(STRATEGY_REGISTRY)
