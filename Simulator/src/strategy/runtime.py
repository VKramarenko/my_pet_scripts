from __future__ import annotations

from dataclasses import dataclass

from src.strategy.base import BaseStrategy
from src.strategy.examples.buy_once import PassiveBuyOnceStrategy
from src.strategy.examples.moving_average_cross import MovingAverageCrossStrategy
from src.strategy.examples.rsi_limit_order_template import RSILimitOrderTemplateStrategy
from src.strategy.examples.rsi_limit_order_timeout import RSILimitOrderTimeoutStrategy
from src.strategy.examples.rsi_dual_book_timeout import RSIDualBookTimeoutStrategy
from src.strategy.examples.rsi_mean_reversion import RSIMeanReversionStrategy


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
    "rsi_limit_order_timeout": StrategySpec(
        name="rsi_limit_order_timeout",
        description="Places passive RSI limit orders and cancels stale orders after a TTL.",
        builder=RSILimitOrderTimeoutStrategy,
    ),
    "rsi_limit_order_template": StrategySpec(
        name="rsi_limit_order_template",
        description="Educational RSI strategy showing passive limit order lifecycle handling.",
        builder=RSILimitOrderTemplateStrategy,
    ),
    "moving_average_cross": StrategySpec(
        name="moving_average_cross",
        description="Trades on short/long moving average crossovers.",
        builder=MovingAverageCrossStrategy,
    ),
    "rsi_dual_book_timeout": StrategySpec(
        name="rsi_dual_book_timeout",
        description="RSI limit-order strategy that trades independently in multiple books with stale-order cancellation.",
        builder=RSIDualBookTimeoutStrategy,
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
