from dataclasses import dataclass
from typing import Literal


@dataclass
class ExecutionConfig:
    mode: Literal["optimistic", "pessimistic"] = "pessimistic"
    fill_price: Literal["order_price", "best_price"] = "best_price"
    require_trade_for_fill: bool = False
    allow_trade_through: bool = False
    max_fill_per_event: float | None = None
    market_slippage_bps: float = 0.0

