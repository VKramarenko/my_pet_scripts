from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulationConfig:
    """Stage 1 simulator configuration."""

    order_mode: str = "LIMIT_WITH_MEMORY"
    resting_fill_model: str = "book_match"
    allow_partial_fills_for_limit: bool = True
    process_resting_orders_before_strategy: bool = True
    modify_as_cancel_replace: bool = True
    market_as_exogenous: bool = True
    deterministic_order_priority: str = "created_at"
    mark_to_mid_missing_policy: str = "keep_last"
