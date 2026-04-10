from __future__ import annotations

from dataclasses import dataclass, field

from src.enums import ActionType, OrderType, Side


@dataclass(slots=True)
class BaseAction:
    """Base action requested by a strategy."""

    action_type: ActionType
    strategy_id: str


@dataclass(slots=True)
class PlaceOrderAction(BaseAction):
    side: Side
    price: float
    qty: float
    order_type: OrderType
    instrument_id: str = "default"
    client_order_id: str | None = None
    action_type: ActionType = field(init=False, default=ActionType.PLACE_ORDER)

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("price must be > 0")
        if self.qty <= 0:
            raise ValueError("qty must be > 0")


@dataclass(slots=True)
class CancelOrderAction(BaseAction):
    order_id: str
    action_type: ActionType = field(init=False, default=ActionType.CANCEL_ORDER)


@dataclass(slots=True)
class ModifyOrderAction(BaseAction):
    order_id: str
    new_price: float | None = None
    new_qty: float | None = None
    action_type: ActionType = field(init=False, default=ActionType.MODIFY_ORDER)

    def __post_init__(self) -> None:
        if self.new_price is not None and self.new_price <= 0:
            raise ValueError("new_price must be > 0")
        if self.new_qty is not None and self.new_qty <= 0:
            raise ValueError("new_qty must be > 0")

