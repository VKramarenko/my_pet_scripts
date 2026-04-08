from __future__ import annotations

from src.actions import CancelOrderAction, ModifyOrderAction, PlaceOrderAction
from src.enums import ActionType, OrderType, Side


def test_place_order_action_is_created_correctly() -> None:
    action = PlaceOrderAction(
        strategy_id="s-1",
        side=Side.BUY,
        price=100.0,
        qty=2.0,
        order_type=OrderType.LIMIT,
        client_order_id="client-1",
    )

    assert action.action_type == ActionType.PLACE_ORDER
    assert action.client_order_id == "client-1"


def test_cancel_order_action_is_created_correctly() -> None:
    action = CancelOrderAction(strategy_id="s-1", order_id="o-1")
    assert action.action_type == ActionType.CANCEL_ORDER
    assert action.order_id == "o-1"


def test_modify_order_action_is_created_correctly() -> None:
    action = ModifyOrderAction(
        strategy_id="s-1",
        order_id="o-1",
        new_price=101.0,
        new_qty=3.0,
    )

    assert action.action_type == ActionType.MODIFY_ORDER
    assert action.new_price == 101.0
    assert action.new_qty == 3.0

