from __future__ import annotations

from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    FOK = "FOK"
    LIMIT = "LIMIT"


class OrderStatus(StrEnum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class LiquidityRole(StrEnum):
    TAKER = "TAKER"
    MAKER = "MAKER"
    UNKNOWN = "UNKNOWN"


class ActionType(StrEnum):
    PLACE_ORDER = "PLACE_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"
    MODIFY_ORDER = "MODIFY_ORDER"


class EventType(StrEnum):
    MARKET_SNAPSHOT = "MARKET_SNAPSHOT"
    ORDER_UPDATE = "ORDER_UPDATE"
    OWN_TRADE = "OWN_TRADE"

