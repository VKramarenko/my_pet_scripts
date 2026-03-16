from __future__ import annotations

from dataclasses import dataclass, field

from sim.exchange.orders import Fill, OrderSide


@dataclass
class Portfolio:
    cash: float = 0.0
    position: float = 0.0
    paid_fees: float = 0.0
    fills: list[Fill] = field(default_factory=list)

    def apply_fill(self, fill: Fill, side: OrderSide) -> None:
        if side == "BUY":
            self.position += fill.qty
            self.cash -= fill.price * fill.qty
        else:
            self.position -= fill.qty
            self.cash += fill.price * fill.qty
        self.cash -= fill.fee
        self.paid_fees += fill.fee
        self.fills.append(fill)

    def equity(self, mark_price: float | None) -> float | None:
        if mark_price is None:
            return None
        return self.cash + self.position * mark_price

