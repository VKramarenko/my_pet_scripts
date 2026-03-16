from __future__ import annotations

from dataclasses import dataclass, field

from sim.core.events import MarketSnapshot
from sim.market.types import BookSide


@dataclass
class OrderBookL2:
    depth: int = 10
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    def update_from_snapshot(self, snapshot: MarketSnapshot) -> None:
        bid_levels = sorted(snapshot.bids, key=lambda x: x[0], reverse=True)[: self.depth]
        ask_levels = sorted(snapshot.asks, key=lambda x: x[0])[: self.depth]
        self.bids = {float(price): float(size) for price, size in bid_levels}
        self.asks = {float(price): float(size) for price, size in ask_levels}

    def best_bid(self) -> float | None:
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks.keys()) if self.asks else None

    def mid(self) -> float | None:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def level(self, side: BookSide, price: float) -> float:
        book = self.bids if side == "BUY" else self.asks
        return float(book.get(price, 0.0))

    def top_n(self, side: BookSide, n: int | None = None) -> list[tuple[float, float]]:
        n = self.depth if n is None else n
        if side == "BUY":
            return sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:n]
        return sorted(self.asks.items(), key=lambda x: x[0])[:n]

