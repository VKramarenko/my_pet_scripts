from __future__ import annotations

from dataclasses import dataclass, field

from sim.exchange.orders import Fill, OrderSide


@dataclass
class MetricsCollector:
    equity_curve: list[tuple[float, float]] = field(default_factory=list)
    num_fills: int = 0
    fill_events: list[dict[str, float | str]] = field(default_factory=list)
    tob_curve: list[tuple[float, float]] = field(default_factory=list)
    market_trades: list[dict[str, float | str]] = field(default_factory=list)

    def on_equity(self, ts: float, equity: float) -> None:
        self.equity_curve.append((ts, equity))

    def on_fill(self, fill: Fill, side: OrderSide) -> None:
        self.num_fills += 1
        self.fill_events.append(
            {
                "ts": fill.ts,
                "order_id": fill.order_id,
                "side": side,
                "price": fill.price,
                "qty": fill.qty,
                "fee": fill.fee,
                "liquidity": fill.liquidity,
                "notional": fill.price * fill.qty,
            }
        )

    def on_tob(self, ts: float, mid: float) -> None:
        self.tob_curve.append((ts, mid))

    def on_market_trade(self, ts: float, price: float, size: float, side: str) -> None:
        self.market_trades.append({"ts": ts, "price": price, "size": size, "side": side})

