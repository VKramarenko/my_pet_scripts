from dataclasses import dataclass

from sim.exchange.orders import Liquidity


@dataclass
class FeeModel:
    maker_rate: float = 0.0
    taker_rate: float = 0.0

    def compute(self, price: float, qty: float, liquidity: Liquidity) -> float:
        notional = price * qty
        if liquidity == "MAKER":
            return notional * self.maker_rate
        return notional * self.taker_rate

