from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.enums import Side


class SlippageModel(ABC):
    @abstractmethod
    def apply(self, side: Side, raw_price: float, qty: float) -> float:
        raise NotImplementedError


class NoSlippage(SlippageModel):
    def apply(self, side: Side, raw_price: float, qty: float) -> float:
        _ = side, qty
        return raw_price


@dataclass(slots=True)
class FixedBpsSlippage(SlippageModel):
    bps: float

    def apply(self, side: Side, raw_price: float, qty: float) -> float:
        _ = qty
        shift = self.bps / 10000.0
        if side == Side.BUY:
            return raw_price * (1.0 + shift)
        return raw_price * (1.0 - shift)

