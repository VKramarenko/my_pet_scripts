from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models import Trade


class CommissionModel(ABC):
    @abstractmethod
    def compute(self, trade: Trade) -> float:
        raise NotImplementedError


class NoCommission(CommissionModel):
    def compute(self, trade: Trade) -> float:
        _ = trade
        return 0.0


@dataclass(slots=True)
class FixedPerTradeCommission(CommissionModel):
    amount: float

    def compute(self, trade: Trade) -> float:
        _ = trade
        return self.amount


@dataclass(slots=True)
class BpsCommission(CommissionModel):
    bps: float

    def compute(self, trade: Trade) -> float:
        return trade.price * trade.qty * self.bps / 10000.0

