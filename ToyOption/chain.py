"""Tradable option chain computed from a pricing model and spread model."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TradableChain:
    """Ряд торгуемых опционов, вычисленных из модели ценообразования.

    Call и Put имеют независимые наборы страйков (разные диапазоны).
    Для каждого страйка хранятся:
    - fair-цены (call_fair, put_fair) — выход модели ценообразования
    - *_raw bid/ask — прямой выход spread-модели (до коррекции паритета)
    - call_bid/ask, put_bid/ask — итоговые котировки (изначально == raw;
      в будущем будут скорректированы через put-call parity)
    """

    # Independent strike grids for calls and puts
    call_strikes: np.ndarray   # arange(call_start, call_end+step/2, step)
    put_strikes: np.ndarray    # arange(put_start,  put_end+step/2,  step)

    # Fair (model) prices
    call_fair: np.ndarray
    put_fair: np.ndarray

    # Raw bid/ask — прямой выход spread-модели, промежуточное состояние
    call_bid_raw: np.ndarray
    call_ask_raw: np.ndarray
    put_bid_raw: np.ndarray
    put_ask_raw: np.ndarray

    # Итоговые bid/ask (изначально = raw; будут обновлены при коррекции паритета)
    call_bid: np.ndarray
    call_ask: np.ndarray
    put_bid: np.ndarray
    put_ask: np.ndarray
