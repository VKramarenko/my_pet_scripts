from __future__ import annotations

from src.enums import Side
from src.slippage_models import FixedBpsSlippage, NoSlippage


def test_no_slippage_keeps_price() -> None:
    assert NoSlippage().apply(Side.BUY, 100.0, 1.0) == 100.0


def test_fixed_bps_slippage_worsens_buy_price_up() -> None:
    assert FixedBpsSlippage(10.0).apply(Side.BUY, 100.0, 1.0) == 100.1


def test_fixed_bps_slippage_worsens_sell_price_down() -> None:
    assert FixedBpsSlippage(10.0).apply(Side.SELL, 100.0, 1.0) == 99.9

