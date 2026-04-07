from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from run_backtest import run
from sim.exchange.exchange_sim import PlaceOrderRequest
from sim.exchange.orders import Fill, Order
from sim.config.backtest import BacktestConfig, load_backtest_config
from sim.market.orderbook_l2 import OrderBookL2
from sim.strategy.strategy_base import StrategyBase


def test_load_backtest_config_uses_defaults_for_missing_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"l2_path": "custom.csv"},
                "strategy": {"name": "taker", "taker": {"window": 50}},
            }
        ),
        encoding="utf-8",
    )

    config = load_backtest_config(config_path)

    assert config.data.l2_path == "custom.csv"
    assert config.strategy.name == "taker"
    assert config.strategy.taker.window == 50
    assert config.strategy.taker.std_mult == BacktestConfig().strategy.taker.std_mult
    assert config.dashboard.port == 860


def test_load_backtest_config_rejects_unknown_strategy(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.json"
    config_path.write_text(
        json.dumps({"strategy": {"name": "unsupported"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strategy.name"):
        load_backtest_config(config_path)


def test_run_detailed_console_output_includes_order_and_book(tmp_path: Path, capsys) -> None:
    l2_path = tmp_path / "l2.csv"
    l2_path.write_text(
        "time,symbol,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "1.0,TEST,101.0,100.0,2.0,2.0\n"
        "2.0,TEST,102.0,101.0,2.0,2.0\n",
        encoding="utf-8",
    )

    run(
        l2_path=str(l2_path),
        console_level=2,
        console_book_levels=1,
    )
    captured = capsys.readouterr()

    assert "[strategy ts=" in captured.out
    assert "place id=" in captured.out
    assert "best_bid=" in captured.out
    assert "best_ask=" in captured.out


@dataclass
class _OneShotMarketBuyStrategy(StrategyBase):
    fired: bool = False

    def on_snapshot(
        self,
        ts: float,
        book: OrderBookL2,
        active_orders: list[Order],
    ) -> list[PlaceOrderRequest]:
        _ = book
        _ = active_orders
        if self.fired:
            return []
        self.fired = True
        return [
            PlaceOrderRequest(
                order_id="market-buy-1",
                side="BUY",
                type="MARKET",
                price=None,
                qty=1.0,
            )
        ]

    def on_fill(self, fill: Fill) -> None:
        _ = fill


def test_run_console_level_3_prints_fill_position_and_equity(tmp_path: Path, capsys) -> None:
    l2_path = tmp_path / "l2.csv"
    l2_path.write_text(
        "time,symbol,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "1.0,TEST,101.0,100.0,2.0,2.0\n"
        "2.0,TEST,102.0,101.0,2.0,2.0\n",
        encoding="utf-8",
    )

    run(
        l2_path=str(l2_path),
        strategy=_OneShotMarketBuyStrategy(),
        console_level=3,
        console_book_levels=2,
    )
    captured = capsys.readouterr()

    assert "[fill ts=" in captured.out
    assert "position=" in captured.out
    assert "cash=" in captured.out
    assert "equity=" in captured.out
