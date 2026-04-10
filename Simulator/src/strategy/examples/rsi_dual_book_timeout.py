from __future__ import annotations

from datetime import timedelta

from src.actions import BaseAction, CancelOrderAction, PlaceOrderAction
from src.enums import OrderType, Side
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.examples.indicators import compute_rsi


class RSIDualBookTimeoutStrategy(BaseStrategy):
    """RSI limit-order strategy that trades independently in multiple books.

    Works in both single-book mode (identical to RSILimitOrderTimeoutStrategy)
    and multi-book mode.  The config key ``trading_instrument_ids`` lists the
    instruments where orders may be placed; any other instrument is info-only.
    If the key is absent every instrument is treated as tradeable.
    """

    def __init__(self, strategy_id: str, config: dict | None = None) -> None:
        super().__init__(strategy_id, config)
        self._mid_prices: dict[str, list[float]] = {}

    def on_simulation_start(self) -> None:
        super().on_simulation_start()
        self._mid_prices.clear()

    def on_snapshot(self, context: StrategyContext) -> list[BaseAction]:
        snapshot = context.snapshot
        if snapshot is None:
            return []

        instrument_id = snapshot.instrument_id
        mid = snapshot.mid_price()
        if mid is None:
            return []

        self._mid_prices.setdefault(instrument_id, []).append(mid)

        # If this instrument is info-only, do not trade
        trading_ids: list[str] = self.config.get("trading_instrument_ids", [])
        if trading_ids and instrument_id not in trading_ids:
            return []

        period = int(self.config.get("rsi_period", 14))
        oversold = float(self.config.get("oversold", 30.0))
        overbought = float(self.config.get("overbought", 70.0))
        order_qty = float(self.config.get("qty", 1.0))
        order_ttl_seconds = float(self.config.get("order_ttl_seconds", 5.0))

        rsi = compute_rsi(self._mid_prices[instrument_id], period)

        # Cancel stale orders for this instrument only
        cancel_actions = self._cancel_stale(context, instrument_id, order_ttl_seconds)
        if cancel_actions:
            return cancel_actions

        # Filter active orders to this instrument
        instrument_orders = {
            oid: o for oid, o in context.active_orders.items()
            if o.instrument_id == instrument_id
        }
        if rsi is None or instrument_orders:
            return []

        position = context.positions.get(instrument_id, context.position if instrument_id == "default" else 0.0)
        best_ask = snapshot.best_ask()
        best_bid = snapshot.best_bid()

        if position <= 0 and rsi <= oversold and best_bid is not None:
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.BUY,
                    price=best_bid.price,
                    qty=order_qty,
                    order_type=OrderType.LIMIT,
                    instrument_id=instrument_id,
                )
            ]

        if position > 0 and rsi >= overbought and best_ask is not None:
            return [
                PlaceOrderAction(
                    strategy_id=self.strategy_id,
                    side=Side.SELL,
                    price=best_ask.price,
                    qty=min(order_qty, position),
                    order_type=OrderType.LIMIT,
                    instrument_id=instrument_id,
                )
            ]

        return []

    def _cancel_stale(
        self,
        context: StrategyContext,
        instrument_id: str,
        order_ttl_seconds: float,
    ) -> list[BaseAction]:
        ttl = timedelta(seconds=order_ttl_seconds)
        stale_ids = [
            order.order_id
            for order in context.active_orders.values()
            if order.instrument_id == instrument_id
            and context.timestamp - order.created_at >= ttl
        ]
        return [CancelOrderAction(strategy_id=self.strategy_id, order_id=oid) for oid in stale_ids]
