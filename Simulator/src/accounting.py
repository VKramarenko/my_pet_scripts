from __future__ import annotations

from dataclasses import replace

from src.enums import Side
from src.models import Snapshot, Trade
from src.pnl import compute_mid_price
from src.strategy.metrics import update_metrics_after_trade, update_metrics_after_valuation
from src.strategy.state import StrategyState


def apply_trade_to_strategy_state(
    state: StrategyState,
    trade: Trade,
    *,
    step_index: int | None = None,
) -> None:
    state.trades.append(replace(trade))
    signed_qty = trade.qty if trade.side == Side.BUY else -trade.qty
    previous_position = state.position
    notional = trade.notional if trade.notional is not None else trade.price * trade.qty

    if trade.side == Side.BUY:
        state.cash -= notional + trade.commission
    else:
        state.cash += notional - trade.commission

    realized_delta = 0.0
    if previous_position == 0:
        state.position = signed_qty
        state.avg_entry_price = trade.price
    elif previous_position > 0:
        if signed_qty > 0:
            new_position = previous_position + signed_qty
            state.avg_entry_price = ((previous_position * (state.avg_entry_price or 0.0)) + (signed_qty * trade.price)) / new_position
            state.position = new_position
        else:
            close_qty = min(previous_position, abs(signed_qty))
            realized_delta += close_qty * (trade.price - (state.avg_entry_price or 0.0))
            new_position = previous_position + signed_qty
            state.position = new_position
            if new_position < 0:
                state.avg_entry_price = trade.price
            elif new_position == 0:
                state.avg_entry_price = None
    else:
        if signed_qty < 0:
            new_position = previous_position + signed_qty
            previous_abs = abs(previous_position)
            added_abs = abs(signed_qty)
            state.avg_entry_price = ((previous_abs * (state.avg_entry_price or 0.0)) + (added_abs * trade.price)) / abs(new_position)
            state.position = new_position
        else:
            close_qty = min(abs(previous_position), signed_qty)
            realized_delta += close_qty * ((state.avg_entry_price or 0.0) - trade.price)
            new_position = previous_position + signed_qty
            state.position = new_position
            if new_position > 0:
                state.avg_entry_price = trade.price
            elif new_position == 0:
                state.avg_entry_price = None

    state.realized_pnl += realized_delta - trade.commission
    if state.position == 0:
        state.avg_entry_price = None
    if step_index is not None:
        state.last_trade_step_index = step_index
    update_metrics_after_trade(state.metrics, trade)

    # Keep per-instrument positions in sync
    state.positions[trade.instrument_id] = state.positions.get(trade.instrument_id, 0.0) + signed_qty
    if trade.instrument_id == "default":
        # state.position is the authoritative value for "default" (computed via complex logic above);
        # override the accumulation to avoid any rounding drift and keep both in sync
        state.positions["default"] = state.position


def mark_strategy_to_market(
    state: StrategyState,
    snapshot: Snapshot,
    *,
    instrument_id: str = "default",
    missing_policy: str = "keep_last",
) -> None:
    mid = compute_mid_price(snapshot)
    if mid is None:
        if missing_policy == "zero":
            state.unrealized_pnl = 0.0
            state.equity = state.cash
            update_metrics_after_valuation(state.metrics, snapshot.timestamp, state.equity, state.realized_pnl)
        elif missing_policy == "none":
            return
        elif missing_policy == "keep_last":
            mid = state.last_mid_prices.get(instrument_id) or state.last_mid_price
            if mid is None:
                return
        else:
            return

    state.last_mid_prices[instrument_id] = mid
    if instrument_id == "default":
        state.last_mid_price = mid

    # Per-instrument unrealized PnL (only tracked for "default" for backward compat)
    if instrument_id == "default":
        if state.position == 0 or state.avg_entry_price is None:
            state.unrealized_pnl = 0.0
        else:
            state.unrealized_pnl = state.position * (mid - state.avg_entry_price)

    # Equity = cash + sum of (position × mid) across all known instruments
    equity = state.cash
    for iid, pos in state.positions.items():
        iid_mid = state.last_mid_prices.get(iid)
        if iid_mid is not None and pos != 0:
            equity += pos * iid_mid
    # Also include "default" position if not already in positions dict
    if "default" not in state.positions and state.position != 0:
        default_mid = state.last_mid_prices.get("default") or state.last_mid_price
        if default_mid is not None:
            equity += state.position * default_mid

    state.equity = equity
    update_metrics_after_valuation(state.metrics, snapshot.timestamp, state.equity, state.realized_pnl)
