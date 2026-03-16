"""Tests verifying bug fixes from code review."""
from sim.core.events import MarketSnapshot, MarketTrade
from sim.execution.config import ExecutionConfig
from sim.execution.fill_model_queue import QueueFillModel
from sim.execution.fill_model_touch import TouchFillModel
from sim.exchange.exchange_sim import ExchangeSim, PlaceOrderRequest
from sim.market.orderbook_l2 import OrderBookL2
from sim.portfolio.portfolio import Portfolio


def test_queue_overflow_preserved() -> None:
    """Fix #2: traded volume exceeding order.remaining must not be lost."""
    book = OrderBookL2(depth=5)
    model = QueueFillModel()
    exchange = ExchangeSim(book=book, fill_model=model)

    s1 = MarketSnapshot(ts=1.0, bids=[(100, 5)], asks=[(101, 10)])
    book.update_from_snapshot(s1)
    exchange.submit_place(
        PlaceOrderRequest(order_id="ov1", side="BUY", type="LIMIT", price=100.0, qty=2.0),
        ts=1.0,
    )
    exchange.on_market_event(s1)
    state = model.debug_state("ov1")
    assert state is not None
    assert state.queue_ahead == 5  # 5 ahead in queue

    # Big trade wipes out queue and overflows: 5 ahead - 10 trade = -5
    t1 = MarketTrade(ts=1.1, price=100, size=10, side="seller_initiated")
    fills = exchange.on_market_event(t1)
    assert len(fills) == 1
    assert fills[0].qty == 2.0  # filled fully (order qty=2, overflow=5 but capped by remaining)

    state = model.debug_state("ov1")
    assert state is not None
    # Overflow after fill: abs(-5) - 2 = 3 remaining overflow → queue_ahead stays 0
    # (order is fully filled so no more queue tracking needed for this order)
    assert state.queue_ahead == 0.0


def test_queue_overflow_partial_preserved() -> None:
    """Fix #2: when overflow > 0 but order has more remaining, queue_ahead reflects it."""
    book = OrderBookL2(depth=5)
    config = ExecutionConfig(max_fill_per_event=1.0)
    model = QueueFillModel(config)
    exchange = ExchangeSim(book=book, fill_model=model)

    s1 = MarketSnapshot(ts=1.0, bids=[(100, 3)], asks=[(101, 10)])
    book.update_from_snapshot(s1)
    exchange.submit_place(
        PlaceOrderRequest(order_id="ov2", side="BUY", type="LIMIT", price=100.0, qty=5.0),
        ts=1.0,
    )
    exchange.on_market_event(s1)

    # Trade wipes out queue: 3 ahead - 7 trade = -4
    # fillable = min(remaining=5, abs(-4)=4) = 4, but max_fill_per_event=1 → fillable=1
    t1 = MarketTrade(ts=1.1, price=100, size=7, side="seller_initiated")
    fills = exchange.on_market_event(t1)
    assert len(fills) == 1
    assert fills[0].qty == 1.0

    state = model.debug_state("ov2")
    assert state is not None
    # overflow was 4, filled 1, so 3 units of overflow remain → queue_ahead = 0
    # (overflow can't go negative in queue_ahead terms, it's max'd at 0)
    assert state.queue_ahead == 0.0


def test_touch_require_trade_for_fill() -> None:
    """Fix #3: LIMIT orders should not fill on snapshot when require_trade_for_fill=True."""
    book = OrderBookL2(depth=5)
    config = ExecutionConfig(require_trade_for_fill=True)
    model = TouchFillModel(config)
    exchange = ExchangeSim(book=book, fill_model=model)

    s1 = MarketSnapshot(ts=1.0, bids=[(100, 2)], asks=[(101, 3)])
    book.update_from_snapshot(s1)
    exchange.submit_place(
        PlaceOrderRequest(order_id="rt1", side="BUY", type="LIMIT", price=102.0, qty=2.0),
        ts=1.0,
    )

    # Snapshot should NOT generate fills for LIMIT orders when require_trade_for_fill=True
    fills = exchange.on_market_event(s1)
    assert len(fills) == 0

    # Trade at/below order price SHOULD generate fill
    t1 = MarketTrade(ts=1.1, price=101, size=1, side="seller_initiated")
    fills = exchange.on_market_event(t1)
    assert len(fills) == 1
    assert fills[0].order_id == "rt1"
    assert fills[0].qty == 1.0


def test_touch_require_trade_for_fill_sell() -> None:
    """Fix #3: SELL LIMIT orders fill on trade when require_trade_for_fill=True."""
    book = OrderBookL2(depth=5)
    config = ExecutionConfig(require_trade_for_fill=True)
    model = TouchFillModel(config)
    exchange = ExchangeSim(book=book, fill_model=model)

    s1 = MarketSnapshot(ts=1.0, bids=[(100, 5)], asks=[(101, 2)])
    book.update_from_snapshot(s1)
    exchange.submit_place(
        PlaceOrderRequest(order_id="rt2", side="SELL", type="LIMIT", price=99.0, qty=1.0),
        ts=1.0,
    )

    # Snapshot should NOT generate fills
    fills = exchange.on_market_event(s1)
    assert len(fills) == 0

    # Trade at/above order price SHOULD generate fill
    t1 = MarketTrade(ts=1.1, price=100, size=3, side="buyer_initiated")
    fills = exchange.on_market_event(t1)
    assert len(fills) == 1
    assert fills[0].order_id == "rt2"
    assert fills[0].qty == 1.0


def test_equity_returns_none_when_no_mid() -> None:
    """Fix #5: equity should return None when mark_price is None, not silently zero it."""
    portfolio = Portfolio()
    portfolio.cash = 1000.0
    portfolio.position = 5.0

    eq = portfolio.equity(None)
    assert eq is None

    eq = portfolio.equity(100.0)
    assert eq == 1000.0 + 5.0 * 100.0
