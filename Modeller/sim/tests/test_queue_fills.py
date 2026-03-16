from sim.core.events import MarketSnapshot, MarketTrade
from sim.execution.config import ExecutionConfig
from sim.execution.fill_model_queue import QueueFillModel
from sim.exchange.exchange_sim import ExchangeSim, PlaceOrderRequest
from sim.market.orderbook_l2 import OrderBookL2


def test_queue_fill_progresses_with_trades() -> None:
    book = OrderBookL2(depth=5)
    model = QueueFillModel()
    exchange = ExchangeSim(book=book, fill_model=model)

    s1 = MarketSnapshot(ts=1.0, bids=[(100, 10)], asks=[(101, 10)])
    book.update_from_snapshot(s1)
    exchange.submit_place(
        PlaceOrderRequest(order_id="q1", side="BUY", type="LIMIT", price=100.0, qty=5.0), ts=1.0
    )
    exchange.on_market_event(s1)
    state = model.debug_state("q1")
    assert state is not None
    assert state.queue_ahead == 10

    t1 = MarketTrade(ts=1.1, price=100, size=7, side="seller_initiated")
    fills_1 = exchange.on_market_event(t1)
    state = model.debug_state("q1")
    assert state is not None
    assert state.queue_ahead == 3
    assert fills_1 == []

    t2 = MarketTrade(ts=1.2, price=100, size=5, side="seller_initiated")
    fills_2 = exchange.on_market_event(t2)
    assert len(fills_2) == 1
    assert fills_2[0].qty == 2


def test_queue_mode_pessimistic_vs_optimistic_on_growth() -> None:
    p_book = OrderBookL2(depth=5)
    p_model = QueueFillModel(ExecutionConfig(mode="pessimistic"))
    p_exchange = ExchangeSim(book=p_book, fill_model=p_model)

    s1 = MarketSnapshot(ts=1.0, bids=[(100, 10)], asks=[(101, 10)])
    p_book.update_from_snapshot(s1)
    p_exchange.submit_place(
        PlaceOrderRequest(order_id="p1", side="BUY", type="LIMIT", price=100.0, qty=1.0), ts=1.0
    )
    p_exchange.on_market_event(s1)

    s2 = MarketSnapshot(ts=1.1, bids=[(100, 13)], asks=[(101, 10)])
    p_book.update_from_snapshot(s2)
    p_exchange.on_market_event(s2)
    p_state = p_model.debug_state("p1")
    assert p_state is not None
    assert p_state.queue_ahead == 13

    o_book = OrderBookL2(depth=5)
    o_model = QueueFillModel(ExecutionConfig(mode="optimistic"))
    o_exchange = ExchangeSim(book=o_book, fill_model=o_model)
    o_book.update_from_snapshot(s1)
    o_exchange.submit_place(
        PlaceOrderRequest(order_id="o1", side="BUY", type="LIMIT", price=100.0, qty=1.0), ts=1.0
    )
    o_exchange.on_market_event(s1)
    o_book.update_from_snapshot(s2)
    o_exchange.on_market_event(s2)
    o_state = o_model.debug_state("o1")
    assert o_state is not None
    assert o_state.queue_ahead == 10

