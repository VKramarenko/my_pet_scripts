from sim.core.events import MarketSnapshot
from sim.execution.config import ExecutionConfig
from sim.execution.fill_model_touch import TouchFillModel
from sim.exchange.exchange_sim import ExchangeSim, PlaceOrderRequest
from sim.market.orderbook_l2 import OrderBookL2


def test_touch_buy_limit_fills_on_cross() -> None:
    book = OrderBookL2(depth=5)
    model = TouchFillModel(ExecutionConfig(fill_price="best_price"))
    exchange = ExchangeSim(book=book, fill_model=model)
    snapshot = MarketSnapshot(ts=1.0, bids=[(100, 2)], asks=[(101, 3)])
    book.update_from_snapshot(snapshot)
    exchange.submit_place(
        PlaceOrderRequest(order_id="o1", side="BUY", type="LIMIT", price=102.0, qty=2.0), ts=1.0
    )

    fills = exchange.on_market_event(snapshot)
    assert len(fills) == 1
    assert fills[0].order_id == "o1"
    assert fills[0].price == 101
    assert fills[0].qty == 2


def test_touch_sell_limit_fills_on_cross() -> None:
    book = OrderBookL2(depth=5)
    model = TouchFillModel()
    exchange = ExchangeSim(book=book, fill_model=model)
    snapshot = MarketSnapshot(ts=1.0, bids=[(100, 5)], asks=[(101, 2)])
    book.update_from_snapshot(snapshot)
    exchange.submit_place(
        PlaceOrderRequest(order_id="o2", side="SELL", type="LIMIT", price=99.0, qty=1.0), ts=1.0
    )

    fills = exchange.on_market_event(snapshot)
    assert len(fills) == 1
    assert fills[0].order_id == "o2"
    assert fills[0].qty == 1


def test_touch_partial_fill_with_top_liquidity_limit() -> None:
    book = OrderBookL2(depth=5)
    model = TouchFillModel()
    exchange = ExchangeSim(book=book, fill_model=model)
    snapshot = MarketSnapshot(ts=1.0, bids=[(100, 5)], asks=[(101, 1.5)])
    book.update_from_snapshot(snapshot)
    exchange.submit_place(
        PlaceOrderRequest(order_id="o3", side="BUY", type="LIMIT", price=102.0, qty=3.0), ts=1.0
    )

    fills = exchange.on_market_event(snapshot)
    assert len(fills) == 1
    assert fills[0].qty == 1.5

