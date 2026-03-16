from pathlib import Path

from sim.data.loaders import (
    load_bybit_l2_snapshots,
    load_bybit_trades,
    load_test_data_orderbooks,
)


def test_bybit_loader_reads_custom_json_streams() -> None:
    base = Path("test_data/bybit_custom_loader")
    l2_path = base / "orderbook_BTCUSDT_spot_2026-03-04.json"
    trades_path = base / "trades_BTCUSDT_spot_2026-03-04.json"

    snapshots = list(load_bybit_l2_snapshots(l2_path))
    trades = list(load_bybit_trades(trades_path))

    assert len(snapshots) > 0
    assert len(trades) > 0
    assert snapshots[0].bids
    assert snapshots[0].asks
    assert trades[0].side in {"buyer_initiated", "seller_initiated"}


def test_test_data_orderbook_loader_resolves_filename() -> None:
    snapshots = list(load_test_data_orderbooks("orderbook_BTCUSDT_spot_2026-03-04.json"))

    assert len(snapshots) > 0
    assert snapshots[0].bids
    assert snapshots[0].asks

