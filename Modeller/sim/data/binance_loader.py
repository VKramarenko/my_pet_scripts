from __future__ import annotations

from collections.abc import Iterator

from sim.core.events import MarketSnapshot


def load_binance_l2(
    data_dir: str,
    symbol: str,
    depth: int = 25,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Iterator[MarketSnapshot]:
    """
    Загружает Binance L2-данные из JSONL.gz (snapshot + diff),
    восстанавливает стакан и конвертирует в MarketSnapshot.

    data_dir  — директория, в которой находятся папки snapshots/ и diff/
    symbol    — тикер, напр. "BTCUSDT"
    depth     — максимальная глубина стакана на каждую сторону
    start_ts  — начало диапазона в миллисекундах (включительно)
    end_ts    — конец диапазона в миллисекундах (включительно)
    """
    # Импорт здесь, чтобы не тянуть зависимость при импорте модуля
    from replay_data_binance import BinanceReplay

    replay = BinanceReplay(data_dir=data_dir, symbol=symbol)

    if start_ts is None:
        idx = replay.snapshot_index
        if not idx:
            raise RuntimeError(
                f"No Binance snapshots found in {data_dir}/snapshots/ "
                f"for symbol {symbol}"
            )
        start_ts = idx[0].ts_event

    if end_ts is None:
        end_ts = 10 ** 18  # effectively infinity

    for book_state in replay.iter_states(start_ts=start_ts, end_ts=end_ts):
        bids = sorted(
            ((float(p), float(s)) for p, s in book_state.bids.items()),
            reverse=True,
        )[:depth]
        asks = sorted(
            ((float(p), float(s)) for p, s in book_state.asks.items())
        )[:depth]
        ts_sec = book_state.ts_event / 1000.0
        yield MarketSnapshot(ts=ts_sec, bids=bids, asks=asks)
