import bisect
import gzip
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


Price = Decimal
Size = Decimal


def _to_decimal(x) -> Decimal:
    return Decimal(str(x))


def _iter_jsonl_gz(path: str) -> Iterator[dict]:
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _list_matching_files(directory: str, prefix: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    files = []
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".jsonl.gz"):
            files.append(os.path.join(directory, name))
    files.sort()
    return files


@dataclass
class BookState:
    exchange: str
    symbol: str
    ts_event: int
    bids: Dict[Price, Size]
    asks: Dict[Price, Size]

    def best_bid(self) -> Optional[Tuple[Price, Size]]:
        if not self.bids:
            return None
        p = max(self.bids)
        return p, self.bids[p]

    def best_ask(self) -> Optional[Tuple[Price, Size]]:
        if not self.asks:
            return None
        p = min(self.asks)
        return p, self.asks[p]

    def mid_price(self) -> Optional[Decimal]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / Decimal("2")

    def spread(self) -> Optional[Decimal]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return ba[0] - bb[0]

    def top_n(self, n: int = 10) -> dict:
        bid_levels = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:n]
        ask_levels = sorted(self.asks.items(), key=lambda x: x[0])[:n]
        return {
            "bids": bid_levels,
            "asks": ask_levels,
        }


class OrderBook:
    """
    Простая L2-модель стакана:
    - key: price
    - value: aggregated size
    """

    def __init__(self):
        self.bids: Dict[Price, Size] = {}
        self.asks: Dict[Price, Size] = {}
        self.ts_event: int = 0

    def load_snapshot(self, bids: List[List[str]], asks: List[List[str]], ts_event: int):
        self.bids = {}
        self.asks = {}

        for price, size in bids:
            p = _to_decimal(price)
            s = _to_decimal(size)
            if s > 0:
                self.bids[p] = s

        for price, size in asks:
            p = _to_decimal(price)
            s = _to_decimal(size)
            if s > 0:
                self.asks[p] = s

        self.ts_event = ts_event

    def apply_side_updates(self, side_book: Dict[Price, Size], updates: List[List[str]]):
        for price, size in updates:
            p = _to_decimal(price)
            s = _to_decimal(size)
            if s == 0:
                side_book.pop(p, None)
            else:
                side_book[p] = s

    def apply_binance_diff(self, event: dict):
        self.apply_side_updates(self.bids, event.get("b", []))
        self.apply_side_updates(self.asks, event.get("a", []))
        self.ts_event = int(event["ts_event"])

    def apply_bybit_delta(self, event: dict):
        self.apply_side_updates(self.bids, event.get("b", []))
        self.apply_side_updates(self.asks, event.get("a", []))
        self.ts_event = int(event["ts_event"])

    def snapshot(self, exchange: str, symbol: str) -> BookState:
        return BookState(
            exchange=exchange,
            symbol=symbol,
            ts_event=self.ts_event,
            bids=dict(self.bids),
            asks=dict(self.asks),
        )


@dataclass
class IndexedSnapshot:
    ts_event: int
    path: str
    line_no: int
    event: dict


class BaseReplay:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def restore_at(self, target_ts: int) -> BookState:
        raise NotImplementedError

    def iter_states(self, start_ts: int, end_ts: int) -> Iterator[BookState]:
        raise NotImplementedError


class BinanceReplay(BaseReplay):
    """
    Ожидает структуру:
      data/
        snapshots/binance_BTCUSDT_YYYY-MM-DD.jsonl.gz
        diff/binance_BTCUSDT_YYYY-MM-DD.jsonl.gz
        meta/binance_BTCUSDT_YYYY-MM-DD.jsonl.gz
    """

    def __init__(self, data_dir: str, symbol: str):
        super().__init__(data_dir)
        self.exchange = "binance"
        self.symbol = symbol.upper()

        self.snap_dir = os.path.join(data_dir, "snapshots")
        self.diff_dir = os.path.join(data_dir, "diff")

        self.snap_prefix = f"binance_{self.symbol}_"
        self.diff_prefix = f"binance_{self.symbol}_"

        self._snapshot_index: Optional[List[IndexedSnapshot]] = None

    def _build_snapshot_index(self) -> List[IndexedSnapshot]:
        out: List[IndexedSnapshot] = []
        for path in _list_matching_files(self.snap_dir, self.snap_prefix):
            for i, event in enumerate(_iter_jsonl_gz(path)):
                if event.get("kind") != "snapshot":
                    continue
                ts = int(event.get("ts_capture", 0))
                out.append(
                    IndexedSnapshot(
                        ts_event=ts,
                        path=path,
                        line_no=i,
                        event=event,
                    )
                )
        out.sort(key=lambda x: x.ts_event)
        return out

    @property
    def snapshot_index(self) -> List[IndexedSnapshot]:
        if self._snapshot_index is None:
            self._snapshot_index = self._build_snapshot_index()
        return self._snapshot_index

    def _find_snapshot_for_ts(self, target_ts: int) -> IndexedSnapshot:
        index = self.snapshot_index
        if not index:
            raise RuntimeError("No Binance snapshots found")

        keys = [x.ts_event for x in index]
        pos = bisect.bisect_right(keys, target_ts) - 1
        if pos < 0:
            raise RuntimeError("No snapshot exists before target_ts")
        return index[pos]

    def _find_next_snapshot_at_or_after(self, ts_ms: int) -> Optional[IndexedSnapshot]:
        """Первый снапшот с ts_event >= ts_ms, или None если не найден."""
        index = self.snapshot_index
        keys = [x.ts_event for x in index]
        pos = bisect.bisect_left(keys, ts_ms)
        return index[pos] if pos < len(index) else None

    def _iter_diff_events(self, start_ts: int, end_ts: int) -> Iterator[dict]:
        for path in _list_matching_files(self.diff_dir, self.diff_prefix):
            for event in _iter_jsonl_gz(path):
                if event.get("kind") != "diff":
                    continue
                ts = int(event["ts_event"])
                if ts < start_ts:
                    continue
                if ts > end_ts:
                    break
                yield event

    def restore_at(self, target_ts: int) -> BookState:
        snap = self._find_snapshot_for_ts(target_ts)

        ob = OrderBook()
        ob.load_snapshot(
            bids=snap.event["bids"],
            asks=snap.event["asks"],
            ts_event=int(snap.event["ts_capture"]),
        )

        last_update_id = int(snap.event["lastUpdateId"])

        for event in self._iter_diff_events(start_ts=snap.ts_event, end_ts=target_ts):
            U = int(event["U"])
            u = int(event["u"])

            if u <= last_update_id:
                continue

            if U > last_update_id + 1:
                ts_event = int(event["ts_event"])
                next_snap = self._find_next_snapshot_at_or_after(ts_event)
                if next_snap is None or next_snap.ts_event > target_ts:
                    break
                print(
                    f"[replay] gap in restore_at (last_id={last_update_id}, U={U}), "
                    f"resyncing from snapshot ts={next_snap.ts_event}"
                )
                ob.load_snapshot(
                    bids=next_snap.event["bids"],
                    asks=next_snap.event["asks"],
                    ts_event=int(next_snap.event["ts_capture"]),
                )
                last_update_id = int(next_snap.event["lastUpdateId"])
                continue

            ob.apply_binance_diff(event)
            last_update_id = u

        return ob.snapshot(exchange=self.exchange, symbol=self.symbol)

    def iter_states(self, start_ts: int, end_ts: int) -> Iterator[BookState]:
        snap = self._find_snapshot_for_ts(start_ts)

        ob = OrderBook()
        ob.load_snapshot(
            bids=snap.event["bids"],
            asks=snap.event["asks"],
            ts_event=int(snap.event["ts_capture"]),
        )

        last_update_id = int(snap.event["lastUpdateId"])

        for event in self._iter_diff_events(start_ts=snap.ts_event, end_ts=end_ts):
            U = int(event["U"])
            u = int(event["u"])

            if u <= last_update_id:
                continue

            if U > last_update_id + 1:
                ts_event = int(event["ts_event"])
                next_snap = self._find_next_snapshot_at_or_after(ts_event)
                if next_snap is None:
                    break
                print(
                    f"[replay] gap detected (last_id={last_update_id}, U={U}), "
                    f"resyncing from snapshot ts={next_snap.ts_event}"
                )
                ob.load_snapshot(
                    bids=next_snap.event["bids"],
                    asks=next_snap.event["asks"],
                    ts_event=int(next_snap.event["ts_capture"]),
                )
                last_update_id = int(next_snap.event["lastUpdateId"])
                continue

            ob.apply_binance_diff(event)
            last_update_id = u

            if ob.ts_event >= start_ts:
                yield ob.snapshot(exchange=self.exchange, symbol=self.symbol)


class BybitReplay(BaseReplay):
    """
    Ожидает структуру:
      data/
        snapshots/bybit_spot_BTCUSDT_YYYY-MM-DD.jsonl.gz
        delta/bybit_spot_BTCUSDT_YYYY-MM-DD.jsonl.gz
        meta/bybit_spot_BTCUSDT_YYYY-MM-DD.jsonl.gz
    """

    def __init__(self, data_dir: str, category: str, symbol: str):
        super().__init__(data_dir)
        self.exchange = "bybit"
        self.category = category
        self.symbol = symbol.upper()

        self.snap_dir = os.path.join(data_dir, "snapshots")
        self.delta_dir = os.path.join(data_dir, "delta")

        self.snap_prefix = f"bybit_{self.category}_{self.symbol}_"
        self.delta_prefix = f"bybit_{self.category}_{self.symbol}_"

        self._snapshot_index: Optional[List[IndexedSnapshot]] = None

    def _build_snapshot_index(self) -> List[IndexedSnapshot]:
        out: List[IndexedSnapshot] = []
        for path in _list_matching_files(self.snap_dir, self.snap_prefix):
            for i, event in enumerate(_iter_jsonl_gz(path)):
                if event.get("kind") != "snapshot":
                    continue
                ts = int(event.get("ts_event", 0))
                out.append(
                    IndexedSnapshot(
                        ts_event=ts,
                        path=path,
                        line_no=i,
                        event=event,
                    )
                )
        out.sort(key=lambda x: x.ts_event)
        return out

    @property
    def snapshot_index(self) -> List[IndexedSnapshot]:
        if self._snapshot_index is None:
            self._snapshot_index = self._build_snapshot_index()
        return self._snapshot_index

    def _find_snapshot_for_ts(self, target_ts: int) -> IndexedSnapshot:
        index = self.snapshot_index
        if not index:
            raise RuntimeError("No Bybit snapshots found")

        keys = [x.ts_event for x in index]
        pos = bisect.bisect_right(keys, target_ts) - 1
        if pos < 0:
            raise RuntimeError("No snapshot exists before target_ts")
        return index[pos]

    def _iter_delta_events(self, start_ts: int, end_ts: int) -> Iterator[dict]:
        for path in _list_matching_files(self.delta_dir, self.delta_prefix):
            for event in _iter_jsonl_gz(path):
                kind = event.get("kind")
                if kind not in ("delta", "snapshot"):
                    continue
                ts = int(event["ts_event"])
                if ts < start_ts:
                    continue
                if ts > end_ts:
                    break
                yield event

    def restore_at(self, target_ts: int) -> BookState:
        snap = self._find_snapshot_for_ts(target_ts)

        ob = OrderBook()
        ob.load_snapshot(
            bids=snap.event["b"],
            asks=snap.event["a"],
            ts_event=int(snap.event["ts_event"]),
        )

        for event in self._iter_delta_events(start_ts=snap.ts_event, end_ts=target_ts):
            kind = event["kind"]

            if kind == "snapshot":
                ob.load_snapshot(
                    bids=event["b"],
                    asks=event["a"],
                    ts_event=int(event["ts_event"]),
                )
            else:
                ob.apply_bybit_delta(event)

        return ob.snapshot(exchange=self.exchange, symbol=self.symbol)

    def iter_states(self, start_ts: int, end_ts: int) -> Iterator[BookState]:
        snap = self._find_snapshot_for_ts(start_ts)

        ob = OrderBook()
        ob.load_snapshot(
            bids=snap.event["b"],
            asks=snap.event["a"],
            ts_event=int(snap.event["ts_event"]),
        )

        for event in self._iter_delta_events(start_ts=snap.ts_event, end_ts=end_ts):
            kind = event["kind"]

            if kind == "snapshot":
                ob.load_snapshot(
                    bids=event["b"],
                    asks=event["a"],
                    ts_event=int(event["ts_event"]),
                )
            else:
                ob.apply_bybit_delta(event)

            if ob.ts_event >= start_ts:
                yield ob.snapshot(exchange=self.exchange, symbol=self.symbol)


# -----------------------------
# Полезные утилиты для исполнения
# -----------------------------

def simulate_market_buy(book: BookState, qty: Decimal) -> dict:
    """
    Купить qty по asks.
    """
    asks = sorted(book.asks.items(), key=lambda x: x[0])

    filled = Decimal("0")
    cost = Decimal("0")

    fills = []
    for price, size in asks:
        if filled >= qty:
            break
        take = min(qty - filled, size)
        if take <= 0:
            continue
        cost += take * price
        filled += take
        fills.append((price, take))

    avg_px = (cost / filled) if filled > 0 else None
    return {
        "requested_qty": qty,
        "filled_qty": filled,
        "notional": cost,
        "avg_price": avg_px,
        "fills": fills,
        "fully_filled": filled == qty,
    }


def simulate_market_sell(book: BookState, qty: Decimal) -> dict:
    """
    Продать qty по bids.
    """
    bids = sorted(book.bids.items(), key=lambda x: x[0], reverse=True)

    filled = Decimal("0")
    proceeds = Decimal("0")

    fills = []
    for price, size in bids:
        if filled >= qty:
            break
        take = min(qty - filled, size)
        if take <= 0:
            continue
        proceeds += take * price
        filled += take
        fills.append((price, take))

    avg_px = (proceeds / filled) if filled > 0 else None
    return {
        "requested_qty": qty,
        "filled_qty": filled,
        "notional": proceeds,
        "avg_price": avg_px,
        "fills": fills,
        "fully_filled": filled == qty,
    }


# -----------------------------
# Пример использования
# -----------------------------

if __name__ == "__main__":
    from pprint import pprint

    # Binance example
    # replay = BinanceReplay(data_dir="./data", symbol="BTCUSDT")

    # Bybit example
    replay = BybitReplay(data_dir="./data", category="spot", symbol="BTCUSDT")

    target_ts = 1710000000000
    book = replay.restore_at(target_ts)

    print("ts_event:", book.ts_event)
    print("best_bid:", book.best_bid())
    print("best_ask:", book.best_ask())
    print("mid:", book.mid_price())
    print("spread:", book.spread())

    print("\nTop 5:")
    pprint(book.top_n(5))

    print("\nSimulate market buy 0.25:")
    result = simulate_market_buy(book, Decimal("0.25"))
    pprint(result)