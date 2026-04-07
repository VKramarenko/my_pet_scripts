"""
recorder_binance.py — запись и скачивание рыночных данных с Binance и Bybit.

Режимы:
  live     — запись стакана в реальном времени (Binance или Bybit)
  history  — скачивание исторических трейдов за период (только Binance)

Использование:
  # Live-запись Binance 1 час:
  python recorder_binance.py --mode live --exchange binance --symbol BTCUSDT --duration 3600

  # Live-запись до конкретного времени:
  python recorder_binance.py --mode live --until "2026-03-22 18:00:00"

  # Исторические трейды за день:
  python recorder_binance.py --mode history --symbol BTCUSDT --start 2026-03-21 --end 2026-03-22
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


try:
    import aiohttp
except ImportError:
    raise SystemExit("Установите aiohttp: pip install aiohttp")

try:
    import websockets
except ImportError:
    raise SystemExit("Установите websockets: pip install websockets")


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_day_from_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def fmt_ts_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


def fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_remaining(elapsed_sec: float, done_frac: float) -> str:
    """Оценка оставшегося времени по доле выполнения."""
    if done_frac <= 0 or elapsed_sec <= 0:
        return "?"
    total_est = elapsed_sec / done_frac
    remaining = total_est - elapsed_sec
    return fmt_elapsed(remaining)


def parse_dt(s: str) -> datetime:
    """Парсит дату или дату+время в UTC."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Не могу распознать дату: {s!r}. Используй YYYY-MM-DD или YYYY-MM-DD HH:MM:SS")


# ---------------------------------------------------------------------------
# Файловый писатель
# ---------------------------------------------------------------------------

class JsonlGzRotatingWriter:
    """Пишет jsonl.gz, ротируя файл по UTC-дню."""

    def __init__(self, base_dir: str, prefix: str):
        self.base_dir = base_dir
        self.prefix = prefix
        self.current_day: Optional[str] = None
        self.fp = None

    def _open_for_day(self, day: str):
        os.makedirs(self.base_dir, exist_ok=True)
        path = os.path.join(self.base_dir, f"{self.prefix}_{day}.jsonl.gz")
        self.fp = gzip.open(path, mode="at", encoding="utf-8")
        self.current_day = day

    def write(self, ts_ms: int, obj: dict):
        day = utc_day_from_ms(ts_ms)
        if self.fp is None or self.current_day != day:
            self.close()
            self._open_for_day(day)
        self.fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None
            self.current_day = None


# ---------------------------------------------------------------------------
# Binance: live-запись стакана
# ---------------------------------------------------------------------------

@dataclass
class BinanceConfig:
    symbol: str
    snapshot_limit: int = 5000
    stream_speed: str = "100ms"


class BinanceDepthRecorder:
    """
    Записывает L2-стакан Binance в реальном времени:
      - snapshots/binance_SYMBOL_YYYY-MM-DD.jsonl.gz
      - diff/binance_SYMBOL_YYYY-MM-DD.jsonl.gz
      - meta/binance_SYMBOL_YYYY-MM-DD.jsonl.gz

    Печатает прогресс каждые progress_interval секунд.
    """

    PROGRESS_INTERVAL = 10  # секунд

    def __init__(self, cfg: BinanceConfig, out_dir: str):
        self.cfg = cfg
        self.symbol = cfg.symbol.upper()
        self.symbol_ws = self.symbol.lower()
        self.out_dir = out_dir

        self.snap_writer = JsonlGzRotatingWriter(
            os.path.join(out_dir, "snapshots"), f"binance_{self.symbol}"
        )
        self.diff_writer = JsonlGzRotatingWriter(
            os.path.join(out_dir, "diff"), f"binance_{self.symbol}"
        )
        self.meta_writer = JsonlGzRotatingWriter(
            os.path.join(out_dir, "meta"), f"binance_{self.symbol}"
        )

        self.local_last_update_id: Optional[int] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Статистика для прогресса
        self._diff_count: int = 0
        self._snap_count: int = 0
        self._gap_count: int = 0
        self._first_ts_ms: Optional[int] = None
        self._last_ts_ms: Optional[int] = None
        self._last_bid: Optional[str] = None
        self._last_ask: Optional[str] = None
        self._start_wall: float = time.monotonic()

    def _print_stats(self):
        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.monotonic() - self._start_wall
        period = (
            f"{fmt_ts_ms(self._first_ts_ms)} → {fmt_ts_ms(self._last_ts_ms)} UTC"
            if self._first_ts_ms and self._last_ts_ms
            else "—"
        )
        bid_ask = (
            f"bid: {self._last_bid}  ask: {self._last_ask}  "
            f"spread: {float(self._last_ask) - float(self._last_bid):.2f}"
            if self._last_bid and self._last_ask
            else "bid/ask: —"
        )
        print(
            f"[{now_str} UTC] {self.symbol} live | "
            f"diffs: {self._diff_count:,} | snapshots: {self._snap_count} | gaps: {self._gap_count}\n"
            f"  period: {period}  (elapsed: {fmt_elapsed(elapsed)})\n"
            f"  {bid_ask}",
            flush=True,
        )

    async def _stats_loop(self):
        while True:
            await asyncio.sleep(self.PROGRESS_INTERVAL)
            self._print_stats()

    async def fetch_snapshot(self) -> dict:
        url = "https://api.binance.com/api/v3/depth"
        params = {"symbol": self.symbol, "limit": self.cfg.snapshot_limit}
        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            data = await resp.json()

        ts_ms = int(time.time() * 1000)
        event = {
            "kind": "snapshot",
            "exchange": "binance",
            "symbol": self.symbol,
            "ts_capture": ts_ms,
            "lastUpdateId": data["lastUpdateId"],
            "bids": data["bids"],
            "asks": data["asks"],
        }
        self.snap_writer.write(ts_ms, event)
        self.local_last_update_id = int(data["lastUpdateId"])
        self._snap_count += 1

        # Обновляем bid/ask из снапшота
        if data["bids"]:
            self._last_bid = data["bids"][0][0]
        if data["asks"]:
            self._last_ask = data["asks"][0][0]

        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str} UTC] Snapshot #{self._snap_count}: lastUpdateId={data['lastUpdateId']}", flush=True)
        return event

    async def resync(self, reason: str):
        ts_ms = int(time.time() * 1000)
        self.meta_writer.write(ts_ms, {
            "kind": "service", "exchange": "binance", "symbol": self.symbol,
            "event": "resync_start", "reason": reason, "ts_capture": ts_ms,
        })
        await self.fetch_snapshot()
        ts_ms = int(time.time() * 1000)
        self.meta_writer.write(ts_ms, {
            "kind": "service", "exchange": "binance", "symbol": self.symbol,
            "event": "resync_done", "local_last_update_id": self.local_last_update_id,
            "ts_capture": ts_ms,
        })

    async def run(self, stop_event: Optional[asyncio.Event] = None):
        stream = f"{self.symbol_ws}@depth"
        if self.cfg.stream_speed:
            stream += f"@{self.cfg.stream_speed}"
        ws_url = f"wss://stream.binance.com:9443/ws/{stream}"

        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str} UTC] Запуск Binance live-записи: {self.symbol} → {self.out_dir}", flush=True)

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.fetch_snapshot()

            stats_task = asyncio.create_task(self._stats_loop())
            try:
                while True:
                    if stop_event and stop_event.is_set():
                        break
                    try:
                        async with websockets.connect(
                            ws_url,
                            ping_interval=20,
                            ping_timeout=20,
                            max_size=2**24,
                        ) as ws:
                            async for raw in ws:
                                if stop_event and stop_event.is_set():
                                    return

                                msg = json.loads(raw)
                                ts_ms = int(msg.get("E", int(time.time() * 1000)))
                                U = int(msg["U"])
                                u = int(msg["u"])

                                raw_event = {
                                    "kind": "diff", "exchange": "binance", "symbol": self.symbol,
                                    "ts_event": ts_ms, "ts_capture": int(time.time() * 1000),
                                    "U": U, "u": u,
                                    "b": msg.get("b", []), "a": msg.get("a", []),
                                }
                                self.diff_writer.write(ts_ms, raw_event)
                                self._diff_count += 1

                                # Трекинг времени и цен
                                if self._first_ts_ms is None:
                                    self._first_ts_ms = ts_ms
                                self._last_ts_ms = ts_ms
                                if msg.get("b"):
                                    self._last_bid = msg["b"][0][0]
                                if msg.get("a"):
                                    self._last_ask = msg["a"][0][0]

                                if self.local_last_update_id is None:
                                    await self.resync("missing_local_state")
                                    continue
                                if u <= self.local_last_update_id:
                                    continue
                                if U > self.local_last_update_id + 1:
                                    self._gap_count += 1
                                    self.meta_writer.write(ts_ms, {
                                        "kind": "service", "exchange": "binance", "symbol": self.symbol,
                                        "event": "gap_detected",
                                        "local_last_update_id": self.local_last_update_id,
                                        "incoming_U": U, "incoming_u": u,
                                        "ts_capture": int(time.time() * 1000),
                                    })
                                    await self.resync("sequence_gap")
                                    continue

                                self.local_last_update_id = u

                    except Exception as e:
                        ts_ms = int(time.time() * 1000)
                        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{now_str} UTC] WS ошибка: {e!r}, переподключение через 3с...", flush=True)
                        self.meta_writer.write(ts_ms, {
                            "kind": "service", "exchange": "binance", "symbol": self.symbol,
                            "event": "ws_error", "error": repr(e), "ts_capture": ts_ms,
                        })
                        await asyncio.sleep(3)
            finally:
                stats_task.cancel()
                self._print_stats()  # финальный отчёт

    def close(self):
        self.snap_writer.close()
        self.diff_writer.close()
        self.meta_writer.close()


# ---------------------------------------------------------------------------
# Binance: скачивание исторических трейдов
# ---------------------------------------------------------------------------

class BinanceHistoricalTradesDownloader:
    """
    Скачивает исторические агрегированные трейды через /api/v3/aggTrades.
    Binance отдаёт до 1000 трейдов за запрос; пагинация по времени.

    Файлы: <out_dir>/trades/binance_SYMBOL_YYYY-MM-DD.jsonl.gz
    """

    BATCH_SIZE = 1000
    PROGRESS_EVERY = 5_000     # печать прогресса каждые N трейдов
    RATE_LIMIT_DELAY = 0.12    # задержка между запросами (~8 req/s, лимит Binance 1200/мин)

    def __init__(self, symbol: str, out_dir: str):
        self.symbol = symbol.upper()
        self.out_dir = out_dir
        self.writer = JsonlGzRotatingWriter(
            os.path.join(out_dir, "trades"), f"binance_{self.symbol}"
        )

    async def download(self, start_dt: datetime, end_dt: datetime):
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        total_range_ms = end_ms - start_ms

        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{now_str} UTC] Скачивание трейдов {self.symbol}\n"
            f"  с:  {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"  по: {end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            flush=True,
        )

        total_count = 0
        current_start_ms = start_ms
        wall_start = time.monotonic()
        last_progress_count = 0

        async with aiohttp.ClientSession() as session:
            while current_start_ms < end_ms:
                batch = await self._fetch_batch(session, current_start_ms, end_ms)
                if not batch:
                    break

                for trade in batch:
                    ts_ms = int(trade["T"])
                    if ts_ms >= end_ms:
                        break
                    side = "buy" if trade["m"] is False else "sell"  # m=True → покупатель маркет-мейкер → сделка пришла от продавца
                    event = {
                        "kind": "trade",
                        "exchange": "binance",
                        "symbol": self.symbol,
                        "ts": ts_ms,
                        "price": trade["p"],
                        "size": trade["q"],
                        "side": side,
                        "agg_id": trade["a"],
                    }
                    self.writer.write(ts_ms, event)
                    total_count += 1
                    current_start_ms = ts_ms

                current_start_ms += 1  # следующий запрос с ts+1 чтобы не дублировать

                # Прогресс
                if total_count - last_progress_count >= self.PROGRESS_EVERY:
                    last_progress_count = total_count
                    elapsed = time.monotonic() - wall_start
                    done_frac = (current_start_ms - start_ms) / max(total_range_ms, 1)
                    current_dt = datetime.fromtimestamp(current_start_ms / 1000, tz=timezone.utc)
                    remaining = fmt_remaining(elapsed, done_frac)
                    now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"[{now_str} UTC] трейдов: {total_count:,} | "
                        f"текущий момент: {current_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                        f"прогресс: {done_frac*100:.1f}% | осталось: ~{remaining}",
                        flush=True,
                    )

                await asyncio.sleep(self.RATE_LIMIT_DELAY)

        elapsed = time.monotonic() - wall_start
        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{now_str} UTC] Готово! Скачано {total_count:,} трейдов за {fmt_elapsed(elapsed)}",
            flush=True,
        )
        self.writer.close()

    async def _fetch_batch(self, session: aiohttp.ClientSession, start_ms: int, end_ms: int) -> list:
        url = "https://api.binance.com/api/v3/aggTrades"
        params = {
            "symbol": self.symbol,
            "startTime": start_ms,
            "endTime": min(end_ms - 1, start_ms + 3_600_000),  # не более 1 часа за раз
            "limit": self.BATCH_SIZE,
        }
        for attempt in range(5):
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", "60"))
                        print(f"  Rate limit! Жду {retry_after}с...", flush=True)
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                wait = 2 ** attempt
                print(f"  Ошибка запроса (попытка {attempt+1}/5): {e!r}, жду {wait}с...", flush=True)
                await asyncio.sleep(wait)
        raise RuntimeError(f"Не удалось получить трейды для startTime={start_ms}")


# ---------------------------------------------------------------------------
# Bybit: live-запись (без изменений, только добавлен прогресс-вывод)
# ---------------------------------------------------------------------------

@dataclass
class BybitConfig:
    category: str   # "spot" | "linear" | "inverse"
    symbol: str
    level: int = 50


class BybitDepthRecorder:
    """Записывает стакан Bybit в реальном времени."""

    PROGRESS_INTERVAL = 10

    def __init__(self, cfg: BybitConfig, out_dir: str):
        self.cfg = cfg
        self.symbol = cfg.symbol.upper()
        self.out_dir = out_dir

        self.snap_writer = JsonlGzRotatingWriter(
            os.path.join(out_dir, "snapshots"), f"bybit_{cfg.category}_{self.symbol}"
        )
        self.delta_writer = JsonlGzRotatingWriter(
            os.path.join(out_dir, "delta"), f"bybit_{cfg.category}_{self.symbol}"
        )
        self.meta_writer = JsonlGzRotatingWriter(
            os.path.join(out_dir, "meta"), f"bybit_{cfg.category}_{self.symbol}"
        )

        self.session: Optional[aiohttp.ClientSession] = None
        self._delta_count: int = 0
        self._snap_count: int = 0
        self._first_ts_ms: Optional[int] = None
        self._last_ts_ms: Optional[int] = None
        self._start_wall: float = time.monotonic()

    def _print_stats(self):
        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.monotonic() - self._start_wall
        period = (
            f"{fmt_ts_ms(self._first_ts_ms)} → {fmt_ts_ms(self._last_ts_ms)} UTC"
            if self._first_ts_ms and self._last_ts_ms else "—"
        )
        print(
            f"[{now_str} UTC] {self.symbol} live (bybit) | "
            f"deltas: {self._delta_count:,} | snapshots: {self._snap_count}\n"
            f"  period: {period}  (elapsed: {fmt_elapsed(elapsed)})",
            flush=True,
        )

    async def _stats_loop(self):
        while True:
            await asyncio.sleep(self.PROGRESS_INTERVAL)
            self._print_stats()

    async def fetch_snapshot(self):
        url = "https://api.bybit.com/v5/market/orderbook"
        params = {
            "category": self.cfg.category,
            "symbol": self.symbol,
            "limit": min(self.cfg.level, 200 if self.cfg.category == "spot" else 500),
        }
        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            resp.raise_for_status()
            data = await resp.json()

        result = data["result"]
        ts_ms = int(result.get("ts", int(time.time() * 1000)))
        event = {
            "kind": "snapshot", "exchange": "bybit",
            "category": self.cfg.category, "symbol": self.symbol,
            "ts_event": ts_ms, "ts_capture": int(time.time() * 1000),
            "u": result.get("u"), "seq": result.get("seq"),
            "b": result.get("b", []), "a": result.get("a", []),
        }
        self.snap_writer.write(ts_ms, event)
        self._snap_count += 1
        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str} UTC] Bybit snapshot #{self._snap_count}", flush=True)
        return event

    async def run(self, stop_event: Optional[asyncio.Event] = None):
        ws_host = {
            "spot": "wss://stream.bybit.com/v5/public/spot",
            "linear": "wss://stream.bybit.com/v5/public/linear",
            "inverse": "wss://stream.bybit.com/v5/public/inverse",
        }[self.cfg.category]
        topic = f"orderbook.{self.cfg.level}.{self.symbol}"

        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str} UTC] Запуск Bybit live-записи: {self.symbol} ({self.cfg.category}) → {self.out_dir}", flush=True)

        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.fetch_snapshot()
            stats_task = asyncio.create_task(self._stats_loop())
            try:
                while True:
                    if stop_event and stop_event.is_set():
                        break
                    try:
                        async with websockets.connect(
                            ws_host, ping_interval=20, ping_timeout=20, max_size=2**24,
                        ) as ws:
                            await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
                            async for raw in ws:
                                if stop_event and stop_event.is_set():
                                    return
                                msg = json.loads(raw)
                                if "topic" not in msg or msg.get("topic") != topic:
                                    continue

                                data = msg["data"]
                                msg_type = msg.get("type")
                                ts_ms = int(msg.get("ts", int(time.time() * 1000)))

                                if self._first_ts_ms is None:
                                    self._first_ts_ms = ts_ms
                                self._last_ts_ms = ts_ms

                                if msg_type == "snapshot":
                                    event = {
                                        "kind": "snapshot", "exchange": "bybit",
                                        "category": self.cfg.category, "symbol": self.symbol,
                                        "ts_event": ts_ms, "ts_capture": int(time.time() * 1000),
                                        "u": data.get("u"), "seq": data.get("seq"),
                                        "b": data.get("b", []), "a": data.get("a", []),
                                    }
                                    self.snap_writer.write(ts_ms, event)
                                    self._snap_count += 1
                                elif msg_type == "delta":
                                    event = {
                                        "kind": "delta", "exchange": "bybit",
                                        "category": self.cfg.category, "symbol": self.symbol,
                                        "ts_event": ts_ms, "ts_capture": int(time.time() * 1000),
                                        "u": data.get("u"), "seq": data.get("seq"),
                                        "b": data.get("b", []), "a": data.get("a", []),
                                    }
                                    self.delta_writer.write(ts_ms, event)
                                    self._delta_count += 1

                    except Exception as e:
                        ts_ms = int(time.time() * 1000)
                        now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{now_str} UTC] WS ошибка: {e!r}, переподключение через 3с...", flush=True)
                        self.meta_writer.write(ts_ms, {
                            "kind": "service", "exchange": "bybit",
                            "category": self.cfg.category, "symbol": self.symbol,
                            "event": "ws_error", "error": repr(e), "ts_capture": ts_ms,
                        })
                        await asyncio.sleep(3)
            finally:
                stats_task.cancel()
                self._print_stats()

    def close(self):
        self.snap_writer.close()
        self.delta_writer.close()
        self.meta_writer.close()


# ---------------------------------------------------------------------------
# Вспомогательная функция: авто-стоп по времени
# ---------------------------------------------------------------------------

async def _stop_after(seconds: float, stop_event: asyncio.Event):
    await asyncio.sleep(seconds)
    stop_event.set()


async def _stop_at(target: datetime, stop_event: asyncio.Event):
    now = utc_now()
    delay = (target - now).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)
    stop_event.set()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace):
    if args.exchange != "binance":
        raise SystemExit("Use recorder_bybit.py for Bybit recording")
    if args.mode == "history":
        # Исторические трейды (только Binance)
        if args.exchange != "binance":
            raise SystemExit("--mode history поддерживается только для --exchange binance")
        if not args.start or not args.end:
            raise SystemExit("--mode history требует --start и --end")

        start_dt = parse_dt(args.start)
        end_dt = parse_dt(args.end)
        if end_dt <= start_dt:
            raise SystemExit("--end должен быть позже --start")

        downloader = BinanceHistoricalTradesDownloader(
            symbol=args.symbol,
            out_dir=args.out_dir,
        )
        await downloader.download(start_dt, end_dt)
        return

    # Live-режим
    stop_event = asyncio.Event()

    # Настройка авто-стопа
    stop_tasks = []
    if args.duration and args.duration > 0:
        stop_tasks.append(asyncio.create_task(_stop_after(args.duration, stop_event)))
        print(f"Авто-стоп через {fmt_elapsed(args.duration)}", flush=True)
    if args.until:
        target_dt = parse_dt(args.until)
        stop_tasks.append(asyncio.create_task(_stop_at(target_dt, stop_event)))
        print(f"Авто-стоп в {target_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)

    try:
        cfg = BinanceConfig(
            symbol=args.symbol,
            snapshot_limit=5000,
            stream_speed="100ms",
        )
        rec = BinanceDepthRecorder(cfg, out_dir=args.out_dir)
        try:
            await rec.run(stop_event=stop_event)
        finally:
            rec.close()
    finally:
        for t in stop_tasks:
            t.cancel()


def main():
    parser = argparse.ArgumentParser(
        description="Запись/скачивание рыночных данных с Binance и Bybit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Binance live 1 час:
  python recorder_binance.py --duration 3600

  # Binance live до конкретного времени:
  python recorder_binance.py --until "2026-03-22 18:00:00"

  # Исторические трейды за день:
  python recorder_binance.py --mode history --start 2026-03-21 --end 2026-03-22

  # Bybit live 30 минут:
  python recorder_binance.py --exchange bybit --duration 1800
""",
    )
    parser.add_argument("--exchange", default="binance", choices=["binance", "bybit"],
                        help="Биржа (default: binance)")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Тикер (default: BTCUSDT)")
    parser.add_argument("--out-dir", default="./data",
                        help="Директория для записи (default: ./data)")
    parser.add_argument("--mode", default="live", choices=["live", "history"],
                        help="live — запись стакана; history — скачать трейды за период (default: live)")

    live_group = parser.add_argument_group("live-режим")
    live_group.add_argument("--duration", type=float, default=0,
                            help="Остановиться через N секунд (0 = бесконечно)")
    live_group.add_argument("--until", default=None,
                            help='Остановиться в это UTC-время, напр. "2026-03-22 18:00:00"')

    hist_group = parser.add_argument_group("history-режим (только Binance, только трейды)")
    hist_group.add_argument("--start", default=None,
                            help='Начало периода, напр. "2026-03-21" или "2026-03-21 06:00:00"')
    hist_group.add_argument("--end", default=None,
                            help='Конец периода, напр. "2026-03-22"')

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
