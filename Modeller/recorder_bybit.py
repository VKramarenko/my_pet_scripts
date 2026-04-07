from __future__ import annotations

import argparse
import asyncio

from recorder_binance import (
    BybitConfig,
    BybitDepthRecorder,
    _stop_after,
    _stop_at,
    fmt_elapsed,
    parse_dt,
)


async def async_main(args: argparse.Namespace) -> None:
    stop_event = asyncio.Event()
    stop_tasks: list[asyncio.Task] = []

    if args.duration and args.duration > 0:
        stop_tasks.append(asyncio.create_task(_stop_after(args.duration, stop_event)))
        print(f"Авто-стоп через {fmt_elapsed(args.duration)}", flush=True)
    if args.until:
        target_dt = parse_dt(args.until)
        stop_tasks.append(asyncio.create_task(_stop_at(target_dt, stop_event)))
        print(f"Авто-стоп в {target_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)

    try:
        cfg = BybitConfig(category=args.category, symbol=args.symbol, level=args.level)
        rec = BybitDepthRecorder(cfg, out_dir=args.out_dir)
        try:
            await rec.run(stop_event=stop_event)
        finally:
            rec.close()
    finally:
        for task in stop_tasks:
            task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Онлайн-запись стакана Bybit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python recorder_bybit.py --duration 1800
  python recorder_bybit.py --symbol ETHUSDT --category spot --level 200
  python recorder_bybit.py --until "2026-03-22 18:00:00"
""",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Тикер (default: BTCUSDT)")
    parser.add_argument("--category", default="spot", choices=["spot", "linear", "inverse"], help="Категория рынка")
    parser.add_argument("--level", type=int, default=200, help="Глубина стакана для подписки (default: 200)")
    parser.add_argument("--out-dir", default="./data", help="Директория для записи (default: ./data)")
    parser.add_argument("--duration", type=float, default=0.0, help="Остановиться через N секунд (0 = бесконечно)")
    parser.add_argument("--until", default=None, help='Остановиться в это UTC-время, напр. "2026-03-22 18:00:00"')
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
