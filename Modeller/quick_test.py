from run_backtest import run, _build_strategy
from sim.config import EmaConfig, ImbalanceConfig, MMConfig, StrategyConfig, TakerConfig
from sim.data.loaders import load_l2_binance

snapshots = list(load_l2_binance(data_dir="data", symbol="BTCUSDT", depth=25, start_ts=None, end_ts=None))
print(f"Loaded {len(snapshots)} snapshots")

strategy_config = StrategyConfig(
    name="mm",
    mm=MMConfig(spread=1.0, quote_size=1.0),
    taker=TakerConfig(window=20, std_mult=2.0, qty=1.0, cooldown=0.0, max_position=1.0),
    ema=EmaConfig(fast=10, slow=30, qty=1.0, max_position=1.0, offset=0.0),
    imbalance=ImbalanceConfig(depth=5, threshold=0.3, smoothing=3, qty=1.0, max_position=1.0),
)

for name in ["mm", "taker", "ema", "imbalance"]:
    strategy_config.name = name
    strategy = _build_strategy(strategy_config)
    metrics = run(snapshots=snapshots, strategy=strategy)
    print(f"{name}: fills={metrics.num_fills}, equity_curve_len={len(metrics.equity_curve) if metrics.equity_curve else 0}")
