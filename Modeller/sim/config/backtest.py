from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("backtest_config.json")
STRATEGY_NAMES = ("mm", "taker", "ema", "imbalance")


@dataclass
class DataConfig:
    l2_path: str = "test_data/test.csv"
    trades_path: str | None = None
    symbol: str | None = None


@dataclass
class MMConfig:
    spread: float = 1.0
    quote_size: float = 1.0


@dataclass
class TakerConfig:
    window: int = 20
    std_mult: float = 2.0
    qty: float = 1.0
    cooldown: float = 0.0
    max_position: float = 1.0


@dataclass
class EmaConfig:
    fast: int = 10
    slow: int = 30
    qty: float = 1.0
    max_position: float = 1.0
    offset: float = 0.0


@dataclass
class ImbalanceConfig:
    depth: int = 5
    threshold: float = 0.3
    smoothing: int = 3
    qty: float = 1.0
    max_position: float = 1.0


@dataclass
class StrategyConfig:
    name: str = "mm"
    mm: MMConfig = field(default_factory=MMConfig)
    taker: TakerConfig = field(default_factory=TakerConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 860
    debug: bool = False


@dataclass
class ConsoleConfig:
    level: int = 1
    book_levels: int = 1


@dataclass
class BacktestConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)


def _ensure_mapping(raw: Any, context: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a JSON object")
    return raw


def _validate_keys(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {context}: {', '.join(unknown)}")


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_data_config(raw: Any) -> DataConfig:
    data = _ensure_mapping(raw, "data")
    _validate_keys(data, {"l2_path", "trades_path", "symbol"}, "data")
    return DataConfig(
        l2_path=str(data.get("l2_path", DataConfig.l2_path)),
        trades_path=_opt_str(data.get("trades_path", DataConfig.trades_path)),
        symbol=_opt_str(data.get("symbol", DataConfig.symbol)),
    )


def _parse_mm_config(raw: Any) -> MMConfig:
    data = _ensure_mapping(raw, "strategy.mm")
    _validate_keys(data, {"spread", "quote_size"}, "strategy.mm")
    return MMConfig(
        spread=float(data.get("spread", MMConfig.spread)),
        quote_size=float(data.get("quote_size", MMConfig.quote_size)),
    )


def _parse_taker_config(raw: Any) -> TakerConfig:
    data = _ensure_mapping(raw, "strategy.taker")
    _validate_keys(
        data,
        {"window", "std_mult", "qty", "cooldown", "max_position"},
        "strategy.taker",
    )
    return TakerConfig(
        window=int(data.get("window", TakerConfig.window)),
        std_mult=float(data.get("std_mult", TakerConfig.std_mult)),
        qty=float(data.get("qty", TakerConfig.qty)),
        cooldown=float(data.get("cooldown", TakerConfig.cooldown)),
        max_position=float(data.get("max_position", TakerConfig.max_position)),
    )


def _parse_ema_config(raw: Any) -> EmaConfig:
    data = _ensure_mapping(raw, "strategy.ema")
    _validate_keys(data, {"fast", "slow", "qty", "max_position", "offset"}, "strategy.ema")
    return EmaConfig(
        fast=int(data.get("fast", EmaConfig.fast)),
        slow=int(data.get("slow", EmaConfig.slow)),
        qty=float(data.get("qty", EmaConfig.qty)),
        max_position=float(data.get("max_position", EmaConfig.max_position)),
        offset=float(data.get("offset", EmaConfig.offset)),
    )


def _parse_imbalance_config(raw: Any) -> ImbalanceConfig:
    data = _ensure_mapping(raw, "strategy.imbalance")
    _validate_keys(
        data,
        {"depth", "threshold", "smoothing", "qty", "max_position"},
        "strategy.imbalance",
    )
    return ImbalanceConfig(
        depth=int(data.get("depth", ImbalanceConfig.depth)),
        threshold=float(data.get("threshold", ImbalanceConfig.threshold)),
        smoothing=int(data.get("smoothing", ImbalanceConfig.smoothing)),
        qty=float(data.get("qty", ImbalanceConfig.qty)),
        max_position=float(data.get("max_position", ImbalanceConfig.max_position)),
    )


def _parse_strategy_config(raw: Any) -> StrategyConfig:
    data = _ensure_mapping(raw, "strategy")
    _validate_keys(data, {"name", "mm", "taker", "ema", "imbalance"}, "strategy")
    name = str(data.get("name", StrategyConfig.name))
    if name not in STRATEGY_NAMES:
        raise ValueError(
            f"strategy.name must be one of {', '.join(STRATEGY_NAMES)}, got {name!r}"
        )
    return StrategyConfig(
        name=name,
        mm=_parse_mm_config(data.get("mm")),
        taker=_parse_taker_config(data.get("taker")),
        ema=_parse_ema_config(data.get("ema")),
        imbalance=_parse_imbalance_config(data.get("imbalance")),
    )


def _parse_dashboard_config(raw: Any) -> DashboardConfig:
    data = _ensure_mapping(raw, "dashboard")
    _validate_keys(data, {"host", "port", "debug"}, "dashboard")
    return DashboardConfig(
        host=str(data.get("host", DashboardConfig.host)),
        port=int(data.get("port", DashboardConfig.port)),
        debug=bool(data.get("debug", DashboardConfig.debug)),
    )


def _parse_console_config(raw: Any) -> ConsoleConfig:
    data = _ensure_mapping(raw, "console")
    _validate_keys(data, {"level", "book_levels"}, "console")
    level = int(data.get("level", ConsoleConfig.level))
    book_levels = int(data.get("book_levels", ConsoleConfig.book_levels))
    if level < 1:
        raise ValueError("console.level must be >= 1")
    if book_levels < 1:
        raise ValueError("console.book_levels must be >= 1")
    return ConsoleConfig(level=level, book_levels=book_levels)


def backtest_config_from_dict(raw: Any) -> BacktestConfig:
    data = _ensure_mapping(raw, "root config")
    _validate_keys(data, {"data", "strategy", "dashboard", "console"}, "root config")
    return BacktestConfig(
        data=_parse_data_config(data.get("data")),
        strategy=_parse_strategy_config(data.get("strategy")),
        dashboard=_parse_dashboard_config(data.get("dashboard")),
        console=_parse_console_config(data.get("console")),
    )


def load_backtest_config(path: str | Path | None = None) -> BacktestConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if path is None and not config_path.exists():
        return BacktestConfig()
    if not config_path.exists():
        raise FileNotFoundError(f"Backtest config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return backtest_config_from_dict(payload)


def backtest_config_to_dict(config: BacktestConfig) -> dict[str, Any]:
    return asdict(config)
