from __future__ import annotations

from pathlib import Path

from run_backtest import run
from sim.visualization.dash_app import _to_equity_points


def _write_l2_csv(path: Path) -> None:
    path.write_text(
        "time,symbol,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "1.0,TEST,101.0,100.0,2.0,2.0\n"
        "2.0,TEST,102.0,101.0,2.0,2.0\n",
        encoding="utf-8",
    )


def test_dash_adapter_points_from_metrics(tmp_path: Path) -> None:
    l2_path = tmp_path / "l2.csv"
    _write_l2_csv(l2_path)

    metrics = run(str(l2_path), trades_path=None)
    points = _to_equity_points(metrics)

    assert len(points) == len(metrics.equity_curve)
    if points:
        assert set(points[0]) == {"ts", "equity"}
