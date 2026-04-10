from __future__ import annotations

import pytest

from run_simulation import infer_csv_depth, main


def test_run_simulation_with_rsi_strategy_prints_summary(tmp_path, capsys) -> None:
    csv_path = tmp_path / "book.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,101,100,1,1\n"
        "2024-01-01T10:00:01,100,99,1,1\n"
        "2024-01-01T10:00:02,99,98,1,1\n"
        "2024-01-01T10:00:03,98,97,1,1\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "--strategy",
            "rsi_mean_reversion",
            "--rsi-period",
            "3",
            "--oversold",
            "35",
            "--qty",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "strategy=RSIMeanReversionStrategy" in captured.out


def test_run_simulation_with_moving_average_strategy_prints_summary(tmp_path, capsys) -> None:
    csv_path = tmp_path / "book.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,100,99,1,1\n"
        "2024-01-01T10:00:01,100,99,1,1\n"
        "2024-01-01T10:00:02,103,102,1,1\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "--strategy",
            "moving_average_cross",
            "--short-window",
            "2",
            "--long-window",
            "3",
            "--qty",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "strategy=MovingAverageCrossStrategy" in captured.out


def test_run_simulation_with_rsi_limit_timeout_strategy_prints_summary(tmp_path, capsys) -> None:
    csv_path = tmp_path / "book.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,101,100,1,1\n"
        "2024-01-01T10:00:01,100,99,1,1\n"
        "2024-01-01T10:00:02,99,98,1,1\n"
        "2024-01-01T10:00:03,98,97,1,1\n"
        "2024-01-01T10:00:09,103,102,1,1\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "--strategy",
            "rsi_limit_order_timeout",
            "--rsi-period",
            "3",
            "--oversold",
            "35",
            "--qty",
            "1",
            "--order-ttl-seconds",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "strategy=RSILimitOrderTimeoutStrategy" in captured.out


def test_run_simulation_with_rsi_limit_template_strategy_prints_summary(tmp_path, capsys) -> None:
    csv_path = tmp_path / "book.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,101,100,1,1\n"
        "2024-01-01T10:00:01,100,99,1,1\n"
        "2024-01-01T10:00:02,99,98,1,1\n"
        "2024-01-01T10:00:03,98,97,1,1\n"
        "2024-01-01T10:00:09,103,102,1,1\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "--strategy",
            "rsi_limit_order_template",
            "--rsi-period",
            "3",
            "--oversold",
            "35",
            "--qty",
            "1",
            "--order-ttl-seconds",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "strategy=RSILimitOrderTemplateStrategy" in captured.out


def test_run_simulation_can_export_execution_report_files(tmp_path, capsys) -> None:
    csv_path = tmp_path / "book.csv"
    json_path = tmp_path / "execution_report.json"
    orders_csv_path = tmp_path / "execution_orders.csv"
    fills_csv_path = tmp_path / "execution_fills.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,100,99,1,1\n"
        "2024-01-01T10:00:01,100,99,1,1\n"
        "2024-01-01T10:00:02,103,102,1,1\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--csv",
            str(csv_path),
            "--strategy",
            "moving_average_cross",
            "--short-window",
            "2",
            "--long-window",
            "3",
            "--qty",
            "1",
            "--execution-report-json",
            str(json_path),
            "--execution-orders-csv",
            str(orders_csv_path),
            "--execution-fills-csv",
            str(fills_csv_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "execution_report:" in captured.out
    assert json_path.exists()
    assert orders_csv_path.exists()
    assert fills_csv_path.exists()


def test_infer_csv_depth_uses_max_complete_level(tmp_path) -> None:
    csv_path = tmp_path / "book_depth_2.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1,"
        "ask_price_2,bid_price_2,ask_size_2,bid_size_2,"
        "ask_price_3,bid_price_3\n"
        "2024-01-01T10:00:00,101,100,1,1,102,99,2,2,103,98\n",
        encoding="utf-8",
    )

    assert infer_csv_depth(csv_path) == 2


def test_infer_csv_depth_raises_when_header_has_no_complete_book_level(tmp_path) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(
        "time,ask_price_1,bid_price_1\n"
        "2024-01-01T10:00:00,101,100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not infer a valid order book depth"):
        infer_csv_depth(csv_path)
