from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.data_loader import iter_snapshots_csv, load_snapshots_csv
from src.validation import (
    CSVFormatError,
    CSVSnapshotLoaderConfig,
    SnapshotValidationError,
    build_required_columns,
    parse_snapshot_row,
)


def write_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "snapshots.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_build_required_columns_depth_2() -> None:
    assert build_required_columns(2) == [
        "time",
        "ask_price_1",
        "bid_price_1",
        "ask_size_1",
        "bid_size_1",
        "ask_price_2",
        "bid_price_2",
        "ask_size_2",
        "bid_size_2",
    ]


def test_load_snapshots_csv_parses_depth_1(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,100.5,100.0,12,10\n",
    )

    snapshots = load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1))

    assert len(snapshots) == 1
    assert snapshots[0].best_ask().price == 100.5
    assert snapshots[0].best_bid().price == 100.0


def test_load_snapshots_csv_parses_depth_2_and_mid_price(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1,ask_price_2,bid_price_2,ask_size_2,bid_size_2\n"
        "2024-01-01T10:00:00,100.5,100.0,12,10,101.0,99.5,20,15\n"
        "2024-01-01T10:00:01,100.6,100.1,8,14,101.2,99.9,11,25\n",
    )

    snapshots = load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=2))

    assert len(snapshots) == 2
    assert snapshots[0].mid_price() == 100.25
    assert snapshots[1].best_bid().price == 100.1


def test_iter_snapshots_csv_returns_ordered_sequence(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,100.5,100.0,12,10\n"
        "2024-01-01T10:00:01,100.6,100.1,11,9\n",
    )

    snapshots = list(iter_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1)))

    assert [snapshot.timestamp for snapshot in snapshots] == [
        datetime.fromisoformat("2024-01-01T10:00:00"),
        datetime.fromisoformat("2024-01-01T10:00:01"),
    ]


def test_missing_time_column_raises_error(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "ask_price_1,bid_price_1,ask_size_1,bid_size_1\n100.5,100.0,12,10\n")

    with pytest.raises(CSVFormatError, match="missing required column: time"):
        load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1))


def test_missing_level_column_raises_error(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "time,bid_price_1,ask_size_1,bid_size_1\n2024-01-01T10:00:00,100.0,12,10\n")

    with pytest.raises(CSVFormatError, match="missing required column: ask_price_1"):
        load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1))


def test_missing_deeper_level_column_raises_error(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1,ask_price_2,bid_price_2,ask_size_2\n"
        "2024-01-01T10:00:00,100.5,100.0,12,10,101.0,99.5,20\n",
    )

    with pytest.raises(CSVFormatError, match="missing required column: bid_size_2"):
        load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=2))


def test_invalid_timestamp_raises_error() -> None:
    row = {
        "time": "not-a-time",
        "ask_price_1": "100.5",
        "bid_price_1": "100.0",
        "ask_size_1": "12",
        "bid_size_1": "10",
    }

    with pytest.raises(SnapshotValidationError, match="invalid timestamp"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_load_snapshots_csv_accepts_unix_timestamps_with_warning(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "1772612553.592,100.5,100.0,12,10\n"
        "1772612554.592,100.6,100.1,11,9\n",
    )

    with pytest.warns(RuntimeWarning, match="interpreting value as Unix timestamp"):
        snapshots = load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1))

    assert len(snapshots) == 2


def test_negative_ask_size_raises_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.5",
        "bid_price_1": "100.0",
        "ask_size_1": "-1",
        "bid_size_1": "10",
    }

    with pytest.raises(SnapshotValidationError, match="negative size"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_negative_bid_size_raises_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.5",
        "bid_price_1": "100.0",
        "ask_size_1": "1",
        "bid_size_1": "-10",
    }

    with pytest.raises(SnapshotValidationError, match="negative size"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_non_positive_price_raises_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "0",
        "bid_price_1": "100.0",
        "ask_size_1": "1",
        "bid_size_1": "10",
    }

    with pytest.raises(SnapshotValidationError, match="non-positive price"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_empty_required_value_raises_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "",
        "bid_price_1": "100.0",
        "ask_size_1": "1",
        "bid_size_1": "10",
    }

    with pytest.raises(SnapshotValidationError, match="empty required value"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_non_numeric_price_raises_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "abc",
        "bid_price_1": "100.0",
        "ask_size_1": "1",
        "bid_size_1": "10",
    }

    with pytest.raises(SnapshotValidationError, match="value is not a valid float"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_non_numeric_size_raises_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.5",
        "bid_price_1": "100.0",
        "ask_size_1": "abc",
        "bid_size_1": "10",
    }

    with pytest.raises(SnapshotValidationError, match="value is not a valid float"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_unsorted_asks_raise_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "101.0",
        "bid_price_1": "100.0",
        "ask_size_1": "1",
        "bid_size_1": "10",
        "ask_price_2": "100.5",
        "bid_price_2": "99.5",
        "ask_size_2": "2",
        "bid_size_2": "9",
    }

    with pytest.raises(SnapshotValidationError, match="asks are not sorted ascending"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=2), row_number=2)


def test_unsorted_bids_raise_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.5",
        "bid_price_1": "99.0",
        "ask_size_1": "1",
        "bid_size_1": "10",
        "ask_price_2": "101.0",
        "bid_price_2": "99.5",
        "ask_size_2": "2",
        "bid_size_2": "9",
    }

    with pytest.raises(SnapshotValidationError, match="bids are not sorted descending"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=2), row_number=2)


def test_crossed_book_raises_error_when_enabled() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.0",
        "bid_price_1": "100.0",
        "ask_size_1": "1",
        "bid_size_1": "10",
    }

    with pytest.raises(SnapshotValidationError, match="crossed book detected"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_crossed_book_is_allowed_when_disabled() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.0",
        "bid_price_1": "100.0",
        "ask_size_1": "1",
        "bid_size_1": "10",
    }

    snapshot = parse_snapshot_row(
        row,
        CSVSnapshotLoaderConfig(depth=1, validate_crossed_book=False),
        row_number=2,
    )

    assert snapshot.best_ask().price == 100.0
    assert snapshot.best_bid().price == 100.0


def test_strictly_increasing_timestamps_pass(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,100.5,100.0,12,10\n"
        "2024-01-01T10:00:01,100.6,100.1,11,9\n",
    )

    snapshots = load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1))

    assert len(snapshots) == 2


def test_equal_timestamps_raise_in_strict_mode(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,100.5,100.0,12,10\n"
        "2024-01-01T10:00:00,100.6,100.1,11,9\n",
    )

    with pytest.raises(SnapshotValidationError, match="duplicate or non-increasing timestamps"):
        load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1, enforce_strict_time_increase=True))


def test_equal_timestamps_allowed_in_non_strict_mode(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:00,100.5,100.0,12,10\n"
        "2024-01-01T10:00:00,100.6,100.1,11,9\n",
    )

    snapshots = load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1, enforce_strict_time_increase=False))

    assert len(snapshots) == 2


def test_decreasing_timestamp_raises_error(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "time,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "2024-01-01T10:00:01,100.5,100.0,12,10\n"
        "2024-01-01T10:00:00,100.6,100.1,11,9\n",
    )

    with pytest.raises(SnapshotValidationError, match="timestamp"):
        load_snapshots_csv(path, CSVSnapshotLoaderConfig(depth=1, enforce_strict_time_increase=False))


def test_zero_sizes_allowed_when_enabled() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.5",
        "bid_price_1": "100.0",
        "ask_size_1": "0",
        "bid_size_1": "0",
    }

    snapshot = parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1, allow_zero_sizes=True), row_number=2)

    assert snapshot.asks[0].qty == 0.0
    assert snapshot.bids[0].qty == 0.0


def test_zero_sizes_rejected_when_disabled() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "100.5",
        "bid_price_1": "100.0",
        "ask_size_1": "0",
        "bid_size_1": "1",
    }

    with pytest.raises(SnapshotValidationError, match="zero size is not allowed"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1, allow_zero_sizes=False), row_number=2)
