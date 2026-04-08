from __future__ import annotations

from datetime import datetime

import pytest

from src.validation import CSVSnapshotLoaderConfig, SnapshotValidationError, parse_snapshot_row


def test_missing_time_raises_error() -> None:
    row = {"ask_price_1": "100.0", "bid_price_1": "99.0", "ask_size_1": "1", "bid_size_1": "1"}
    with pytest.raises(SnapshotValidationError, match="missing required value"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_missing_price_column_value_raises_error() -> None:
    row = {"time": "2024-01-01T10:00:00", "ask_price_1": "", "bid_price_1": "99.0", "ask_size_1": "1", "bid_size_1": "1"}
    with pytest.raises(SnapshotValidationError, match="empty required value"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_missing_size_column_value_raises_error() -> None:
    row = {"time": "2024-01-01T10:00:00", "ask_price_1": "100.0", "bid_price_1": "99.0", "ask_size_1": "", "bid_size_1": "1"}
    with pytest.raises(SnapshotValidationError, match="empty required value"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_invalid_timestamp_raises_error() -> None:
    row = {"time": "bad", "ask_price_1": "100.0", "bid_price_1": "99.0", "ask_size_1": "1", "bid_size_1": "1"}
    with pytest.raises(SnapshotValidationError, match="invalid timestamp"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_unix_timestamp_is_accepted_with_warning() -> None:
    row = {
        "time": "1772612553.592",
        "ask_price_1": "100.0",
        "bid_price_1": "99.0",
        "ask_size_1": "1",
        "bid_size_1": "1",
    }

    with pytest.warns(RuntimeWarning, match="interpreting value as Unix timestamp"):
        snapshot = parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)

    assert snapshot.timestamp == datetime(2026, 3, 4, 8, 22, 33, 592000)


def test_negative_size_raises_error() -> None:
    row = {"time": "2024-01-01T10:00:00", "ask_price_1": "100.0", "bid_price_1": "99.0", "ask_size_1": "-1", "bid_size_1": "1"}
    with pytest.raises(SnapshotValidationError, match="negative size"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_non_positive_price_raises_error() -> None:
    row = {"time": "2024-01-01T10:00:00", "ask_price_1": "0", "bid_price_1": "99.0", "ask_size_1": "1", "bid_size_1": "1"}
    with pytest.raises(SnapshotValidationError, match="non-positive price"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1), row_number=2)


def test_asks_out_of_order_raise_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "101.0",
        "bid_price_1": "99.0",
        "ask_size_1": "1",
        "bid_size_1": "1",
        "ask_price_2": "100.0",
        "bid_price_2": "98.0",
        "ask_size_2": "1",
        "bid_size_2": "1",
    }
    with pytest.raises(SnapshotValidationError, match="asks are not sorted ascending"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=2), row_number=2)


def test_bids_out_of_order_raise_error() -> None:
    row = {
        "time": "2024-01-01T10:00:00",
        "ask_price_1": "101.0",
        "bid_price_1": "98.0",
        "ask_size_1": "1",
        "bid_size_1": "1",
        "ask_price_2": "102.0",
        "bid_price_2": "99.0",
        "ask_size_2": "1",
        "bid_size_2": "1",
    }
    with pytest.raises(SnapshotValidationError, match="bids are not sorted descending"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=2), row_number=2)


def test_crossed_book_raises_with_validation_enabled() -> None:
    row = {"time": "2024-01-01T10:00:00", "ask_price_1": "100.0", "bid_price_1": "100.0", "ask_size_1": "1", "bid_size_1": "1"}
    with pytest.raises(SnapshotValidationError, match="crossed book detected"):
        parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1, validate_crossed_book=True), row_number=2)


def test_crossed_book_allowed_when_validation_disabled() -> None:
    row = {"time": "2024-01-01T10:00:00", "ask_price_1": "100.0", "bid_price_1": "100.0", "ask_size_1": "1", "bid_size_1": "1"}
    snapshot = parse_snapshot_row(row, CSVSnapshotLoaderConfig(depth=1, validate_crossed_book=False), row_number=2)
    assert snapshot.best_bid().price == 100.0
