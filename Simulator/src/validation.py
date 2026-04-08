from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
import warnings

from src.models import Level, Snapshot


class CSVFormatError(ValueError):
    """Raised when CSV schema or structure is invalid."""

    def __init__(
        self,
        message: str,
        *,
        row_number: int | None = None,
        column: str | None = None,
    ) -> None:
        self.row_number = row_number
        self.column = column
        super().__init__(self._compose_message(message))

    def _compose_message(self, message: str) -> str:
        parts = [message]
        if self.row_number is not None:
            parts.append(f"row={self.row_number}")
        if self.column is not None:
            parts.append(f"column={self.column}")
        return " | ".join(parts)


class SnapshotValidationError(ValueError):
    """Raised when a CSV row cannot be converted to a valid snapshot."""

    def __init__(
        self,
        message: str,
        *,
        row_number: int | None = None,
        column: str | None = None,
    ) -> None:
        self.row_number = row_number
        self.column = column
        super().__init__(self._compose_message(message))

    def _compose_message(self, message: str) -> str:
        parts = [message]
        if self.row_number is not None:
            parts.append(f"row={self.row_number}")
        if self.column is not None:
            parts.append(f"column={self.column}")
        return " | ".join(parts)


@dataclass(slots=True)
class CSVSnapshotLoaderConfig:
    depth: int
    time_column: str = "time"
    enforce_strict_time_increase: bool = True
    validate_crossed_book: bool = True
    allow_zero_sizes: bool = True

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth must be > 0")
        if not self.time_column:
            raise ValueError("time_column must be non-empty")


def build_required_columns(depth: int, time_column: str = "time") -> list[str]:
    if depth <= 0:
        raise ValueError("depth must be > 0")

    columns = [time_column]
    for level in range(1, depth + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"bid_price_{level}",
                f"ask_size_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def validate_csv_columns(columns: list[str], depth: int, time_column: str = "time") -> None:
    missing = [column for column in build_required_columns(depth, time_column) if column not in columns]
    if missing:
        raise CSVFormatError(
            f"missing required column: {missing[0]}",
            column=missing[0],
        )


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")

    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as iso_exc:
        try:
            unix_seconds = float(text)
        except ValueError as exc:
            raise ValueError("invalid timestamp") from exc

        warnings.warn(
            "Non-standard timestamp format detected: interpreting value as Unix timestamp seconds in UTC.",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError("invalid timestamp") from exc


def _require_value(
    row: Mapping[str, Any],
    column: str,
    *,
    row_number: int | None,
) -> str:
    if column not in row:
        raise SnapshotValidationError("missing required value", row_number=row_number, column=column)

    raw_value = row[column]
    if raw_value is None:
        raise SnapshotValidationError("missing required value", row_number=row_number, column=column)

    value = str(raw_value).strip()
    if value == "":
        raise SnapshotValidationError("empty required value", row_number=row_number, column=column)
    return value


def _parse_float(
    row: Mapping[str, Any],
    column: str,
    *,
    row_number: int | None,
) -> float:
    value = _require_value(row, column, row_number=row_number)
    try:
        return float(value)
    except ValueError as exc:
        raise SnapshotValidationError("value is not a valid float", row_number=row_number, column=column) from exc


def _validate_price(price: float, column: str, *, row_number: int | None) -> None:
    if price <= 0:
        raise SnapshotValidationError("non-positive price", row_number=row_number, column=column)


def _validate_size(size: float, column: str, *, row_number: int | None, allow_zero_sizes: bool) -> None:
    if size < 0:
        raise SnapshotValidationError("negative size", row_number=row_number, column=column)
    if not allow_zero_sizes and size == 0:
        raise SnapshotValidationError("zero size is not allowed", row_number=row_number, column=column)


def validate_snapshot_row(
    row: Mapping[str, Any],
    config: CSVSnapshotLoaderConfig,
    *,
    row_number: int | None = None,
) -> None:
    try:
        _ = parse_timestamp(_require_value(row, config.time_column, row_number=row_number))
    except ValueError as exc:
        raise SnapshotValidationError(str(exc), row_number=row_number, column=config.time_column) from exc

    ask_prices: list[float] = []
    bid_prices: list[float] = []

    for level in range(1, config.depth + 1):
        ask_price_col = f"ask_price_{level}"
        bid_price_col = f"bid_price_{level}"
        ask_size_col = f"ask_size_{level}"
        bid_size_col = f"bid_size_{level}"

        ask_price = _parse_float(row, ask_price_col, row_number=row_number)
        bid_price = _parse_float(row, bid_price_col, row_number=row_number)
        ask_size = _parse_float(row, ask_size_col, row_number=row_number)
        bid_size = _parse_float(row, bid_size_col, row_number=row_number)

        _validate_price(ask_price, ask_price_col, row_number=row_number)
        _validate_price(bid_price, bid_price_col, row_number=row_number)
        _validate_size(
            ask_size,
            ask_size_col,
            row_number=row_number,
            allow_zero_sizes=config.allow_zero_sizes,
        )
        _validate_size(
            bid_size,
            bid_size_col,
            row_number=row_number,
            allow_zero_sizes=config.allow_zero_sizes,
        )

        ask_prices.append(ask_price)
        bid_prices.append(bid_price)

    for index in range(len(ask_prices) - 1):
        if ask_prices[index] >= ask_prices[index + 1]:
            raise SnapshotValidationError(
                "asks are not sorted ascending",
                row_number=row_number,
                column=f"ask_price_{index + 2}",
            )

    for index in range(len(bid_prices) - 1):
        if bid_prices[index] <= bid_prices[index + 1]:
            raise SnapshotValidationError(
                "bids are not sorted descending",
                row_number=row_number,
                column=f"bid_price_{index + 2}",
            )

    if config.validate_crossed_book and bid_prices[0] >= ask_prices[0]:
        raise SnapshotValidationError(
            "crossed book detected",
            row_number=row_number,
            column="ask_price_1",
        )


def parse_snapshot_row(
    row: Mapping[str, Any],
    config: CSVSnapshotLoaderConfig,
    *,
    row_number: int | None = None,
) -> Snapshot:
    validate_snapshot_row(row, config, row_number=row_number)

    timestamp = parse_timestamp(str(row[config.time_column]))
    asks = [
        Level(
            price=float(str(row[f"ask_price_{level}"]).strip()),
            qty=float(str(row[f"ask_size_{level}"]).strip()),
        )
        for level in range(1, config.depth + 1)
    ]
    bids = [
        Level(
            price=float(str(row[f"bid_price_{level}"]).strip()),
            qty=float(str(row[f"bid_size_{level}"]).strip()),
        )
        for level in range(1, config.depth + 1)
    ]
    return Snapshot(
        timestamp=timestamp,
        asks=asks,
        bids=bids,
        validate_crossed_book=config.validate_crossed_book,
    )
