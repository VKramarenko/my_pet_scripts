from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from src.models import Snapshot
from src.validation import (
    CSVFormatError,
    CSVSnapshotLoaderConfig,
    SnapshotValidationError,
    parse_snapshot_row,
    validate_csv_columns,
)


def _validate_time_sequence(
    current: Snapshot,
    previous: Snapshot | None,
    *,
    config: CSVSnapshotLoaderConfig,
    row_number: int,
) -> None:
    if previous is None:
        return

    if config.enforce_strict_time_increase:
        is_invalid = current.timestamp <= previous.timestamp
        message = "duplicate or non-increasing timestamps"
    else:
        is_invalid = current.timestamp < previous.timestamp
        message = "decreasing timestamp detected"

    if is_invalid:
        raise SnapshotValidationError(message, row_number=row_number, column=config.time_column)


def iter_snapshots_csv(path: str | Path, config: CSVSnapshotLoaderConfig) -> Iterator[Snapshot]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CSVFormatError("CSV file is missing header row")

        validate_csv_columns(reader.fieldnames, config.depth, config.time_column)

        previous_snapshot: Snapshot | None = None
        for row_index, row in enumerate(reader, start=2):
            snapshot = parse_snapshot_row(row, config, row_number=row_index)
            _validate_time_sequence(snapshot, previous_snapshot, config=config, row_number=row_index)
            previous_snapshot = snapshot
            yield snapshot


def load_snapshots_csv(path: str | Path, config: CSVSnapshotLoaderConfig) -> list[Snapshot]:
    return list(iter_snapshots_csv(path, config))
