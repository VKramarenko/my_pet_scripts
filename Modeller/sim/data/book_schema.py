from __future__ import annotations

import re

BOOK_LEVEL_COL_RE = re.compile(
    r"^(?P<side>ask|bid)_(?P<kind>price|size)_(?P<idx>\d+)$"
)


def detect_book_depth(columns: list[str] | tuple[str, ...]) -> int:
    depth = 0
    for col in columns:
        m = BOOK_LEVEL_COL_RE.match(str(col))
        if m:
            depth = max(depth, int(m.group("idx")))
    return depth


def level_column_names(i: int) -> tuple[str, str, str, str]:
    return (
        f"ask_price_{i}",
        f"bid_price_{i}",
        f"ask_size_{i}",
        f"bid_size_{i}",
    )


def required_wide_level_columns(depth: int) -> list[str]:
    names: list[str] = []
    for i in range(1, depth + 1):
        names.extend(level_column_names(i))
    return names


def resolve_time_column(df_columns: list[str] | tuple[str, ...]) -> str:
    cols = set(df_columns)
    if "time" in cols:
        return "time"
    if "ts" in cols:
        return "ts"
    raise ValueError(
        "Wide book frame must have a time column named 'time' or 'ts'"
    )
