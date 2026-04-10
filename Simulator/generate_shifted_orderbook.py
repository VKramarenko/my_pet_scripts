"""
Generate a shifted order book CSV from an existing one.

Usage:
    python generate_shifted_orderbook.py

Edit the CONFIG section below to control the output.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_CSV  = Path(r"..\Modeller\test_data\ob_big.csv")
OUTPUT_CSV = Path(r"..\Modeller\test_data\ob_big_shifted_info.csv")

# Timestamp jitter: each row gets a random offset in [-MAX_JITTER_S, +MAX_JITTER_S] seconds.
# Set to 0 to disable.
MAX_JITTER_S: float = 4.0

# Constant offset added to every timestamp (seconds). Positive = shift into future.
# Useful to make the two books align on the same timeline but start at different moments.
TIMESTAMP_OFFSET_S: float = 6.0

# Price transformation applied to every price column.
# PRICE_MULTIPLIER scales all prices (e.g. 0.1 → prices ÷10, 1.0 → no change).
# PRICE_ADDITIVE adds a fixed amount after scaling.
PRICE_MULTIPLIER: float = 0.5
PRICE_ADDITIVE: float = 0.1

# New symbol name written into the `symbol` column (if the column exists).
NEW_SYMBOL: str = "INFUSDT"

# Random seed for reproducibility. Set to None for a fresh random result each run.
RANDOM_SEED: int | None = 42

# ---------------------------------------------------------------------------

def is_price_column(col: str) -> bool:
    return "price" in col.lower()


def transform_price(value: str) -> str:
    if not value:
        return value
    try:
        return str(float(value) * PRICE_MULTIPLIER + PRICE_ADDITIVE)
    except ValueError:
        return value


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Empty or header-less CSV")

        rows = list(reader)

    print(f"Read {len(rows)} rows from {INPUT_CSV}")

    out_rows: list[dict] = []
    for row in rows:
        new_row = dict(row)

        # --- timestamp ---
        jitter = random.uniform(-MAX_JITTER_S, MAX_JITTER_S) if MAX_JITTER_S else 0.0
        new_row["time"] = str(float(row["time"]) + TIMESTAMP_OFFSET_S + jitter)

        # --- symbol ---
        if "symbol" in new_row:
            new_row["symbol"] = NEW_SYMBOL

        # --- prices ---
        for col in fieldnames:
            if is_price_column(col):
                new_row[col] = transform_price(row[col])

        out_rows.append(new_row)

    # Sort by timestamp (jitter can break order for adjacent rows)
    out_rows.sort(key=lambda r: float(r["time"]))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Written {len(out_rows)} rows to {OUTPUT_CSV}")
    print(f"  timestamp offset: +{TIMESTAMP_OFFSET_S}s  jitter: ±{MAX_JITTER_S}s")
    print(f"  price: × {PRICE_MULTIPLIER} + {PRICE_ADDITIVE}  symbol: {NEW_SYMBOL}")


if __name__ == "__main__":
    main()
