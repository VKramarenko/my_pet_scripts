from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sim.data.book_converters import bybit_snapshots_to_wide, legacy_bids_asks_to_wide
from sim.data.level_utils import parse_levels


def _write_wide(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".csv":
        df.to_csv(path, index=False)
    elif suf == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension {suf!r}; use .csv or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy L2 (ts+bids+asks CSV/Parquet) or Bybit orderbook JSON to wide L2 format."
    )
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output path (.csv or .parquet)")
    parser.add_argument(
        "--format",
        choices=["auto", "bybit", "legacy"],
        default="auto",
        help="Input format (default: .json -> bybit, otherwise legacy)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Book depth N in output columns (default: max depth found in input for both formats)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="For legacy input: symbol string written on every row. For bybit without symbol in JSON: fallback symbol.",
    )
    parser.add_argument(
        "--time-col",
        default="ts",
        help="Legacy only: time column name (default: ts)",
    )
    args = parser.parse_args()
    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    fmt = args.format
    if fmt == "auto":
        fmt = "bybit" if inp.suffix.lower() == ".json" else "legacy"

    if fmt == "bybit":
        wide = bybit_snapshots_to_wide(
            inp,
            depth=args.depth,
            symbol_default=args.symbol,
        )
    else:
        if not args.symbol:
            raise SystemExit("legacy format requires --symbol (instrument id for the wide table)")
        frame = pd.read_csv(inp) if inp.suffix.lower() == ".csv" else pd.read_parquet(inp)
        depth = args.depth
        if depth is None:
            depth = 1
            for _, row in frame.iterrows():
                depth = max(
                    depth,
                    len(parse_levels(row["bids"])),
                    len(parse_levels(row["asks"])),
                )
        wide = legacy_bids_asks_to_wide(
            frame,
            depth=depth,
            symbol=args.symbol,
            time_col=args.time_col,
        )

    _write_wide(wide, Path(args.output))
    print(f"Wrote {len(wide)} rows -> {args.output}")


if __name__ == "__main__":
    main()
