from __future__ import annotations

import math

import pandas as pd
import pytest

from sim.data.book_converters import (
    bybit_snapshots_to_wide,
    dataframe_rows_to_snapshots,
    legacy_bids_asks_to_wide,
    wide_to_legacy_lists,
)
from sim.data.book_schema import detect_book_depth
from sim.data.level_utils import parse_levels
from sim.data.loaders import load_wide_l2_snapshots


def test_detect_book_depth() -> None:
    cols = ["time", "symbol", "ask_price_1", "bid_price_1", "ask_size_1", "bid_size_1", "ask_price_2"]
    assert detect_book_depth(cols) == 2


def test_legacy_wide_roundtrip_levels() -> None:
    df = pd.DataFrame(
        {
            "ts": [1.0, 2.0],
            "bids": ["[[100.0, 1.0]]", "[[101.0, 2.0]]"],
            "asks": ["[[101.0, 1.0]]", "[[102.0, 2.0]]"],
        }
    )
    wide = legacy_bids_asks_to_wide(df, depth=2, symbol="X")
    assert list(wide.columns)[:2] == ["time", "symbol"]
    legacy_back = wide_to_legacy_lists(wide)
    for orig, back in zip(
        df.itertuples(index=False),
        legacy_back.itertuples(index=False),
        strict=True,
    ):
        assert parse_levels(orig.bids) == parse_levels(back.bids)
        assert parse_levels(orig.asks) == parse_levels(back.asks)


def test_bybit_records_to_wide_and_snapshots() -> None:
    records = [
        {
            "symbol": "X",
            "timestamp": 1.0,
            "bids": [["100", "1"]],
            "asks": [["101", "1"]],
        },
    ]
    wide = bybit_snapshots_to_wide(records, depth=2, symbol_default="FALLBACK")
    snaps = list(dataframe_rows_to_snapshots(wide, 2, "time"))
    assert len(snaps) == 1
    assert snaps[0].ts == 1.0
    assert snaps[0].symbol == "X"
    assert snaps[0].bids == [(100.0, 1.0)]
    assert snaps[0].asks == [(101.0, 1.0)]


def test_load_wide_l2_csv(tmp_path) -> None:
    p = tmp_path / "w.csv"
    p.write_text(
        "time,symbol,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "1.0,AAA,101,100,1,2\n"
        "2.0,AAA,102,99,1,3\n",
        encoding="utf-8",
    )
    snaps = list(load_wide_l2_snapshots(p))
    assert len(snaps) == 2
    assert snaps[0].symbol == "AAA"
    assert snaps[0].asks == [(101.0, 1.0)]
    assert snaps[0].bids == [(100.0, 2.0)]


def test_load_wide_multi_symbol_requires_filter(tmp_path) -> None:
    p = tmp_path / "m.csv"
    p.write_text(
        "time,symbol,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "1.0,A,101,100,1,1\n"
        "2.0,B,101,100,1,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple symbols"):
        list(load_wide_l2_snapshots(p))
    snaps = list(load_wide_l2_snapshots(p, symbol_filter="B"))
    assert len(snaps) == 1
    assert snaps[0].symbol == "B"


def test_wide_ts_alias(tmp_path) -> None:
    p = tmp_path / "t.csv"
    p.write_text(
        "ts,symbol,ask_price_1,bid_price_1,ask_size_1,bid_size_1\n"
        "10.0,Z,201,200,1,1\n",
        encoding="utf-8",
    )
    snaps = list(load_wide_l2_snapshots(p))
    assert snaps[0].ts == 10.0


def test_wide_sparse_level_skipped() -> None:
    wide = pd.DataFrame(
        [
            {
                "time": 0.0,
                "symbol": "S",
                "ask_price_1": 101.0,
                "bid_price_1": 100.0,
                "ask_size_1": 1.0,
                "bid_size_1": 1.0,
                "ask_price_2": math.nan,
                "bid_price_2": math.nan,
                "ask_size_2": math.nan,
                "bid_size_2": math.nan,
            }
        ]
    )
    snaps = list(dataframe_rows_to_snapshots(wide, 2, "time"))
    assert len(snaps[0].bids) == 1
