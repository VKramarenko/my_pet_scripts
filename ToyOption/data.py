"""Canonical data format for option quotes."""

from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np


@dataclass
class CanonicalQuoteSet:
    """Unified internal format for a single-expiry option quote set.

    Each call/put entry is (strike, price, weight).
    """

    F: float  # forward price
    T: float  # time to expiry in years
    calls: list[tuple[float, float, float]] = field(default_factory=list)
    puts: list[tuple[float, float, float]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_manual(
        cls,
        F: float,
        T: float,
        calls: list[tuple[float, float]],
        puts: list[tuple[float, float]],
        meta: Optional[dict] = None,
    ) -> CanonicalQuoteSet:
        """Create from simple (K, price) lists; weights default to 1."""
        _validate_positive(F, "F")
        _validate_positive(T, "T")
        c = [(k, p, 1.0) for k, p in calls]
        p = [(k, p, 1.0) for k, p in puts]
        return cls(F=F, T=T, calls=c, puts=p, meta=meta or {})

    @classmethod
    def from_dict(cls, d: dict) -> CanonicalQuoteSet:
        """Create from a plain dictionary (e.g. JSON payload from UI)."""
        return cls.from_manual(
            F=float(d["F"]),
            T=float(d["T"]),
            calls=[(float(r["K"]), float(r["price"])) for r in d.get("calls", [])],
            puts=[(float(r["K"]), float(r["price"])) for r in d.get("puts", [])],
            meta=d.get("meta", {}),
        )

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> CanonicalQuoteSet:
        """Load from CSV file.

        Expected format (see example_data.csv):
            F,<value>
            T,<value>
            type,K,price,weight
            call,90,12.5,1.0
            put,90,2.4,1.0
            ...

        The 'weight' column is optional (defaults to 1.0).
        """
        path = Path(path)
        F = T = None
        calls: list[tuple[float, float]] = []
        puts: list[tuple[float, float]] = []
        weights_c: list[float] = []
        weights_p: list[float] = []

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header_seen = False
            for row in reader:
                if not row or not row[0].strip():
                    continue
                tag = row[0].strip()

                # Metadata rows
                if tag == "F":
                    F = float(row[1])
                    continue
                if tag == "T":
                    T = float(row[1])
                    continue

                # Header row
                if tag == "type":
                    header_seen = True
                    continue

                # Data rows
                if header_seen and tag in ("call", "put"):
                    K = float(row[1])
                    price = float(row[2])
                    w = float(row[3]) if len(row) > 3 and row[3].strip() else 1.0
                    if tag == "call":
                        calls.append((K, price))
                        weights_c.append(w)
                    else:
                        puts.append((K, price))
                        weights_p.append(w)

        if F is None or T is None:
            raise ValueError("CSV must contain F and T rows")

        _validate_positive(F, "F")
        _validate_positive(T, "T")
        c = [(k, p, w) for (k, p), w in zip(calls, weights_c)]
        p = [(k, p, w) for (k, p), w in zip(puts, weights_p)]
        return cls(F=F, T=T, calls=c, puts=p, meta={"source": str(path)})

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def call_strikes(self) -> np.ndarray:
        return np.array([k for k, _, _ in self.calls])

    def call_prices(self) -> np.ndarray:
        return np.array([p for _, p, _ in self.calls])

    def call_weights(self) -> np.ndarray:
        return np.array([w for _, _, w in self.calls])

    def put_strikes(self) -> np.ndarray:
        return np.array([k for k, _, _ in self.puts])

    def put_prices(self) -> np.ndarray:
        return np.array([p for _, p, _ in self.puts])

    def put_weights(self) -> np.ndarray:
        return np.array([w for _, _, w in self.puts])

    def all_strikes(self) -> np.ndarray:
        """Sorted unique strikes across calls and puts."""
        s = set(k for k, _, _ in self.calls) | set(k for k, _, _ in self.puts)
        return np.sort(list(s))

    def n_points(self) -> int:
        return len(self.calls) + len(self.puts)


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
