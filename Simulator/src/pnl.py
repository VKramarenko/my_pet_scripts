from __future__ import annotations

from src.models import Snapshot


def compute_mid_price(snapshot: Snapshot) -> float | None:
    return snapshot.mid_price()

