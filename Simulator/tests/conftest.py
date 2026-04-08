from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture
def base_time() -> datetime:
    return datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)


@pytest.fixture
def next_time(base_time: datetime) -> datetime:
    return base_time + timedelta(seconds=1)

