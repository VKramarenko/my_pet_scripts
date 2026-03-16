from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventLogger:
    records: list[dict[str, Any]] = field(default_factory=list)

    def log(self, payload: dict[str, Any]) -> None:
        self.records.append(payload)

