from dataclasses import dataclass


@dataclass
class SimClock:
    ts: float = 0.0

    def set(self, ts: float) -> None:
        self.ts = ts

