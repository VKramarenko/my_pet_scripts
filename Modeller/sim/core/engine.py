from __future__ import annotations

from collections.abc import Callable, Iterable

from sim.core.events import Event


class ReplayEngine:
    def __init__(self) -> None:
        self._handlers: list[Callable[[Event], None]] = []

    def register_handler(self, handler: Callable[[Event], None]) -> None:
        self._handlers.append(handler)

    def run(self, events: Iterable[Event]) -> None:
        for event in events:
            for handler in self._handlers:
                handler(event)

