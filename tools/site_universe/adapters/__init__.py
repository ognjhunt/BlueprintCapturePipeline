"""Pure source adapters: raw bytes in, canonical SourceRecords out. No network."""

from __future__ import annotations

from dataclasses import dataclass, field


class AdapterError(ValueError):
    """Raw bytes do not have the layout the adapter was written for."""


@dataclass
class ParseResult:
    records: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def count(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount
