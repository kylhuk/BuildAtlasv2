from __future__ import annotations

from typing import Sequence

from backend.engine.archive.store import ArchiveStoreEntry


class ArchiveEmitter:
    name: str

    def select(self, entries: Sequence[ArchiveStoreEntry], budget: int) -> list[ArchiveStoreEntry]:
        raise NotImplementedError


class ExploitEmitter(ArchiveEmitter):
    name = "exploit"

    def select(self, entries: Sequence[ArchiveStoreEntry], budget: int) -> list[ArchiveStoreEntry]:
        if budget <= 0:
            return []
        sorted_entries = sorted(
            entries,
            key=lambda entry: (-entry.score, entry.build_id),
        )
        return sorted_entries[:budget]


class NoveltyEmitter(ArchiveEmitter):
    name = "novelty"

    def select(self, entries: Sequence[ArchiveStoreEntry], budget: int) -> list[ArchiveStoreEntry]:
        if budget <= 0:
            return []
        selected: list[ArchiveStoreEntry] = []
        seen: set[str] = set()
        for entry in sorted(entries, key=lambda item: item.bin_key):
            if entry.bin_key in seen:
                continue
            seen.add(entry.bin_key)
            selected.append(entry)
            if len(selected) >= budget:
                break
        return selected


class UncertaintyEmitter(ArchiveEmitter):
    name = "uncertainty"

    def select(self, entries: Sequence[ArchiveStoreEntry], budget: int) -> list[ArchiveStoreEntry]:
        return []


def deterministic_allocator(
    total_budget: int, emitters: Sequence[ArchiveEmitter]
) -> dict[str, int]:
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative")
    if not emitters:
        return {}
    base = total_budget // len(emitters)
    remainder = total_budget % len(emitters)
    allocation: dict[str, int] = {}
    for index, emitter in enumerate(emitters):
        allocation[emitter.name] = base + (1 if index < remainder else 0)
    return allocation
