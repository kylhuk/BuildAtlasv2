from __future__ import annotations

from typing import Any, Sequence

from backend.engine.archive.store import ArchiveStoreEntry


def _coerce_probability(value: Any) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, probability))


def _uncertainty_score(entry: ArchiveStoreEntry) -> float:
    probability = _coerce_probability(entry.metadata.get("pass_probability"))
    if probability is None:
        return -1.0
    return 1.0 - abs(probability - 0.5) * 2.0


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
        if budget <= 0:
            return []
        scored_entries: list[tuple[float, ArchiveStoreEntry]] = []
        for entry in entries:
            score = _uncertainty_score(entry)
            scored_entries.append((score, entry))
        scored_entries.sort(key=lambda scored: (-scored[0], scored[1].build_id))
        return [entry for _, entry in scored_entries[:budget]]


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
