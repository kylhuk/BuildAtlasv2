from __future__ import annotations

import math
from typing import Iterable, Sequence


def novelty_score(reference: Sequence[float], candidate: Sequence[float]) -> float:
    length = max(len(reference), len(candidate))
    total = 0.0
    for index in range(length):
        ref_val = reference[index] if index < len(reference) else 0.0
        cand_val = candidate[index] if index < len(candidate) else 0.0
        diff = ref_val - cand_val
        total += diff * diff
    return math.sqrt(total)


def enforce_quota(scores: Iterable[float], quota: int) -> dict:
    sorted_scores = sorted(scores, reverse=True)
    accepted = sorted_scores[:quota]
    rejected = sorted_scores[quota:]
    return {
        "quota": quota,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
    }
