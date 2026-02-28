from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

OperatorRow = Mapping[str, Any]

MODEL_VERSION = "V6-03"


class RunningStat:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def stdev(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / self.count)


class OperatorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.metrics: Dict[str, RunningStat] = {}

    def update(self, delta: Mapping[str, float]) -> None:
        self.count += 1
        for metric, value in delta.items():
            stat = self.metrics.setdefault(metric, RunningStat())
            stat.add(float(value))

    def summarize(self) -> Dict[str, Dict[str, float]]:
        return {
            metric: {
                "mean": round(stat.mean, 2),
                "stdev": round(stat.stdev(), 2),
            }
            for metric, stat in self.metrics.items()
        }


def train_surrogate(rows: Iterable[OperatorRow]) -> Dict[str, Any]:
    accumulators: Dict[str, OperatorAccumulator] = {}
    for row in rows:
        operator = row.get("operator")
        delta = row.get("delta")
        if not operator or not isinstance(delta, Mapping):
            continue
        accumulator = accumulators.setdefault(operator, OperatorAccumulator())
        accumulator.update(delta)

    model: Dict[str, Any] = {
        "metadata": {
            "version": MODEL_VERSION,
            "operator_count": len(accumulators),
        },
        "operator_stats": {},
    }

    for operator, accumulator in accumulators.items():
        model["operator_stats"][operator] = {
            "count": accumulator.count,
            "metrics": accumulator.summarize(),
        }

    return model


def write_model(path: Path | str, model: Mapping[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)


def read_model(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def infer(model: Mapping[str, Any], operator: str, features: Mapping[str, float]) -> Dict[str, Any]:
    operator_stats = model.get("operator_stats", {}).get(operator)
    if not operator_stats:
        return {
            "operator": operator,
            "prediction": {},
            "uncertainty": 1.0,
        }

    prediction: Dict[str, float] = {}
    deltas: Dict[str, float] = {}
    total_stdev = 0.0
    for metric, stats in operator_stats.get("metrics", {}).items():
        mean = float(stats.get("mean", 0.0))
        stdev = float(stats.get("stdev", 0.0))
        base = float(features.get(metric, 0.0))
        deltas[metric] = round(mean, 2)
        prediction[metric] = round(base + mean, 2)
        total_stdev += stdev

    operator_count = operator_stats.get("count", 1)
    uncertainty = min(1.0, total_stdev / (1 + operator_count))

    return {
        "operator": operator,
        "prediction": prediction,
        "delta": deltas,
        "uncertainty": round(uncertainty, 3),
        "sample_count": operator_count,
    }


def load_dataset(path: Path | str) -> List[OperatorRow]:
    rows: List[OperatorRow] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows
