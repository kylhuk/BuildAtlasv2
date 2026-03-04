from __future__ import annotations

import argparse
import importlib
import json
import math
import queue
import statistics
import subprocess
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Any, Callable

DEFAULT_BASELINE_DIR = Path(".sisyphus") / "baselines"


class _MockStdOut:
    def __init__(self) -> None:
        self._buffer: queue.Queue[str] = queue.Queue()

    def push_line(self, line: str) -> None:
        if line and not line.endswith("\n"):
            line = f"{line}\n"
        self._buffer.put(line)

    def readline(self) -> str:
        return self._buffer.get()

    def close(self) -> None:
        self._buffer.put("")


class _MockStdIn:
    def __init__(self, on_line: Callable[[str], None]) -> None:
        self._buffer = ""
        self._on_line = on_line

    def write(self, data: str) -> None:
        self._buffer += data
        while "\n" in self._buffer:
            line, _, remainder = self._buffer.partition("\n")
            self._buffer = remainder
            self._on_line(line)

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class _MockProcess:
    def __init__(self, responder: Callable[["_MockProcess", dict[str, Any]], None]) -> None:
        self.stdin = _MockStdIn(self._handle_line)
        self.stdout = _MockStdOut()
        self._responder = responder
        self._terminated = False

    def _handle_line(self, raw_line: str) -> None:
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError:
            return
        self._responder(self, request)

    def poll(self) -> int | None:
        return 0 if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True
        self.stdout.close()

    def kill(self) -> None:
        self._terminated = True
        self.stdout.close()

    def wait(self, timeout: float | None = None) -> int:
        _ = timeout
        return 0


class _MockSubprocessModule:
    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL

    def __init__(self) -> None:
        self.instances: list[_MockProcess] = []

    def _responder(self) -> Callable[["_MockProcess", dict[str, Any]], None]:
        def responder(process: _MockProcess, request: dict[str, Any]) -> None:
            response = {
                "id": request.get("id"),
                "ok": True,
                "result": {"seed": request.get("params", {}).get("seed")},
            }
            process.stdout.push_line(json.dumps(response))

        return responder

    def Popen(self, *args: Any, **kwargs: Any) -> _MockProcess:  # noqa: N802
        _ = args, kwargs
        process = _MockProcess(self._responder())
        self.instances.append(process)
        return process


def _resolve_optional_path(value: str | None, default_path: Path) -> Path | None:
    if value is None:
        return None
    if value == "":
        return default_path
    return Path(value)


def _write_baseline(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"baseline saved: {path}")


def _compare_against_baseline(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        print(f"baseline not found: {path}")
        return
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"unable to read baseline {path}: {exc}")
        return
    baseline_metrics = baseline.get("metrics") if isinstance(baseline, dict) else None
    current_metrics = payload.get("metrics")
    if not isinstance(baseline_metrics, dict) or not isinstance(current_metrics, dict):
        print("baseline comparison skipped: invalid metric payload")
        return

    print(f"baseline comparison: {path}")
    common_keys = sorted(set(baseline_metrics).intersection(current_metrics))
    for key in common_keys:
        baseline_value = baseline_metrics.get(key)
        current_value = current_metrics.get(key)
        if not isinstance(baseline_value, (int, float)) or not isinstance(
            current_value, (int, float)
        ):
            continue
        delta = float(current_value) - float(baseline_value)
        if baseline_value != 0:
            pct = (delta / float(baseline_value)) * 100.0
            print(
                f"  {key}: current={current_value:.6f}, baseline={baseline_value:.6f}, delta={delta:+.6f} ({pct:+.2f}%)"
            )
        else:
            print(
                f"  {key}: current={current_value:.6f}, baseline={baseline_value:.6f}, delta={delta:+.6f}"
            )


def _finalize_results(
    benchmark_name: str,
    metrics: dict[str, Any],
    baseline_arg: str | None,
    compare_arg: str | None,
) -> None:
    baseline_path = _resolve_optional_path(
        baseline_arg, DEFAULT_BASELINE_DIR / f"benchmark_{benchmark_name}.json"
    )
    compare_path = _resolve_optional_path(
        compare_arg, DEFAULT_BASELINE_DIR / f"benchmark_{benchmark_name}.json"
    )

    payload = {
        "benchmark": benchmark_name,
        "metrics": metrics,
    }
    print(json.dumps(payload, indent=2))

    if baseline_path is not None:
        _write_baseline(baseline_path, payload)
    if compare_path is not None:
        _compare_against_baseline(compare_path, payload)


def benchmark_worker_pool(args: argparse.Namespace) -> int:
    count = max(1, int(args.count))
    workers = max(1, int(args.workers))
    payloads = [
        {
            "build": {"items": [seed], "tree": {"nodes": [seed]}},
            "scenario": "benchmark",
            "seed": seed,
        }
        for seed in range(count)
    ]

    worker_pool_module = importlib.import_module("backend.engine.worker_pool")
    worker_pool_cls = getattr(worker_pool_module, "WorkerPool")
    pool = worker_pool_cls(
        num_workers=workers,
        request_timeout=max(0.01, float(args.request_timeout)),
        subprocess_module=_MockSubprocessModule(),
    )
    try:
        started = monotonic()
        responses = pool.evaluate_batch(payloads, progress_label="benchmark")
        elapsed = max(1e-9, monotonic() - started)
    finally:
        pool.close()

    ok_count = sum(1 for row in responses if row.get("ok") is True)
    failed_count = count - ok_count
    throughput = count / elapsed
    latency_ms = (elapsed / count) * 1000.0
    metrics = {
        "count": count,
        "workers": workers,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "elapsed_seconds": elapsed,
        "throughput_eval_per_sec": throughput,
        "latency_ms_per_eval": latency_ms,
    }
    _finalize_results("worker_pool", metrics, args.baseline, args.compare)
    return 0


def _single_ml_iteration(size: int) -> float:
    started = monotonic()
    acc = 0.0
    for idx in range(size):
        value = idx / max(1, size)
        acc += math.sqrt(value + 1.0) * math.log1p(value + 1.0)
    _ = acc
    return max(1e-9, monotonic() - started)


def benchmark_ml_loop(args: argparse.Namespace) -> int:
    iterations = max(1, int(args.iterations))
    workload_size = max(1, int(args.workload_size))

    iteration_times = [_single_ml_iteration(workload_size) for _ in range(iterations)]
    total_elapsed = sum(iteration_times)
    metrics = {
        "iterations": iterations,
        "workload_size": workload_size,
        "total_seconds": total_elapsed,
        "average_iteration_seconds": statistics.mean(iteration_times),
        "min_iteration_seconds": min(iteration_times),
        "max_iteration_seconds": max(iteration_times),
        "iteration_seconds": iteration_times,
    }
    _finalize_results("ml_loop", metrics, args.baseline, args.compare)
    return 0


def benchmark_api(args: argparse.Namespace) -> int:
    from fastapi.testclient import TestClient

    app_module = importlib.import_module("app.main")
    app = getattr(app_module, "app")

    count = max(1, int(args.count))
    latencies_ms: list[float] = []
    with TestClient(app) as client:
        for _ in range(count):
            started = monotonic()
            response = client.get(args.path)
            elapsed = max(1e-9, monotonic() - started)
            if response.status_code != 200:
                raise RuntimeError(f"api benchmark failed: {args.path} -> {response.status_code}")
            latencies_ms.append(elapsed * 1000.0)

    metrics = {
        "count": count,
        "path": args.path,
        "latency_ms_avg": statistics.mean(latencies_ms),
        "latency_ms_min": min(latencies_ms),
        "latency_ms_max": max(latencies_ms),
        "latency_ms_p50": statistics.median(latencies_ms),
        "latency_ms_p95": _percentile(latencies_ms, 95),
    }
    _finalize_results("api", metrics, args.baseline, args.compare)
    return 0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline",
        nargs="?",
        const="",
        default=None,
        help="Save baseline JSON (optional path, default .sisyphus/baselines)",
    )
    parser.add_argument(
        "--compare",
        nargs="?",
        const="",
        default=None,
        help="Compare against baseline JSON (optional path, default .sisyphus/baselines)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Performance benchmark harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker_parser = subparsers.add_parser("worker-pool", help="Benchmark worker pool throughput")
    worker_parser.add_argument("--count", type=int, default=100, help="Number of evaluations")
    worker_parser.add_argument("--workers", type=int, default=4, help="Worker pool size")
    worker_parser.add_argument(
        "--request-timeout",
        type=float,
        default=5.0,
        help="Per-request timeout in seconds",
    )
    _add_common_arguments(worker_parser)

    ml_parser = subparsers.add_parser("ml-loop", help="Benchmark ML loop iteration time")
    ml_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    ml_parser.add_argument(
        "--workload-size",
        type=int,
        default=200000,
        help="Workload units per iteration",
    )
    _add_common_arguments(ml_parser)

    api_parser = subparsers.add_parser("api", help="Benchmark API latency")
    api_parser.add_argument("--count", type=int, default=50, help="Number of requests")
    api_parser.add_argument("--path", default="/health", help="API path to call")
    _add_common_arguments(api_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "worker-pool":
        return benchmark_worker_pool(args)
    if args.command == "ml-loop":
        return benchmark_ml_loop(args)
    if args.command == "api":
        return benchmark_api(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
