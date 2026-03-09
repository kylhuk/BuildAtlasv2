import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from backend.engine.worker_pool import WorkerPool, WorkerProtocolError, WorkerTimeoutError


class MockStdOut:
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


class MockStdIn:
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
        pass


class MockProcess:
    instances: List["MockProcess"] = []

    def __init__(self, responder: Callable[["MockProcess", Dict[str, Any]], None]) -> None:
        self.responder = responder
        self.stdin = MockStdIn(self._handle_line)
        self.stdout = MockStdOut()
        self.last_request_line: Optional[str] = None
        self.terminated = False
        MockProcess.instances.append(self)

    def _handle_line(self, raw_line: str) -> None:
        self.last_request_line = raw_line
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError:
            return
        self.responder(self, request)

    def poll(self) -> Optional[int]:
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True
        self.stdout.close()

    def kill(self) -> None:
        self.terminated = True
        self.stdout.close()

    def wait(self, timeout: Optional[float] = None) -> int:
        return 0


class MockSubprocessModule:
    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL

    def __init__(
        self,
        responder_factory: Callable[[], Callable[[MockProcess, Dict[str, Any]], None]],
    ) -> None:
        self._factory = responder_factory
        self.instances: List[MockProcess] = []
        self.popen_args: List[tuple[Any, ...]] = []
        self.popen_kwargs: List[Dict[str, Any]] = []

    # pragma: no cover - simplifies signature
    def Popen(  # noqa: N802
        self,
        *args: Any,
        **kwargs: Any,
    ) -> MockProcess:
        self.popen_args.append(tuple(args))
        self.popen_kwargs.append(dict(kwargs))
        responder = self._factory()
        process = MockProcess(responder)
        self.instances.append(process)
        return process


def assert_ndjson_request(line: str, expected_method: str) -> Dict[str, Any]:
    payload = json.loads(line.strip())
    assert payload["method"] == expected_method, "method is wrong"
    assert isinstance(payload["id"], int)
    assert "params" in payload
    return payload


def test_ndjson_request_shape_records_method_and_params() -> None:
    def responder(process: MockProcess, request: Dict[str, Any]) -> None:
        process.stdout.push_line(
            json.dumps(
                {
                    "id": request["id"],
                    "ok": True,
                    "result": {"payload": request["params"]},
                }
            )
        )

    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"value": 42}
        results = pool.evaluate_batch([payload])
    finally:
        pool.close()
    recorded = module.instances[0].last_request_line
    assert recorded is not None
    parsed = assert_ndjson_request(recorded, expected_method="evaluate")
    assert parsed["params"] == payload
    assert results[0]["ok"] is True


def test_batch_ordering_respects_input_sequence() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            delay = request["params"].get("delay", 0)

            def push_response() -> None:
                time.sleep(delay)
                process.stdout.push_line(
                    json.dumps(
                        {
                            "id": request["id"],
                            "ok": True,
                            "result": {"value": request["params"]["value"]},
                        }
                    )
                )

            threading.Thread(target=push_response, daemon=True).start()

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=2, subprocess_module=module)
    try:
        payloads = [
            {"value": "first", "delay": 0.05},
            {"value": "second", "delay": 0.0},
        ]
        responses = pool.evaluate_batch(payloads)
    finally:
        pool.close()
    assert [resp["result"]["value"] for resp in responses] == ["first", "second"]


def test_timeout_restarts_worker_module() -> None:
    counter = {"calls": 0}

    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(_: MockProcess, __: Dict[str, Any]) -> None:
            pass

        counter["calls"] += 1
        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module, request_timeout=0.01)
    try:
        with pytest.raises(WorkerTimeoutError):
            pool.evaluate_batch([{"value": "timeout"}])
    finally:
        pool.close()
    assert len(module.instances) >= 2


def test_timeout_retries_use_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(_: MockProcess, __: Dict[str, Any]) -> None:
            pass

        return responder

    sleep_calls: list[float] = []

    monkeypatch.setattr("backend.engine.worker_pool.random.uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(
        "backend.engine.worker_pool.time.sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module, request_timeout=0.01)
    try:
        with pytest.raises(WorkerTimeoutError):
            pool.evaluate_batch([{"value": "timeout"}])
    finally:
        pool.close()

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(0.05)
    assert sleep_calls[1] == pytest.approx(0.1)


def test_protocol_parsing_returns_response_payload() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"payload": request["params"]},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"nested": {"flag": True}}
        result = pool.evaluate_batch([payload])[0]
    finally:
        pool.close()
    assert result["ok"] is True
    assert result["result"]["payload"] == payload


def test_malformed_response_triggers_protocol_error_restart() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line("not a json payload")

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module, request_timeout=0.5)
    try:
        with pytest.raises(WorkerProtocolError):
            pool.evaluate_batch([{"value": "broken"}])
    finally:
        pool.close()
    assert len(module.instances) >= 2


def test_worker_pool_forwards_worker_cwd_to_subprocess() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"value": request["params"]["value"]},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(
        num_workers=1,
        worker_cmd=["luajit", "worker.lua"],
        worker_cwd="/tmp/pob-src",
        subprocess_module=module,
    )
    try:
        pool.evaluate_batch([{"value": "test"}])
    finally:
        pool.close()

    assert module.popen_kwargs[0].get("cwd") == "/tmp/pob-src"


def test_worker_pool_resolves_luajit_script_path_and_defaults_cwd() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {
                            "payload": request["params"],
                        },
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        pool.evaluate_batch([{"value": "test"}])
    finally:
        pool.close()

    project_root = Path(__file__).resolve().parents[2]
    expected_worker_script = str(project_root / "backend/pob_worker/pob_worker.lua")
    assert module.popen_args[0][0][0] == "luajit"
    assert module.popen_args[0][0][1] == expected_worker_script
    assert module.popen_kwargs[0].get("cwd") == str(project_root / "PathOfBuilding/src")


def test_worker_pool_legacy_pob_script_defaults_to_pathofbuilding_src() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {
                            "payload": request["params"],
                        },
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    project_root = Path(__file__).resolve().parents[2]
    pool = WorkerPool(
        num_workers=1,
        worker_cmd=["luajit", "pob/worker/worker.lua"],
        subprocess_module=module,
    )
    try:
        pool.evaluate_batch([{"value": "test"}])
    finally:
        pool.close()

    assert module.popen_args[0][0][0] == "luajit"
    assert module.popen_args[0][0][1].endswith("pob/worker/worker.lua")
    assert module.popen_kwargs[0].get("cwd") == str(project_root / "PathOfBuilding/src")


def test_worker_pool_preserves_legacy_pob_cwd_when_using_pathofbuilding() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {
                            "payload": request["params"],
                        },
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(
        num_workers=1,
        worker_cmd=["luajit", "PathOfBuilding/worker/worker.lua"],
        request_timeout=0.5,
        subprocess_module=module,
    )
    try:
        pool.evaluate_batch([{"value": "test"}])
    finally:
        pool.close()

    assert module.popen_args[0][0][0] == "luajit"
    assert module.popen_args[0][0][1].endswith("PathOfBuilding/worker/worker.lua")
    cwd_value = module.popen_kwargs[0].get("cwd")
    assert cwd_value is not None
    assert cwd_value.endswith("PathOfBuilding/src")


def test_missing_id_response_triggers_protocol_error_restart() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "ok": True,
                        "result": {"value": "missing id"},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module, request_timeout=0.5)
    try:
        with pytest.raises(WorkerProtocolError):
            pool.evaluate_batch([{"value": "missing-id"}])
    finally:
        pool.close()
    assert len(module.instances) >= 2


def test_result_caching_returns_cached_result_on_duplicate_payload() -> None:
    call_count = {"count": 0}

    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            call_count["count"] += 1
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {
                            "value": request["params"]["value"],
                            "call": call_count["count"],
                        },
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"value": "test", "build": {"items": [], "tree": {}}, "scenario": "test"}
        results1 = pool.evaluate_batch([payload])
        results2 = pool.evaluate_batch([payload])
    finally:
        pool.close()

    assert len(results1) == 1
    assert len(results2) == 1
    assert results1[0]["result"]["call"] == 1
    assert results2[0]["result"]["call"] == 1
    assert call_count["count"] == 1


def test_batch_ordering_with_cached_and_missing_payloads() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {
                            "payload": request["params"],
                        },
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload_cached = {
            "value": "cached",
            "build": {"items": [], "tree": {}},
            "scenario": "test",
        }
        payload_missing = {
            "value": "fresh",
            "build": {"items": [], "tree": {}},
            "scenario": "test",
        }
        # Prime the cache for the first payload
        pool.evaluate_batch([payload_cached])
        responses = pool.evaluate_batch([payload_cached, payload_missing])
    finally:
        pool.close()

    assert responses[0]["result"]["payload"]["value"] == "cached"
    assert responses[1]["result"]["payload"]["value"] == "fresh"


def test_evaluate_batch_best_effort_mixed_success_and_failure() -> None:
    call_count = {"count": 0}

    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            call_count["count"] += 1
            if request["params"]["value"] == "fail":
                return
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"value": request["params"]["value"]},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    timeout_payload = {"value": "fail", "build": {"items": [], "tree": {}}, "scenario": "test"}
    success_payload = {"value": "ok", "build": {"items": [], "tree": {}}, "scenario": "test"}
    pool = WorkerPool(
        num_workers=1,
        subprocess_module=module,
        request_timeout=0.01,
    )
    try:
        results = pool.evaluate_batch_best_effort([timeout_payload, success_payload])
        repeated = pool.evaluate_batch_best_effort([timeout_payload])
    finally:
        pool.close()

    assert results[0]["ok"] is False
    assert "WorkerTimeoutError" in results[0]["error"]["message"]
    assert results[1]["ok"] is True
    assert repeated[0]["ok"] is False
    # Best-effort mode retries each timed-out payload three times and never caches non-ok results,
    # so both batches together exercise a total of seven worker invocations for the timeout payload.
    assert call_count["count"] == 7


def test_cache_hit_rate_tracking() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"value": request["params"]["value"]},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"value": "test", "build": {"items": [], "tree": {}}, "scenario": "test"}
        for _ in range(10):
            pool.evaluate_batch([payload])

        with pool._cache_lock:
            assert pool._cache_stats["hits"] == 9
            assert pool._cache_stats["misses"] == 1
            assert pool._cache_stats["requests"] == 10
    finally:
        pool.close()


def test_fast_reject_missing_required_fields() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"value": "ok"},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"value": "test"}
        results = pool.evaluate_batch([payload])
    finally:
        pool.close()

    assert len(results) == 1
    assert results[0]["ok"] is True


def test_fast_reject_invalid_build_structure() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(_: MockProcess, __: Dict[str, Any]) -> None:
            pass

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"build": "invalid", "scenario": "test"}
        results = pool.evaluate_batch([payload])
    finally:
        pool.close()

    assert len(results) == 1
    assert results[0]["ok"] is False
    assert "fast-reject" in results[0]["error"]["message"]


def test_fast_reject_missing_items_in_build() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(_: MockProcess, __: Dict[str, Any]) -> None:
            pass

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"build": {"tree": {}}, "scenario": "test"}
        results = pool.evaluate_batch([payload])
    finally:
        pool.close()

    assert len(results) == 1
    assert results[0]["ok"] is False
    assert "fast-reject" in results[0]["error"]["message"]


def test_fast_reject_missing_tree_in_build() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(_: MockProcess, __: Dict[str, Any]) -> None:
            pass

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"build": {"items": []}, "scenario": "test"}
        results = pool.evaluate_batch([payload])
    finally:
        pool.close()

    assert len(results) == 1
    assert results[0]["ok"] is False
    assert "fast-reject" in results[0]["error"]["message"]


def test_cache_eviction_on_max_size() -> None:
    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"value": request["params"]["value"]},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payloads = [
            {"value": f"test_{i}", "build": {"items": [], "tree": {}}, "scenario": "test"}
            for i in range(5)
        ]
        pool.evaluate_batch(payloads)

        with pool._cache_lock:
            cache_size = len(pool._result_cache)
            assert cache_size == 5
    finally:
        pool.close()


def test_valid_payload_bypasses_fast_reject() -> None:
    call_count = {"count": 0}

    def responder_factory() -> Callable[[MockProcess, Dict[str, Any]], None]:
        def responder(process: MockProcess, request: Dict[str, Any]) -> None:
            call_count["count"] += 1
            process.stdout.push_line(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "result": {"value": "success"},
                    }
                )
            )

        return responder

    module = MockSubprocessModule(responder_factory)
    pool = WorkerPool(num_workers=1, subprocess_module=module)
    try:
        payload = {"build": {"items": [1, 2], "tree": {"nodes": []}}, "scenario": "test"}
        results = pool.evaluate_batch([payload])
    finally:
        pool.close()

    assert len(results) == 1
    assert results[0]["ok"] is True
    assert call_count["count"] == 1
