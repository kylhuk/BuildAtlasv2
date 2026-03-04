import concurrent.futures
import itertools
import json
import logging
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKER_CMD: Sequence[str] = ("luajit", "backend/pob_worker/pob_worker.lua")

WORKER_TERMINATED_ERROR_CODE = -32000
WORKER_PROTOCOL_ERROR_CODE = -32001

logger = logging.getLogger(__name__)


def _resolve_luajit_script_path(script_path: Path) -> Path:
    if script_path.is_absolute():
        return script_path

    candidate_paths = [
        Path.cwd() / script_path,
        PROJECT_ROOT / script_path,
        PROJECT_ROOT / "backend" / script_path,
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

    # Keep invocation stable under cwd changes even when the script is absent
    # in the local checkout (for example, when PathOfBuilding assets are not
    # initialized yet): prefer an absolute project-root path.
    return PROJECT_ROOT / script_path


def _normalize_worker_invocation(
    worker_cmd: Sequence[str], worker_cwd: str | None
) -> tuple[tuple[str, ...], str | None]:
    if len(worker_cmd) < 2:
        return tuple(worker_cmd), worker_cwd

    command_name = worker_cmd[0]
    if command_name not in {"luajit", "luajit.exe"}:
        return tuple(worker_cmd), worker_cwd

    script = Path(worker_cmd[1])
    resolved_script = _resolve_luajit_script_path(script)
    normalized_cmd = [
        str(worker_cmd[0]),
        str(resolved_script),
        *[str(arg) for arg in worker_cmd[2:]],
    ]

    resolved_cwd = worker_cwd
    script_parent = resolved_script.parent
    script_root_name = script_parent.parent.name
    is_pathofbuilding_worker = (
        resolved_script.name == "worker.lua"
        and script_parent.name == "worker"
        and script_root_name in {"PathOfBuilding", "pob"}
    )
    is_backend_pob_worker = (
        resolved_script.name == "pob_worker.lua"
        and script_parent.name == "pob_worker"
        and script_root_name == "backend"
    )
    if resolved_cwd is None and (is_pathofbuilding_worker or is_backend_pob_worker):
        resolved_cwd = str(PROJECT_ROOT / "PathOfBuilding" / "src")

    return tuple(normalized_cmd), resolved_cwd


def _is_worker_log_line(line: str) -> bool:
    log_prefixes = (
        # Original PoB worker prefixes
        "Loading",
        "Unicode support detected",
        "Removing legacy",
        "Startup time:",
        "Uniques loaded",
        "Rares loaded",
        "Processing tree",
        "PoB",  # PoB LuaJIT worker status lines (started, exiting, etc.)
        # Backend PoB worker initialization debug lines
        "Pre-initialized",
        "HeadlessWrapper",
        "Applying",
        "Global",
        "Captured",
        "mainObject",
        "Calling",
        "After",
        "No ",
        "Workaround",
        "ERROR",
        "loadBuildFromXML",
    )
    return any(line.startswith(prefix) for prefix in log_prefixes)


class WorkerPoolError(Exception):
    pass


class WorkerTimeoutError(WorkerPoolError):
    pass


class WorkerCrashedError(WorkerPoolError):
    pass


class WorkerProtocolError(WorkerPoolError):
    pass


class WorkerPoolClosedError(WorkerPoolError):
    pass


class WorkerProcess:
    def __init__(
        self,
        cmd: Sequence[str],
        request_timeout: float,
        subprocess_module: Optional[Any] = None,
        worker_id: int = 0,
        worker_cwd: str | None = None,
    ) -> None:
        self._cmd = tuple(cmd)
        self._request_timeout = request_timeout
        self._subprocess = subprocess_module or subprocess
        self.worker_id = worker_id
        self._worker_cwd = worker_cwd
        self._id_counter = itertools.count(1)
        self._lock = threading.Lock()
        self._pending: Dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._start_process()

    def send_request(self, method: str, params: Any) -> Dict[str, Any]:
        if self._process is None or self._process.poll() is not None:
            raise WorkerCrashedError("worker not running")

        request_id = next(self._id_counter)
        response_queue: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue

        payload = {"id": request_id, "method": method, "params": params}
        try:
            with self._lock:
                self._write_payload(payload)
        except (BrokenPipeError, OSError) as exc:
            raise WorkerCrashedError("worker pipe closed") from exc

        try:
            response = response_queue.get(timeout=self._request_timeout)
        except queue.Empty as exc:
            raise WorkerTimeoutError("worker timed out waiting for response") from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
        error = response.get("error") if isinstance(response, dict) else None
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message", "worker error")
            if code == WORKER_PROTOCOL_ERROR_CODE:
                raise WorkerProtocolError(message)
            if code == WORKER_TERMINATED_ERROR_CODE:
                if message == "worker terminated":
                    raise WorkerCrashedError(message)
        return response

    def close(self, wait_timeout: float = 1.0) -> None:
        if self._process is None:
            return
        self._stop_event.set()
        process = self._process
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=wait_timeout)
        except Exception:
            try:
                process.kill()
            except Exception:  # pragma: no cover - fallback path
                pass
        finally:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except Exception:
                pass
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass
            self._process = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=wait_timeout)
            self._reader_thread = None

    def restart(self) -> "WorkerProcess":
        self.close()
        self._pending.clear()
        self._start_process()
        return self

    def _start_process(self) -> None:
        self._stop_event = threading.Event()
        self._process = self._subprocess.Popen(
            list(self._cmd),
            stdin=self._subprocess.PIPE,
            stdout=self._subprocess.PIPE,
            stderr=self._subprocess.DEVNULL,
            cwd=self._worker_cwd,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"worker-reader-{self.worker_id}",
            daemon=True,
        )
        self._reader_thread.start()

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        assert self._process is not None
        self._process.stdin.write(json.dumps(payload))
        self._process.stdin.write("\n")
        self._process.stdin.flush()

    def _notify_pending_protocol_error(self, message: str) -> None:
        with self._pending_lock:
            pending_items = list(self._pending.items())
        for request_id, pending_queue in pending_items:
            try:
                pending_queue.put(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": {
                            "code": WORKER_PROTOCOL_ERROR_CODE,
                            "message": message,
                        },
                    },
                    block=False,
                )
            except queue.Full:
                continue

    def _reader_loop(self) -> None:
        assert self._process is not None
        stdout = self._process.stdout
        while not self._stop_event.is_set():
            line = stdout.readline()
            if line == "":
                break
            line = line.strip()
            if not line:
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                # LuaJIT worker startup may emit plain-text status logs on stdout.
                if _is_worker_log_line(line):
                    continue
                self._notify_pending_protocol_error(f"malformed json response: {exc}")
                break
            request_id = decoded.get("id")
            if request_id is None:
                self._notify_pending_protocol_error("missing id in response")
                break
            with self._pending_lock:
                queue_item = self._pending.get(request_id)
            if queue_item is not None:
                queue_item.put(decoded)
        with self._pending_lock:
            pending_items = list(self._pending.items())
        for request_id, pending_queue in pending_items:
            try:
                pending_queue.put(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": {
                            "code": WORKER_TERMINATED_ERROR_CODE,
                            "message": "worker terminated",
                        },
                    },
                    block=False,
                )
            except queue.Full:
                continue


class WorkerPool:
    def __init__(
        self,
        num_workers: int = 1,
        worker_cmd: Optional[Sequence[str]] = None,
        request_timeout: float = 5.0,
        subprocess_module: Optional[Any] = None,
        worker_cwd: str | None = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        base_cmd = tuple(worker_cmd or DEFAULT_WORKER_CMD)
        self._worker_cmd, self._worker_cwd = _normalize_worker_invocation(base_cmd, worker_cwd)
        self._request_timeout = request_timeout
        self._subprocess = subprocess_module or subprocess
        self._workers: List[WorkerProcess] = [
            WorkerProcess(
                cmd=self._worker_cmd,
                request_timeout=self._request_timeout,
                subprocess_module=self._subprocess,
                worker_id=i,
                worker_cwd=self._worker_cwd,
            )
            for i in range(num_workers)
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self._workers_lock = threading.Lock()
        self._round_robin_lock = threading.Lock()
        self._next_worker = 0
        self._closing = threading.Event()

    def evaluate_batch(
        self,
        payloads: Sequence[Any],
        *,
        progress_label: str | None = None,
    ) -> List[Dict[str, Any]]:
        if self._closing.is_set():
            raise WorkerPoolClosedError("worker pool is closed")
        if not payloads:
            return []
        futures: List[concurrent.futures.Future] = []
        future_to_index: dict[concurrent.futures.Future, int] = {}
        for index, payload in enumerate(payloads):
            worker_idx = self._assign_worker()
            future = self._executor.submit(self._send_with_retries, worker_idx, payload)
            futures.append(future)
            future_to_index[future] = index
        total = len(futures)
        results: list[dict[str, Any] | None] = [None] * total
        pending: set[concurrent.futures.Future] = set(futures)
        completed = 0
        batch_started_at = time.monotonic()
        heartbeat_interval = max(5.0, min(30.0, self._request_timeout / 2.0))

        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=heartbeat_interval,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                elapsed = max(0.0, time.monotonic() - batch_started_at)
                label_prefix = f"{progress_label} " if progress_label else ""
                logger.warning(
                    "worker pool batch %sstill running (completed=%d/%d pending=%d elapsed=%.1fs)",
                    label_prefix,
                    completed,
                    total,
                    len(pending),
                    elapsed,
                )
                continue
            for future in done:
                index = future_to_index[future]
                results[index] = future.result()
                completed += 1

        if total > 1:
            elapsed = max(0.0, time.monotonic() - batch_started_at)
            label_prefix = f"{progress_label} " if progress_label else ""
            logger.info(
                "worker pool batch %scompleted (completed=%d elapsed=%.1fs)",
                label_prefix,
                total,
                elapsed,
            )

        typed_results: List[Dict[str, Any]] = []
        for result in results:
            if result is None:
                raise WorkerProtocolError("worker pool returned incomplete batch results")
            typed_results.append(result)
        return typed_results

    def close(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        self._executor.shutdown(wait=True)
        for worker in self._workers:
            worker.close()

    def _assign_worker(self) -> int:
        with self._round_robin_lock:
            idx = self._next_worker
            self._next_worker = (self._next_worker + 1) % len(self._workers)
            return idx

    def _send_with_retries(self, worker_idx: int, payload: Any) -> Dict[str, Any]:
        worker = self._workers[worker_idx]
        last_exc: Optional[Exception] = None
        for _ in range(2):
            try:
                return worker.send_request("evaluate", payload)
            except (WorkerTimeoutError, WorkerCrashedError, WorkerProtocolError) as exc:
                last_exc = exc
                worker = self._restart_worker(worker_idx)
        assert last_exc is not None
        raise last_exc

    def _restart_worker(self, idx: int) -> WorkerProcess:
        with self._workers_lock:
            old_worker = self._workers[idx]
            old_worker.close()
            new_worker = WorkerProcess(
                cmd=self._worker_cmd,
                request_timeout=self._request_timeout,
                subprocess_module=self._subprocess,
                worker_id=idx,
                worker_cwd=self._worker_cwd,
            )
            self._workers[idx] = new_worker
            return new_worker
