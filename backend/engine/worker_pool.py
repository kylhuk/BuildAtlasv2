import concurrent.futures
import itertools
import json
import queue
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_WORKER_CMD: Sequence[str] = ("lua", "pob/worker/worker.lua")

WORKER_TERMINATED_ERROR_CODE = -32000
WORKER_PROTOCOL_ERROR_CODE = -32001


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
    ) -> None:
        self._cmd = tuple(cmd)
        self._request_timeout = request_timeout
        self._subprocess = subprocess_module or subprocess
        self.worker_id = worker_id
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
                raise WorkerCrashedError(message)
        return response

    def close(self, wait_timeout: float = 1.0) -> None:
        if self._process is None:
            return
        self._stop_event.set()
        try:
            if self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=wait_timeout)
        except Exception:
            try:
                self._process.kill()
            except Exception:  # pragma: no cover - fallback path
                pass
        finally:
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
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        self._worker_cmd = tuple(worker_cmd or DEFAULT_WORKER_CMD)
        self._request_timeout = request_timeout
        self._subprocess = subprocess_module or subprocess
        self._workers: List[WorkerProcess] = [
            WorkerProcess(
                cmd=self._worker_cmd,
                request_timeout=self._request_timeout,
                subprocess_module=self._subprocess,
                worker_id=i,
            )
            for i in range(num_workers)
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self._workers_lock = threading.Lock()
        self._round_robin_lock = threading.Lock()
        self._next_worker = 0
        self._closing = threading.Event()

    def evaluate_batch(self, payloads: Sequence[Any]) -> List[Dict[str, Any]]:
        if self._closing.is_set():
            raise WorkerPoolClosedError("worker pool is closed")
        if not payloads:
            return []
        futures: List[concurrent.futures.Future] = []
        for payload in payloads:
            worker_idx = self._assign_worker()
            futures.append(self._executor.submit(self._send_with_retries, worker_idx, payload))
        results: List[Dict[str, Any]] = []
        for future in futures:
            results.append(future.result())
        return results

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
            )
            self._workers[idx] = new_worker
            return new_worker
