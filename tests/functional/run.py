"""Server lifecycle and provenance for this extension's functional suites.

A subset of Gufo's tests/functional/run.py (lfm3 branch, 500df164): the model
suites here need only an owned server and the harness identity.
"""
import contextlib
import hashlib
import http.client
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

TESTS = Path(__file__).resolve().parent


def harness_files():
    """Every Python module of these suites."""
    return sorted(TESTS.glob("*.py"))


def provenance():
    source = hashlib.sha256()
    for path in harness_files():
        source.update(path.read_bytes())
    kernel_command = Path("/proc/cmdline")
    return {
        "harness_sha256": source.hexdigest(),
        "environment": {"host": platform.node(), "kernel": platform.release(),
                        "kernel_command_line": (kernel_command.read_text().strip()
                                                if kernel_command.is_file() else None), **{
            key: os.environ.get(key) for key in (
                "GPU_MAX_HW_QUEUES", "HSA_ENABLE_SDMA", "HSA_ENABLE_INTERRUPT",
                "HSA_USERPTR_FOR_PAGED_MEM", "HSA_USE_SVM",
                "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH", "LD_PRELOAD")}},
    }


def option(command, name, default=None):
    values = []
    for index, arg in enumerate(command):
        if arg == name:
            if index + 1 == len(command):
                raise ValueError(f"{name} needs a value")
            values.append(command[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"duplicate {name}")
    return values[0] if values else default


SERVER_POPEN = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt" else {})


def interrupt_server(process):
    """Requests the graceful shutdown that Ctrl+C starts."""
    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)


@contextlib.contextmanager
def server(command, log_path, startup_timeout, environment=None):
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    with log_path.open("w") as log:
        process = None
        previous_handlers = {}
        try:
            stop_signals = [signal.SIGTERM]
            if hasattr(signal, "SIGHUP"):
                stop_signals.append(signal.SIGHUP)
            previous_handlers = {signum: signal.signal(signum, interrupt)
                                 for signum in stop_signals}
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, **SERVER_POPEN)
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
                connection = http.client.HTTPConnection(
                    "127.0.0.1", int(option(command, "--port")), timeout=1)
                try:
                    connection.request("GET", "/health")
                    if connection.getresponse().status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                time.sleep(.1)
            else:
                raise TimeoutError(f"server startup timed out; see {log_path}")
            yield process
        finally:
            if process is not None and process.poll() is None:
                interrupt_server(process)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
