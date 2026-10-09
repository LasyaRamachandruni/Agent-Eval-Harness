"""Run untrusted Python (model-written code) in a separate, locked-down process.

Used by the HumanEval adapter, both to grade completions against the hidden
tests and to let agents run their own code with the `run_python` tool.

What it does, in layers:

1. A fresh interpreter (`python -I -B`): isolated mode ignores PYTHON* env
   vars and the user site directory; the child gets an empty environment and a
   temporary working directory that is deleted afterwards.
2. A wall-clock timeout; on expiry the whole process group is killed.
3. Resource limits set inside the child before the code runs: CPU seconds,
   address space (memory), file size and open files.
4. No network: on Linux the child runs in a new network namespace
   (`unshare -rn`) when the kernel allows it, so it has no interfaces at all.
   Everywhere, an audit hook installed before the code runs refuses socket
   connections and DNS lookups, starting processes, and `os.system`/`exec`.

This is good isolation for benchmark code written by a model in good faith.
It is not a security boundary against deliberately malicious code; for that,
run the harness in a disposable container or VM.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .logs import get_logger

log = get_logger(__name__)

MAX_OUTPUT_CHARS = 4000

_GUARD = textwrap.dedent(
    """
    import sys

    def _limits(cpu_s, mem_mb):
        try:
            import resource
        except ImportError:  # not available on Windows
            return
        for name, value in (
            ("RLIMIT_CPU", cpu_s),
            ("RLIMIT_AS", mem_mb * 1024 * 1024),
            ("RLIMIT_FSIZE", 10 * 1024 * 1024),
            ("RLIMIT_NOFILE", 64),
        ):
            limit = getattr(resource, name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass

    _BLOCKED = (
        "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto",
        "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
        "os.kill", "os.killpg", "pty.spawn", "shutil.rmtree", "ctypes.dlopen",
    )

    def _audit(event, args):
        if event.startswith(_BLOCKED):
            raise PermissionError("blocked in sandbox: " + event)

    _limits({cpu}, {mem})
    _code = open({path!r}, encoding="utf-8").read()
    sys.addaudithook(_audit)
    sys.argv = ["solution.py"]
    exec(compile(_code, "solution.py", "exec"), {{"__name__": "__main__"}})
    """
)


@dataclass(frozen=True)
class SandboxResult:
    """How a sandboxed program ended."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    network_isolated: bool = False

    @property
    def ok(self) -> bool:
        """True if the program exited with status 0 within the time limit."""
        return self.returncode == 0 and not self.timed_out

    def summary(self, limit: int = MAX_OUTPUT_CHARS) -> str:
        """Exit status and (clipped) output, as shown to an agent."""
        status = "timed out" if self.timed_out else f"exit code {self.returncode}"
        parts = [status]
        if self.stdout.strip():
            parts.append("stdout:\n" + _clip(self.stdout, limit))
        if self.stderr.strip():
            parts.append("stderr:\n" + _clip(self.stderr, limit))
        return "\n".join(parts)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[-limit:]


@lru_cache(maxsize=1)
def network_namespace_available() -> bool:
    """True if `unshare -rn` works here (Linux with unprivileged user namespaces)."""
    if not sys.platform.startswith("linux") or shutil.which("unshare") is None:
        return False
    try:
        probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def run_python(code: str, timeout: float = 10.0, memory_mb: int = 1024) -> SandboxResult:
    """Run `code` in a sandboxed interpreter and return its exit status and output."""
    isolated = network_namespace_available()
    with tempfile.TemporaryDirectory(prefix="agent-eval-sandbox-") as tmp:
        solution = Path(tmp) / "solution.py"
        solution.write_text(code, encoding="utf-8")
        guard = Path(tmp) / "guard.py"
        guard.write_text(_GUARD.format(cpu=int(timeout) + 1, mem=memory_mb, path=str(solution)), encoding="utf-8")
        cmd = [sys.executable, "-I", "-B", str(guard)]
        if isolated:
            cmd = ["unshare", "-rn", *cmd]
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "HOME": tmp, "LC_ALL": "C.UTF-8"}
        proc = subprocess.Popen(
            cmd,
            cwd=tmp,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
            return SandboxResult(proc.returncode, out, err, network_isolated=isolated)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, err = proc.communicate()
            log.info("sandbox timeout", extra={"fields": {"timeout_s": timeout}})
            return SandboxResult(-1, out or "", err or "", timed_out=True, network_isolated=isolated)


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()
