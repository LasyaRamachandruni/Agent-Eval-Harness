import sys

import pytest

from agent_eval.sandbox import network_namespace_available, run_python

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="sandbox limits are POSIX-only")


def test_runs_code_and_captures_output():
    r = run_python("print(6 * 7)")
    assert r.ok and r.stdout.strip() == "42" and r.summary().startswith("exit code 0")


def test_failing_code_reports_the_error():
    r = run_python("assert 1 == 2, 'nope'")
    assert not r.ok and "AssertionError: nope" in r.stderr


def test_timeout_kills_the_program():
    r = run_python("while True:\n    pass\n", timeout=1)
    assert r.timed_out and not r.ok and r.summary().startswith("timed out")


@pytest.mark.parametrize(
    "code, event",
    [
        ("import socket\nsocket.create_connection(('example.com', 80), timeout=1)", "socket.getaddrinfo"),
        ("import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 80))", "socket.connect"),
        ("import subprocess\nsubprocess.run(['ls'])", "subprocess.Popen"),
        ("import os\nos.system('ls')", "os.system"),
    ],
)
def test_network_and_processes_are_blocked(code, event):
    r = run_python(code)
    assert not r.ok and f"blocked in sandbox: {event}" in r.stderr


def test_memory_limit():
    r = run_python("x = bytearray(4 * 1024 ** 3)", memory_mb=512)
    assert not r.ok and "MemoryError" in r.stderr


def test_environment_is_empty_and_cwd_is_temporary(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "do-not-leak")
    r = run_python("import os\nprint(os.environ.get('SECRET_TOKEN'))\nprint(os.getcwd())")
    out = r.stdout.splitlines()
    assert out[0] == "None" and "agent-eval-sandbox-" in out[1]


def test_network_namespace_probe_returns_a_bool():
    assert isinstance(network_namespace_available(), bool)
