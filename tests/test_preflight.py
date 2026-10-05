"""`agent-eval check` finds setup problems without calling any model."""

import importlib.util

from agent_eval import cli
from agent_eval.models import preflight


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    problems = preflight("openai:gpt-4o-mini")
    assert any("OPENAI_API_KEY is not set" in p for p in problems)


def test_missing_sdk_is_reported(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "anthropic" else real(name))
    problems = preflight("anthropic:claude-haiku-4-5")
    assert problems == ['the \'anthropic\' package is not installed (pip install -e ".[anthropic]")']


def test_ready_when_sdk_and_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert preflight("openai:gpt-4o") == []


def test_unknown_provider():
    assert "unknown model provider 'gemeni'" in preflight("gemeni:pro")[0]


def test_unreachable_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens on the discard port
    problems = preflight("ollama:llama3.1", timeout=0.5)
    assert len(problems) == 1 and "cannot reach Ollama" in problems[0]


def test_check_command_exit_code(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert cli.main(["check", "--model", "openai:gpt-4o"]) == 0
    assert cli.main(["check", "--model", "openai:gpt-4o", "--model", "anthropic:claude-sonnet-4-5"]) == 1
    out = capsys.readouterr().out
    assert "[OK] openai:gpt-4o" in out and "[NOT READY] anthropic:claude-sonnet-4-5" in out
