import pytest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Keep every test's response cache and data in its own temp folder, and run one task at a time
    so scripted models replay their replies in order."""
    monkeypatch.setenv("AGENT_EVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AGENT_EVAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_EVAL_CONCURRENCY", "1")
    monkeypatch.delenv("AGENT_EVAL_LOG_LEVEL", raising=False)
    monkeypatch.delenv("AGENT_EVAL_LOG_FORMAT", raising=False)
