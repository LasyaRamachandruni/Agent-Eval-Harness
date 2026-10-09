"""Typed settings read from environment variables.

Everything that changes between machines (where to cache responses, where
datasets live, how many requests to run at once) comes from the environment,
so nothing machine-specific is hard-coded. Command-line flags override these.

    AGENT_EVAL_DATA_DIR      where downloaded benchmark data lives  (default: data)
    AGENT_EVAL_CACHE_DIR     response cache directory               (default: .cache/agent-eval)
    AGENT_EVAL_CONCURRENCY   runs in flight at once                 (default: 4)
    AGENT_EVAL_MAX_RETRIES   retries on rate limits / server errors (default: 6)
    AGENT_EVAL_LOG_LEVEL     DEBUG, INFO, WARNING, ERROR            (default: WARNING)
    AGENT_EVAL_LOG_FORMAT    json or text                           (default: json)
    OLLAMA_HOST              Ollama server URL                      (default: http://localhost:11434)

API keys are read by the provider SDKs themselves (ANTHROPIC_API_KEY,
OPENAI_API_KEY) and are never stored or logged here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
LOG_FORMATS = ("json", "text")


class ConfigError(ValueError):
    """An environment variable holds a value that cannot be used."""


def _int(env: Mapping[str, str], key: str, default: int, minimum: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be at least {minimum}, got {value}")
    return value


def _choice(env: Mapping[str, str], key: str, default: str, choices: tuple[str, ...], upper: bool = False) -> str:
    raw = env.get(key) or default
    value = raw.upper() if upper else raw.lower()
    if value not in choices:
        raise ConfigError(f"{key} must be one of {', '.join(choices)}, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings; build them with Settings.from_env()."""

    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache/agent-eval")
    concurrency: int = 4
    max_retries: int = 6
    log_level: str = "WARNING"
    log_format: str = "json"
    ollama_host: str = "http://localhost:11434"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read settings from `env` (default: os.environ); raises ConfigError on bad values."""
        env = os.environ if env is None else env
        return cls(
            data_dir=Path(env.get("AGENT_EVAL_DATA_DIR") or "data"),
            cache_dir=Path(env.get("AGENT_EVAL_CACHE_DIR") or ".cache/agent-eval"),
            concurrency=_int(env, "AGENT_EVAL_CONCURRENCY", 4, 1),
            max_retries=_int(env, "AGENT_EVAL_MAX_RETRIES", 6, 0),
            log_level=_choice(env, "AGENT_EVAL_LOG_LEVEL", "WARNING", LOG_LEVELS, upper=True),
            log_format=_choice(env, "AGENT_EVAL_LOG_FORMAT", "json", LOG_FORMATS),
            ollama_host=(env.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/"),
        )
