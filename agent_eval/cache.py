"""Response cache keyed by (model, prompt, params).

Re-running an evaluation, or resuming one after a crash, should not pay for
the same model call twice. `CachingModel` stores every reply in a small SQLite
file, keyed by a SHA-256 of:

    model name, sampling params, system prompt, messages, tool specs, sample index

The sample index matters: with temperature > 0, trial 3 of a task must be an
independent sample, not a copy of trial 1. The runner sets it per trial with
`sample_scope(trial)`, so repeats get their own cache entries while re-running
the same trial is free.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from .logs import get_logger
from .models import Message, ModelClient, ModelReply, ToolCall, ToolSpec

log = get_logger(__name__)

_sample: contextvars.ContextVar[int] = contextvars.ContextVar("agent_eval_sample", default=0)


@contextlib.contextmanager
def sample_scope(index: int) -> Iterator[None]:
    """Make model calls inside this block use sample `index` in their cache key."""
    token = _sample.set(index)
    try:
        yield
    finally:
        _sample.reset(token)


def current_sample() -> int:
    """The sample index model calls are cached under right now."""
    return _sample.get()


def cache_key(
    model: str,
    params: dict[str, object],
    system: str,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    sample: int,
) -> str:
    """Stable SHA-256 over everything that determines a model reply."""
    blob = json.dumps(
        {
            "model": model,
            "params": params,
            "system": system,
            "messages": messages,
            "tools": [asdict(t) for t in tools] if tools else None,
            "sample": sample,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    """A thread-safe SQLite key-value store of model replies."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT NOT NULL)")
            self._db.commit()

    def get(self, key: str) -> ModelReply | None:
        with self._lock:
            row = self._db.execute("SELECT reply FROM replies WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        calls = [ToolCall(**c) for c in data.pop("tool_calls", [])]
        return ModelReply(**data, tool_calls=calls)

    def put(self, key: str, reply: ModelReply) -> None:
        data = asdict(reply)
        data.pop("cached", None)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, json.dumps(data)))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM replies").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


class CachingModel(ModelClient):
    """Serve repeated calls from `cache`; replies served from the cache have `cached=True`."""

    def __init__(self, inner: ModelClient, cache: ResponseCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.params = inner.params
        self.supports_native_tools = inner.supports_native_tools
        self.hits = 0
        self.misses = 0

    def _lookup(self, key: str) -> ModelReply | None:
        reply = self.cache.get(key)
        if reply is not None:
            self.hits += 1
            reply.cached = True
            log.debug("cache hit", extra={"fields": {"model": self.name, "key": key[:12]}})
        else:
            self.misses += 1
        return reply

    def _key(self, system: str, messages: list[Message], tools: list[ToolSpec] | None) -> str:
        return cache_key(self.name, self.params.as_dict(), system, messages, tools, current_sample())

    def complete(self, system: str, messages: list[Message]) -> ModelReply:
        key = self._key(system, messages, None)
        hit = self._lookup(key)
        if hit is not None:
            return hit
        reply = self.inner.complete(system, messages)
        self.cache.put(key, reply)
        return reply

    def complete_with_tools(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        key = self._key(system, messages, tools)
        hit = self._lookup(key)
        if hit is not None:
            return hit
        reply = self.inner.complete_with_tools(system, messages, tools)
        self.cache.put(key, reply)
        return reply
