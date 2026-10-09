"""Retries with exponential backoff for rate limits and transient API errors.

`RetryingModel` wraps any ModelClient. A call that fails with a rate limit
(HTTP 429), an overloaded or server error (5xx, Anthropic's 529), a timeout or
a dropped connection is retried with exponential backoff and full jitter,
honouring a `retry-after` header when the provider sends one. Anything else
(a bad request, a missing key) fails immediately, because retrying cannot fix
it. Each retry is logged with the wait time and the reason.
"""

from __future__ import annotations

import random
import time
import urllib.error
from collections.abc import Callable
from typing import Any

from .logs import get_logger
from .models import Message, ModelClient, ModelReply, ToolSpec

log = get_logger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
RETRYABLE_NAMES = frozenset(
    {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
    }
)


def status_code(exc: BaseException) -> int | None:
    """The HTTP status of a provider error, if it carries one."""
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits, overloads, server errors, timeouts and connection drops."""
    if type(exc).__name__ in RETRYABLE_NAMES:
        return True
    code = status_code(exc)
    if code is not None:
        return code in RETRYABLE_STATUS
    return isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError))


def retry_after(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait (from a retry-after header), if any."""
    headers: Any = getattr(getattr(exc, "response", None), "headers", None) or getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except AttributeError:
        return None
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None


class RetryingModel(ModelClient):
    """Retry transient failures of `inner` with exponential backoff and full jitter."""

    def __init__(
        self,
        inner: ModelClient,
        max_retries: int = 6,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        self.inner = inner
        self.name = inner.name
        self.params = inner.params
        self.supports_native_tools = inner.supports_native_tools
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.retries = 0  # total retries made, for reporting

    def _delay(self, attempt: int, exc: BaseException) -> float:
        hinted = retry_after(exc)
        if hinted is not None:
            return min(hinted, self.max_delay)
        return self._rng.uniform(0, min(self.max_delay, self.base_delay * 2**attempt))

    def _call(self, fn: Callable[[], ModelReply]) -> ModelReply:
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:
                if attempt >= self.max_retries or not is_retryable(exc):
                    raise
                wait = self._delay(attempt, exc)
                attempt += 1
                self.retries += 1
                log.warning(
                    "retrying model call",
                    extra={
                        "fields": {
                            "model": self.name,
                            "attempt": attempt,
                            "wait_s": round(wait, 2),
                            "error": type(exc).__name__,
                            "status": status_code(exc),
                        }
                    },
                )
                self._sleep(wait)

    def complete(self, system: str, messages: list[Message]) -> ModelReply:
        return self._call(lambda: self.inner.complete(system, messages))

    def complete_with_tools(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        return self._call(lambda: self.inner.complete_with_tools(system, messages, tools))
