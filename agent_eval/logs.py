"""Structured logging.

Diagnostics (retries, cache hits, sandbox timeouts, resumed runs) go through
the standard `logging` module as one JSON object per line on stderr, so they
can be grepped or shipped to a log store. Results the user asked for (the
live PASS/FAIL lines and the summary) are command output and stay on stdout.

    log = get_logger(__name__)
    log.info("retrying", extra={"fields": {"attempt": 2, "wait_s": 1.5}})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_ROOT = "agent_eval"


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON line with its `fields` merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable variant: `level logger event key=value ...`."""

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", None)
        extra = " ".join(f"{k}={v}" for k, v in fields.items()) if isinstance(fields, dict) else ""
        return f"{record.levelname.lower():<7} {record.name} {record.getMessage()} {extra}".rstrip()


def configure_logging(level: str = "WARNING", fmt: str = "json") -> None:
    """Send this package's logs to stderr in the given format (idempotent)."""
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger.handlers = [handler]
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """A logger under the package root, e.g. get_logger(__name__)."""
    return logging.getLogger(name if name.startswith(_ROOT) else f"{_ROOT}.{name}")
