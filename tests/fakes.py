"""Fake models for offline tests: deterministic, thread-safe, no network."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from agent_eval.models import Message, ModelClient, ModelReply, ToolCall, ToolSpec

Handler = Callable[[str, list[Message], list[ToolSpec] | None], ModelReply | str]


class FunctionModel(ModelClient):
    """A model whose reply is computed by a plain function of (system, messages, tools).

    The function may return a ModelReply or just the reply text. Every call is
    recorded in `calls` so tests can inspect the prompts the strategy sent.
    """

    def __init__(self, fn: Handler, name: str = "fake:model", native: bool = True, tokens: tuple[int, int] = (10, 5)):
        self.fn = fn
        self.name = name
        self.supports_native_tools = native
        self.tokens = tokens
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _reply(self, system: str, messages: list[Message], tools: list[ToolSpec] | None) -> ModelReply:
        with self._lock:
            self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        out = self.fn(system, messages, tools)
        if isinstance(out, str):
            return ModelReply(out, *self.tokens)
        return out

    def complete(self, system: str, messages: list[Message]) -> ModelReply:
        return self._reply(system, messages, None)

    def complete_with_tools(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        return self._reply(system, messages, tools)


def act(tool: str, **args: Any) -> str:
    """A JSON-protocol tool call."""
    return json.dumps({"thought": "", "tool": tool, "args": args})


def final(answer: str) -> str:
    """A JSON-protocol final answer."""
    return json.dumps({"thought": "", "final": answer})


def native_call(name: str, call_id: str = "c1", **args: Any) -> ModelReply:
    """A native-protocol reply with one tool call."""
    return ModelReply("", 10, 5, [ToolCall(call_id, name, args)])


def is_review(system: str) -> bool:
    """True for the reviewer calls made by the reflection and reflexion strategies."""
    return system.startswith(("You are a strict reviewer", "You evaluate one attempt"))


def turns(messages: list[Message]) -> int:
    """How many assistant turns the conversation already has."""
    return sum(1 for m in messages if m["role"] == "assistant")
