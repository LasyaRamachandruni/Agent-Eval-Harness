"""Model clients.

The agent talks to every model through one small interface:

    complete(system, messages) -> ModelReply                      # JSON-action protocol
    complete_with_tools(system, messages, tools) -> ModelReply    # native tool calling

Messages are plain dicts. For the JSON-action protocol they are just
{"role": "user"|"assistant", "content": str}. Native tool calling adds two
provider-neutral shapes, which each client converts to its own API:

    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Provider SDKs are imported lazily, so you only need the one you actually use.
Sampling settings (temperature, max tokens) live in `GenerationParams` and are
part of the response-cache key.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class GenerationParams:
    """Sampling settings sent with every request."""

    temperature: float = 0.7
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

    def as_dict(self) -> dict[str, Any]:
        """Plain dict form, used in cache keys and run metadata."""
        return asdict(self)


@dataclass(frozen=True)
class ToolSpec:
    """A tool as offered to a model with native tool calling (JSON-schema parameters)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model. `error` is set if its arguments were not valid JSON."""

    id: str
    name: str
    args: dict[str, Any]
    error: str | None = None


@dataclass
class ModelReply:
    """The text of one model reply, any native tool calls, and the tokens it used."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    cached: bool = False


class ModelClient:
    """Base class for models. Subclass it and implement `complete` to plug in any model.

    `name` identifies the model in results and reports, e.g. "openai:gpt-4o".
    Override `complete_with_tools` and set `supports_native_tools = True` to
    support the native tool-calling protocol as well.
    """

    name: str = "base"
    params: GenerationParams = GenerationParams()
    supports_native_tools: bool = False

    def complete(self, system: str, messages: list[Message]) -> ModelReply:  # pragma: no cover
        """Return the model's next reply to a system prompt and a list of chat messages."""
        raise NotImplementedError

    def complete_with_tools(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        """Like `complete`, but the model may answer with native tool calls."""
        raise NotImplementedError(f"{self.name} does not support native tool calling (use --protocol json)")


class ScriptedModel(ModelClient):
    """Replays a fixed list of replies. Used in tests and offline demos."""

    def __init__(self, replies: list[str], name: str = "scripted"):
        self.replies = list(replies)
        self.name = name

    def complete(self, system: str, messages: list[Message]) -> ModelReply:
        if not self.replies:
            return ModelReply(text='{"final": "(scripted model ran out of replies)"}')
        return ModelReply(text=self.replies.pop(0))


def _parse_args(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Tool-call arguments arrive as a dict or a JSON string; return (args, error)."""
    if isinstance(raw, dict):
        return raw, None
    if raw in (None, ""):
        return {}, None
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        return {}, f"tool arguments were not valid JSON: {exc}"
    if not isinstance(value, dict):
        return {}, "tool arguments must be a JSON object"
    return value, None


def _openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in tools
    ]


# --- Anthropic ------------------------------------------------------------


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to the Anthropic format (tool results become user tool_result blocks)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m["tool_calls"]:
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]})
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": role, "content": m["content"]})
    return out


class AnthropicModel(ModelClient):
    """Claude models through the Anthropic SDK (reads ANTHROPIC_API_KEY)."""

    supports_native_tools = True

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int | None = None,
        params: GenerationParams | None = None,
    ):
        import anthropic  # lazy import

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.params = params or GenerationParams(max_tokens=max_tokens or 1024)
        self.name = f"anthropic:{model}"

    def _create(self, system: str, messages: list[Message], tools: list[ToolSpec] | None) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.params.max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "temperature": self.params.temperature,
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        return self.client.messages.create(**kwargs)

    def _reply(self, resp: Any) -> ModelReply:
        text = "".join(block.text for block in resp.content if block.type == "text")
        calls = [
            ToolCall(block.id, block.name, *_parse_args(block.input))
            for block in resp.content
            if block.type == "tool_use" and hasattr(block, "id")
        ]
        return ModelReply(text, resp.usage.input_tokens, resp.usage.output_tokens, calls)

    def complete(self, system: str, messages: list[Message]) -> ModelReply:
        return self._reply(self._create(system, messages, None))

    def complete_with_tools(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        return self._reply(self._create(system, messages, tools))


# --- OpenAI ---------------------------------------------------------------


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to the OpenAI chat format."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                        }
                        for c in m["tool_calls"]
                    ],
                }
            )
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class OpenAIModel(ModelClient):
    """OpenAI chat models through the OpenAI SDK (reads OPENAI_API_KEY)."""

    supports_native_tools = True

    def __init__(
        self, model: str = "gpt-4o-mini", max_tokens: int | None = None, params: GenerationParams | None = None
    ):
        import openai  # lazy import

        self.client = openai.OpenAI()  # reads OPENAI_API_KEY
        self.model = model
        self.params = params or GenerationParams(max_tokens=max_tokens or 1024)
        self.name = f"openai:{model}"

    def _create(self, system: str, messages: list[Message], tools: list[ToolSpec] | None) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.params.max_tokens,
            "temperature": self.params.temperature,
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = _openai_tools(tools)
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, *_parse_args(c.function.arguments))
            for c in (getattr(msg, "tool_calls", None) or [])
        ]
        usage = resp.usage
        return ModelReply(
            msg.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            calls,
        )

    def complete(self, system: str, messages: list[Message]) -> ModelReply:
        return self._create(system, messages, None)

    def complete_with_tools(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        return self._create(system, messages, tools)


# --- Ollama ---------------------------------------------------------------


def to_ollama_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to Ollama's /api/chat format."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "tool":
            out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [{"function": {"name": c["name"], "arguments": c["args"]}} for c in m["tool_calls"]],
                }
            )
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class OllamaModel(ModelClient):
    """Free local models via Ollama (https://ollama.com). No API key needed."""

    supports_native_tools = True

    def __init__(
        self,
        model: str = "llama3.1",
        host: str | None = None,
        params: GenerationParams | None = None,
        timeout: float = 300.0,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.params = params or GenerationParams()
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def _chat(self, system: str, messages: list[Message], tools: list[ToolSpec] | None) -> ModelReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama_messages(system, messages),
            "stream": False,
            "options": {"temperature": self.params.temperature, "num_predict": self.params.max_tokens},
        }
        if tools:
            payload["tools"] = _openai_tools(tools)
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        msg = data["message"]
        calls = [
            ToolCall(
                f"call_{uuid.uuid4().hex[:12]}", c["function"]["name"], *_parse_args(c["function"].get("arguments"))
            )
            for c in msg.get("tool_calls") or []
        ]
        return ModelReply(msg.get("content") or "", data.get("prompt_eval_count", 0), data.get("eval_count", 0), calls)

    def complete(self, system: str, messages: list[Message]) -> ModelReply:
        return self._chat(system, messages, None)

    def complete_with_tools(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        return self._chat(system, messages, tools)


def load_model(spec: str, params: GenerationParams | None = None) -> ModelClient:
    """Build a client from a spec like 'anthropic:claude-haiku-4-5-20251001' or 'ollama:llama3.1'."""
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicModel(model, params=params) if model else AnthropicModel(params=params)
    if provider == "openai":
        return OpenAIModel(model, params=params) if model else OpenAIModel(params=params)
    if provider == "ollama":
        return OllamaModel(model, params=params) if model else OllamaModel(params=params)
    raise ValueError(f"unknown model provider '{provider}' (use anthropic, openai or ollama)")


# provider -> (Python package it needs, environment variable holding its API key)
PROVIDERS: dict[str, tuple[str | None, str | None]] = {
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
    "openai": ("openai", "OPENAI_API_KEY"),
    "ollama": (None, None),
}


def preflight(spec: str, timeout: float = 3.0) -> list[str]:
    """Problems that would stop `spec` from running, found without calling the model.

    Checks that the provider is known, its SDK is installed and its API key is
    set; for Ollama, that the local server answers and the model is pulled.
    An empty list means the model is ready. Nothing here costs tokens.
    """
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS:
        return [f"unknown model provider '{provider}' (use {', '.join(PROVIDERS)})"]
    package, key_var = PROVIDERS[provider]
    problems = []
    if package and importlib.util.find_spec(package) is None:
        problems.append(f"the '{package}' package is not installed (pip install -e \".[{package}]\")")
    if key_var and not os.environ.get(key_var):
        problems.append(f"{key_var} is not set (see .env.example)")
    if provider == "ollama":
        problems += _ollama_problems(model or "llama3.1", timeout)
    return problems


def _ollama_problems(model: str, timeout: float) -> list[str]:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            names = {m.get("name", "") for m in json.loads(r.read()).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return [f"cannot reach Ollama at {host} ({exc}); start it with 'ollama serve'"]
    if model not in names and f"{model}:latest" not in names:
        return [f"Ollama model '{model}' is not pulled (ollama pull {model})"]
    return []
