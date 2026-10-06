"""Model clients.

The agent talks to every model through the same tiny interface:
    complete(system, messages) -> ModelReply

Messages are plain {"role": "user"|"assistant", "content": str} dicts, so any
chat model works. Provider SDKs are imported lazily, so you only need the one
you actually use.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ModelReply:
    """The text of one model reply and the tokens it used."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ModelClient:
    """Base class for models. Subclass it and implement `complete` to plug in any model.

    `name` identifies the model in results and reports, e.g. "openai:gpt-4o".
    """

    name: str = "base"

    def complete(self, system: str, messages: list[dict]) -> ModelReply:  # pragma: no cover
        """Return the model's next reply to a system prompt and a list of chat messages."""
        raise NotImplementedError


class ScriptedModel(ModelClient):
    """Replays a fixed list of replies. Used in tests and offline demos."""

    def __init__(self, replies: list[str], name: str = "scripted"):
        self.replies = list(replies)
        self.name = name

    def complete(self, system: str, messages: list[dict]) -> ModelReply:
        if not self.replies:
            return ModelReply(text='{"final": "(scripted model ran out of replies)"}')
        return ModelReply(text=self.replies.pop(0))


class AnthropicModel(ModelClient):
    """Claude models through the Anthropic SDK (reads ANTHROPIC_API_KEY)."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1024):
        import anthropic  # lazy import

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, messages: list[dict]) -> ModelReply:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            temperature=0,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ModelReply(text, resp.usage.input_tokens, resp.usage.output_tokens)


class OpenAIModel(ModelClient):
    """OpenAI chat models through the OpenAI SDK (reads OPENAI_API_KEY)."""

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1024):
        import openai  # lazy import

        self.client = openai.OpenAI()  # reads OPENAI_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"openai:{model}"

    def complete(self, system: str, messages: list[dict]) -> ModelReply:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, *messages],
        )
        usage = resp.usage
        return ModelReply(
            resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class OllamaModel(ModelClient):
    """Free local models via Ollama (https://ollama.com). No API key needed."""

    def __init__(self, model: str = "llama3.1", host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.name = f"ollama:{model}"

    def complete(self, system: str, messages: list[dict]) -> ModelReply:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        return ModelReply(
            data["message"]["content"],
            data.get("prompt_eval_count", 0),
            data.get("eval_count", 0),
        )


def load_model(spec: str) -> ModelClient:
    """Build a client from a spec like 'anthropic:claude-sonnet-4-5' or 'ollama:llama3.1'."""
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicModel(model) if model else AnthropicModel()
    if provider == "openai":
        return OpenAIModel(model) if model else OpenAIModel()
    if provider == "ollama":
        return OllamaModel(model) if model else OllamaModel()
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
