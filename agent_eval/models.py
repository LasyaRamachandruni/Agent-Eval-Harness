"""Model clients.

The agent talks to every model through the same tiny interface:
    complete(system, messages) -> ModelReply

Messages are plain {"role": "user"|"assistant", "content": str} dicts, so any
chat model works. Provider SDKs are imported lazily, so you only need the one
you actually use.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass
class ModelReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ModelClient:
    name: str = "base"

    def complete(self, system: str, messages: list[dict]) -> ModelReply:  # pragma: no cover
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
