"""Offline tests for the model clients: fake SDKs and a fake Ollama server, no API calls."""

import io
import json
import sys
import types
from types import SimpleNamespace as NS

import pytest

from agent_eval.models import AnthropicModel, OllamaModel, OpenAIModel, ScriptedModel, load_model

MESSAGES = [{"role": "user", "content": "hi"}]


class Recorder:
    """Stands in for an SDK endpoint: remembers the request and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_sdk(monkeypatch, name, client_class, client):
    module = types.ModuleType(name)
    setattr(module, client_class, lambda: client)
    monkeypatch.setitem(sys.modules, name, module)


def test_anthropic_client(monkeypatch):
    endpoint = Recorder(NS(
        content=[NS(type="text", text='{"final": '), NS(type="tool_use"), NS(type="text", text='"4"}')],
        usage=NS(input_tokens=12, output_tokens=3),
    ))
    fake_sdk(monkeypatch, "anthropic", "Anthropic", NS(messages=endpoint))
    model = AnthropicModel("claude-test")
    reply = model.complete("SYS", MESSAGES)
    assert model.name == "anthropic:claude-test"
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ('{"final": "4"}', 12, 3)
    assert endpoint.kwargs["system"] == "SYS"
    assert endpoint.kwargs["messages"] == MESSAGES
    assert endpoint.kwargs["temperature"] == 0


def test_openai_client(monkeypatch):
    endpoint = Recorder(NS(
        choices=[NS(message=NS(content='{"final": "4"}'))],
        usage=NS(prompt_tokens=20, completion_tokens=5),
    ))
    fake_sdk(monkeypatch, "openai", "OpenAI", NS(chat=NS(completions=endpoint)))
    reply = OpenAIModel("gpt-test").complete("SYS", MESSAGES)
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ('{"final": "4"}', 20, 5)
    assert endpoint.kwargs["messages"] == [{"role": "system", "content": "SYS"}, *MESSAGES]


def test_openai_client_handles_empty_content_and_missing_usage(monkeypatch):
    endpoint = Recorder(NS(choices=[NS(message=NS(content=None))], usage=None))
    fake_sdk(monkeypatch, "openai", "OpenAI", NS(chat=NS(completions=endpoint)))
    reply = OpenAIModel().complete("SYS", MESSAGES)
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("", 0, 0)


def test_ollama_client(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"], sent["body"] = req.full_url, json.loads(req.data)
        payload = {"message": {"content": '{"final": "4"}'}, "prompt_eval_count": 30, "eval_count": 6}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    reply = OllamaModel("llama-test", host="http://box:11434/").complete("SYS", MESSAGES)
    assert sent["url"] == "http://box:11434/api/chat"
    assert sent["body"]["model"] == "llama-test"
    assert sent["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent["body"]["stream"] is False
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ('{"final": "4"}', 30, 6)


def test_load_model_specs(monkeypatch):
    assert load_model("ollama:qwen2.5").name == "ollama:qwen2.5"
    assert load_model("ollama").name == "ollama:llama3.1"
    with pytest.raises(ValueError, match="unknown model provider"):
        load_model("mystery:model")


def test_scripted_model_finishes_when_out_of_replies():
    model = ScriptedModel(['{"final": "a"}'])
    assert model.complete("", []).text == '{"final": "a"}'
    assert "ran out of replies" in model.complete("", []).text
