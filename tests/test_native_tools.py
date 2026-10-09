"""Native tool calling in each client, tested offline with fake SDKs and a fake Ollama server."""

import io
import json
import sys
import types
from types import SimpleNamespace as NS

import pytest

from agent_eval.models import (
    AnthropicModel,
    GenerationParams,
    OllamaModel,
    OpenAIModel,
    ScriptedModel,
    ToolSpec,
    load_model,
    to_anthropic_messages,
    to_ollama_messages,
    to_openai_messages,
)
from agent_eval.tools import Workspace, build_tools

TOOLS = [ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}})]

# A neutral conversation with one tool round-trip.
CONVERSATION = [
    {"role": "user", "content": "what is in a.txt?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "read_file", "args": {"path": "a.txt"}}]},
    {"role": "tool", "tool_call_id": "t1", "name": "read_file", "content": "hello"},
]


class Recorder:
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


def test_tool_spec_from_tool():
    spec = build_tools(Workspace())["write_file"].spec()
    assert spec.parameters["required"] == ["path", "content"]
    assert spec.parameters["properties"]["path"]["type"] == "string"
    assert spec.parameters["additionalProperties"] is False


def test_message_conversions():
    anthropic = to_anthropic_messages(CONVERSATION)
    assert anthropic[1]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.txt"}}
    ]
    assert anthropic[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "hello"}],
    }

    openai = to_openai_messages("SYS", CONVERSATION)
    assert openai[0] == {"role": "system", "content": "SYS"}
    assert openai[2]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a.txt"}'}
    assert openai[3] == {"role": "tool", "tool_call_id": "t1", "content": "hello"}

    ollama = to_ollama_messages("SYS", CONVERSATION)
    assert ollama[2]["tool_calls"] == [{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]
    assert ollama[3] == {"role": "tool", "tool_name": "read_file", "content": "hello"}


def test_parallel_tool_results_share_one_anthropic_user_message():
    msgs = [
        *CONVERSATION,
        {"role": "tool", "tool_call_id": "t2", "name": "read_file", "content": "bye"},
    ]
    out = to_anthropic_messages(msgs)
    assert len(out) == 3 and len(out[2]["content"]) == 2


def test_anthropic_native_tool_call(monkeypatch):
    endpoint = Recorder(
        NS(
            content=[
                NS(type="text", text="let me look"),
                NS(type="tool_use", id="t9", name="read_file", input={"path": "a"}),
            ],
            usage=NS(input_tokens=30, output_tokens=8),
        )
    )
    fake_sdk(monkeypatch, "anthropic", "Anthropic", NS(messages=endpoint))
    model = AnthropicModel("claude-test", params=GenerationParams(temperature=0.2, max_tokens=50))
    reply = model.complete_with_tools("SYS", CONVERSATION[:1], TOOLS)
    assert reply.text == "let me look"
    assert [(c.id, c.name, c.args) for c in reply.tool_calls] == [("t9", "read_file", {"path": "a"})]
    assert endpoint.kwargs["tools"][0]["input_schema"] == TOOLS[0].parameters
    assert endpoint.kwargs["temperature"] == 0.2 and endpoint.kwargs["max_tokens"] == 50


def test_openai_native_tool_call_with_bad_arguments(monkeypatch):
    calls = [
        NS(id="a", function=NS(name="read_file", arguments='{"path": "x"}')),
        NS(id="b", function=NS(name="read_file", arguments="{not json")),
    ]
    endpoint = Recorder(
        NS(choices=[NS(message=NS(content=None, tool_calls=calls))], usage=NS(prompt_tokens=5, completion_tokens=2))
    )
    fake_sdk(monkeypatch, "openai", "OpenAI", NS(chat=NS(completions=endpoint)))
    reply = OpenAIModel("gpt-test").complete_with_tools("SYS", CONVERSATION, TOOLS)
    assert reply.tool_calls[0].args == {"path": "x"} and reply.tool_calls[0].error is None
    assert reply.tool_calls[1].args == {} and "not valid JSON" in reply.tool_calls[1].error
    assert endpoint.kwargs["tools"][0]["function"]["name"] == "read_file"
    assert endpoint.kwargs["messages"][3]["role"] == "tool"


def test_ollama_native_tool_call(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data)
        message = {"content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]}
        return io.BytesIO(json.dumps({"message": message, "prompt_eval_count": 3, "eval_count": 1}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    reply = OllamaModel("llama-test", host="http://box:11434").complete_with_tools("SYS", CONVERSATION[:1], TOOLS)
    assert reply.tool_calls[0].name == "read_file" and reply.tool_calls[0].id.startswith("call_")
    assert sent["body"]["tools"][0]["type"] == "function"
    assert sent["body"]["options"] == {"temperature": 0.7, "num_predict": 1024}


def test_generation_params_validation_and_load_model():
    with pytest.raises(ValueError):
        GenerationParams(temperature=3)
    with pytest.raises(ValueError):
        GenerationParams(max_tokens=0)
    model = load_model("ollama:qwen", GenerationParams(temperature=0.0))
    assert model.params.temperature == 0.0


def test_models_without_native_support_say_so():
    with pytest.raises(NotImplementedError, match="--protocol json"):
        ScriptedModel([]).complete_with_tools("", [], TOOLS)
