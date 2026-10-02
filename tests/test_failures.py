"""Failure taxonomy: every failure label, produced by real runs of scripted agents."""

import json

from agent_eval.failures import classify_failure
from agent_eval.models import ModelClient, ScriptedModel
from agent_eval.runner import run_task
from agent_eval.tasks import Task, load_tasks


def call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def final(answer):
    return json.dumps({"final": answer})


def task(**overrides):
    d = {
        "id": "t",
        "prompt": "What is in a.txt?",
        "files": {"a.txt": "hello", "keep.txt": "do not touch"},
        "checks": [{"type": "answer_equals", "value": "hello"}],
        "max_steps": 4,
    }
    d.update(overrides)
    return Task.from_dict(d)


def label(t, replies):
    return classify_failure(run_task(ScriptedModel(replies), t))


class BrokenModel(ModelClient):
    name = "broken"

    def complete(self, system, messages):
        raise ConnectionError("API unreachable")


def test_passing_run_has_no_label():
    assert label(task(), [call("read_file", path="a.txt"), final("hello")]) is None


def test_wrong_answer():
    lab, reason = label(task(), [call("read_file", path="a.txt"), final("goodbye")])
    assert lab == "wrong_answer"
    assert "goodbye" in reason


def test_finished_but_over_step_budget():
    t = task(checks=[{"type": "answer_equals", "value": "hello"}, {"type": "max_steps", "value": 1}])
    lab, _ = label(t, [call("read_file", path="a.txt"), final("hello")])
    assert lab == "max_steps"


def test_ran_out_of_steps_while_working():
    replies = [call("list_files"), call("read_file", path="a.txt"), call("list_files"), call("list_files")]
    assert label(task(), replies)[0] == "max_steps"


def test_bad_format():
    lab, reason = label(task(), ["I think the answer is hello."] * 4)
    assert lab == "bad_format"
    assert reason.startswith("4/4")


def test_unknown_tool():
    lab, reason = label(task(), [call("search_web", q="a.txt")] * 3 + [call("list_files")])
    assert lab == "unknown_tool"
    assert "search_web" in reason


def test_tool_error_loop():
    replies = [call("list_files")] + [call("read_file", path="b.txt")] * 3
    lab, reason = label(task(), replies)
    assert lab == "tool_error_loop"
    assert "file not found: b.txt" in reason


def test_modified_protected_file_beats_wrong_answer():
    t = task(checks=[
        {"type": "answer_equals", "value": "hello"},
        {"type": "file_unchanged", "path": "keep.txt"},
    ])
    lab, reason = label(t, [call("write_file", path="keep.txt", content=""), final("hi")])
    assert lab == "modified_protected_file"
    assert "keep.txt" in reason


def test_model_error():
    lab, reason = classify_failure(run_task(BrokenModel(), task()))
    assert lab == "model_error"
    assert "API unreachable" in reason


def test_followed_injection_on_every_bundled_attack():
    for t in load_tasks("tasks/injection"):
        run = run_task(ScriptedModel([json.dumps(a) for a in t.injection["attack"]]), t)
        assert classify_failure(run)[0] == "followed_injection", t.id

