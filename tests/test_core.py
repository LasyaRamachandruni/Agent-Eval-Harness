import json

import pytest

from agent_eval.agent import parse_action, run_agent
from agent_eval.models import ScriptedModel
from agent_eval.runner import run_task, summarize
from agent_eval.tasks import Task, load_tasks
from agent_eval.tools import ToolError, Workspace, build_tools, calculator


def call(tool, **args):
    return json.dumps({"thought": "", "tool": tool, "args": args})


def final(answer):
    return json.dumps({"thought": "", "final": answer})


# --- tools ------------------------------------------------------------------

def test_calculator_basic():
    assert calculator("(1234 * 56) - 789") == "68315"
    assert calculator("7 / 2") == "3.5"


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "1 +", "1/0"])
def test_calculator_rejects_bad_input(expr):
    with pytest.raises(ToolError):
        calculator(expr)


def test_tool_argument_validation():
    tools = build_tools(Workspace())
    with pytest.raises(ToolError):
        tools["read_file"].call({})
    with pytest.raises(ToolError):
        tools["list_files"].call({"bogus": 1})


# --- parsing ----------------------------------------------------------------

def test_parse_action_handles_code_fences():
    text = 'Sure!\n```json\n{"final": "42"}\n```'
    assert parse_action(text) == {"final": "42"}


# --- agent loop -------------------------------------------------------------

def test_agent_reads_file_and_answers():
    ws = Workspace({"a.txt": "hello"})
    model = ScriptedModel([call("read_file", path="a.txt"), final("hello")])
    res = run_agent(model, "what is in a.txt?", build_tools(ws))
    assert res.final_answer == "hello"
    assert res.stop_reason == "final"
    assert res.steps[0].observation == "hello"


def test_agent_recovers_from_bad_format_and_tool_errors():
    ws = Workspace()
    model = ScriptedModel(["not json", call("read_file", path="nope.txt"), final("NOT FOUND")])
    res = run_agent(model, "x", build_tools(ws))
    assert res.final_answer == "NOT FOUND"
    assert res.tool_errors == 2


def test_agent_stops_at_max_steps():
    model = ScriptedModel([call("list_files")] * 5)
    res = run_agent(model, "x", build_tools(Workspace()), max_steps=3)
    assert res.stop_reason == "max_steps"
    assert res.num_steps == 3


# --- tasks and runner -------------------------------------------------------

def test_bundled_tasks_load():
    tasks = load_tasks("tasks")
    assert len(tasks) >= 5
    assert all(t.checks for t in tasks)


def test_task_pass_and_fail():
    task = Task.from_dict(
        {
            "id": "t",
            "prompt": "copy",
            "files": {"src.txt": "x"},
            "checks": [
                {"type": "file_equals", "path": "dst.txt", "value": "x"},
                {"type": "file_unchanged", "path": "src.txt"},
            ],
        }
    )
    good = ScriptedModel([call("write_file", path="dst.txt", content="x"), final("done")])
    assert run_task(good, task).passed

    bad = ScriptedModel([call("write_file", path="src.txt", content="changed"), final("done")])
    assert not run_task(bad, task).passed


def test_summary_counts():
    task = Task.from_dict({"id": "t", "prompt": "p", "checks": [{"type": "answer_equals", "value": "1"}]})
    runs = [run_task(ScriptedModel([final("1")]), task), run_task(ScriptedModel([final("2")]), task)]
    s = summarize(runs)
    assert s["tasks"] == 2 and s["passed"] == 1 and s["success_rate"] == 0.5
