"""Every strategy, in both protocols, driven by fake models."""

import json

import pytest

from agent_eval.agent import run_agent
from agent_eval.environment import WorkspaceEnv
from agent_eval.grounding import check_grounding, extract_facts, score_attempt
from agent_eval.models import ModelReply
from agent_eval.runner import run_task, summarize
from agent_eval.strategies import STRATEGIES, get_strategy
from agent_eval.tasks import Task
from agent_eval.tools import Workspace, build_tools
from tests.fakes import FunctionModel, act, final, is_review, native_call, turns

APPLES = Task.from_dict(
    {
        "id": "apples",
        "category": "files",
        "prompt": "How many apples are listed in fruit.txt? Reply with just the number.",
        "files": {"fruit.txt": "apples: 42\npears: 7\n"},
        "checks": [{"type": "answer_equals", "value": "42"}],
        "max_steps": 4,
    }
)

WRITE = Task.from_dict(
    {
        "id": "write-total",
        "category": "files",
        "prompt": "Write the number of apples in fruit.txt to total.txt, then reply done.",
        "files": {"fruit.txt": "apples: 42\n"},
        "checks": [{"type": "file_equals", "path": "total.txt", "value": "42"}],
        "max_steps": 5,
    }
)


def solver(answer="42"):
    """A JSON-protocol agent: read fruit.txt, then answer."""

    def fn(system, messages, tools):
        if is_review(system):
            return json.dumps({"verdict": "accept", "critique": "none", "score": 10, "reflection": ""})
        return act("read_file", path="fruit.txt") if turns(messages) == 0 else final(answer)

    return FunctionModel(fn, native=False)


def test_registry_and_validation():
    assert set(STRATEGIES) == {"direct", "react", "reflection", "reflexion"}
    with pytest.raises(ValueError, match="unknown strategy"):
        get_strategy("tree-of-thought")
    with pytest.raises(ValueError, match="unknown protocol"):
        get_strategy("react", protocol="xml")
    assert get_strategy("reflexion", protocol="native").label == "reflexion/native"
    assert get_strategy("direct", protocol="native").label == "direct"


def test_direct_sees_files_inline_and_answers_in_one_call():
    model = FunctionModel(lambda s, m, t: "42", native=False)
    run = run_task(model, APPLES, strategy=get_strategy("direct"))
    assert run.passed and run.agent.model_calls == 1 and run.strategy == "direct"
    assert "apples: 42" in model.calls[0]["messages"][0]["content"]
    assert model.calls[0]["tools"] is None


def test_direct_accepts_json_final_and_cannot_write_files():
    model = FunctionModel(lambda s, m, t: final("done"), native=False)
    run = run_task(model, WRITE, strategy=get_strategy("direct"))
    assert run.agent.final_answer == "done" and not run.passed and run.failure == "wrong_answer"


def test_direct_reports_model_errors():
    def boom(s, m, t):
        raise RuntimeError("down")

    run = run_task(FunctionModel(boom), APPLES, strategy=get_strategy("direct"))
    assert run.agent.stop_reason == "model_error" and run.failure == "model_error"


def test_react_json_protocol_matches_run_agent():
    run = run_task(solver(), APPLES, strategy=get_strategy("react"))
    assert run.passed and run.agent.num_steps == 2 and run.agent.strategy == "react"
    res = run_agent(solver(), APPLES.prompt, build_tools(Workspace(dict(APPLES.files))))
    assert res.final_answer == "42"


def native_solver(system, messages, tools):
    assert tools and {t.name for t in tools} >= {"read_file", "write_file"}
    if turns(messages) == 0:
        return native_call("read_file", path="fruit.txt")
    if turns(messages) == 1:
        assert messages[-1]["role"] == "tool" and "apples: 42" in messages[-1]["content"]
        return native_call("write_file", call_id="c2", path="total.txt", content="42")
    return ModelReply("done", 10, 5)


def test_react_native_protocol_runs_tools_and_finishes_on_text():
    model = FunctionModel(native_solver)
    run = run_task(model, WRITE, strategy=get_strategy("react", protocol="native"))
    assert run.passed and run.strategy == "react/native"
    assert [s.tool for s in run.agent.steps if s.action == "tool"] == ["read_file", "write_file"]
    assert "final answer as plain text" in model.calls[0]["system"]


def test_native_protocol_feeds_errors_back_and_handles_empty_replies():
    def fn(system, messages, tools):
        n = turns(messages)
        if n == 0:
            return native_call("delete_everything")
        if n == 1:
            assert messages[-1]["content"].startswith("Error: unknown tool")
            return ModelReply("", 1, 1)
        return ModelReply("42", 1, 1)

    run = run_task(FunctionModel(fn), APPLES, strategy=get_strategy("react", protocol="native"))
    assert run.passed and run.agent.tool_errors == 2


def test_native_protocol_needs_a_capable_model():
    with pytest.raises(ValueError, match="native tool calling"):
        run_task(solver(), APPLES, strategy=get_strategy("react", protocol="native"))


def test_reflection_revises_after_a_critique():
    def fn(system, messages, tools):
        if is_review(system):
            wrong = "'7'" in messages[0]["content"]
            verdict = (
                {"verdict": "revise", "critique": "You reported pears, not apples."} if wrong else {"verdict": "accept"}
            )
            return json.dumps(verdict)
        if turns(messages) == 0:
            return act("read_file", path="fruit.txt")
        if turns(messages) == 1:
            return final("7")
        assert "You reported pears" in messages[-1]["content"]
        return final("42")

    run = run_task(FunctionModel(fn, native=False), APPLES, strategy=get_strategy("reflection"))
    assert run.passed and run.agent.reflections == ["You reported pears, not apples."]
    assert [s.action for s in run.agent.steps] == ["tool", "final", "reflect", "final", "reflect"]
    assert run.agent.num_steps == 3 and run.agent.model_calls == 5


def test_reflection_stops_on_accept_or_unparseable_review():
    run = run_task(solver(), APPLES, strategy=get_strategy("reflection", rounds=3))
    assert run.passed and run.agent.model_calls == 3  # read, answer, one accepting review

    def fn(system, messages, tools):
        return "no json here" if is_review(system) else final("42")

    run = run_task(FunctionModel(fn, native=False), APPLES, strategy=get_strategy("reflection"))
    assert run.passed and run.agent.steps[-1].error.startswith("bad format")


def test_reflexion_retries_from_a_reset_workspace_with_lessons():
    attempts = []

    def fn(system, messages, tools):
        if is_review(system):
            if "'41'" in messages[0]["content"]:
                assert "Not found in the task or any tool result: 41" in messages[0]["content"]
                return json.dumps({"score": 2, "reflection": "Copy the number exactly from fruit.txt."})
            return json.dumps({"score": 10, "reflection": ""})
        if turns(messages) == 0:
            attempts.append(system)
            return (
                act("write_file", path="scratch.txt", content="x")
                if len(attempts) == 1
                else act("read_file", path="fruit.txt")
            )
        if turns(messages) == 1 and len(attempts) == 1:
            return final("41")
        if turns(messages) == 1:
            return act("write_file", path="total.txt", content="42")
        return final("42")

    fields = {k: v for k, v in WRITE.__dict__.items() if v is not None}
    task = Task.from_dict({**fields, "checks": [*WRITE.checks, {"type": "file_absent", "path": "scratch.txt"}]})
    run = run_task(FunctionModel(fn, native=False), task, strategy=get_strategy("reflexion", trials=3))
    assert run.passed and run.agent.attempts == 2
    assert "Copy the number exactly" in attempts[1] and "Copy the number" not in attempts[0]
    assert run.agent.reflections == ["Copy the number exactly from fruit.txt."]


def test_reflexion_keeps_the_best_attempt_when_a_retry_is_worse():
    def fn(system, messages, tools):
        if is_review(system):
            return json.dumps({"score": 3, "reflection": "try again"})  # 40 + 30 + 10 + 6 = 86 < 90
        first_attempt = "Lessons" not in system
        if turns(messages) == 0:
            return act("read_file", path="fruit.txt")
        return final("42") if first_attempt else act("read_file", path="missing.txt")

    run = run_task(FunctionModel(fn, native=False), APPLES, strategy=get_strategy("reflexion", trials=3))
    assert run.agent.attempts == 2 and run.agent.final_answer == "42" and run.passed
    assert run.agent.stop_reason == "final"


def test_reflexion_uses_the_environment_self_check_when_there_is_one():
    class Checked(WorkspaceEnv):
        def feedback(self, answer):
            return (1.0, "2/2 examples pass") if answer == "42" else (0.0, "0/2 examples pass")

    class CheckedTask(Task):
        def make_env(self):
            return Checked(self.files)

    task = CheckedTask(**APPLES.__dict__)
    prompts = []

    def fn(system, messages, tools):
        if is_review(system):
            prompts.append(messages[0]["content"])
            return json.dumps({"score": 10})
        return final("42")

    run = run_task(FunctionModel(fn, native=False), task, strategy=get_strategy("reflexion"))
    assert run.passed and run.agent.attempts == 1  # 40 + 30 + 10 + 20 = 100 >= accept score
    assert "Visible self-check: 2/2 examples pass" in prompts[0]


def test_reflexion_model_error_on_first_attempt():
    def boom(s, m, t):
        raise RuntimeError("down")

    run = run_task(FunctionModel(boom), APPLES, strategy=get_strategy("reflexion"))
    assert run.agent.stop_reason == "model_error" and not run.passed


@pytest.mark.parametrize("name", ["react", "reflection", "reflexion"])
def test_every_tool_strategy_works_in_both_protocols(name):
    for protocol, model in (("json", solver()), ("native", FunctionModel(native_solver))):
        task = APPLES if protocol == "json" else WRITE
        if protocol == "native" and name != "react":
            # Reviewers always answer in JSON; the actor uses native calls.
            def fn(system, messages, tools, _inner=native_solver):
                return (
                    json.dumps({"verdict": "accept", "score": 10})
                    if is_review(system)
                    else _inner(system, messages, tools)
                )

            model = FunctionModel(fn)
        run = run_task(model, task, strategy=get_strategy(name, protocol=protocol))
        assert run.passed, (name, protocol, run.failure_reason)
        summary = summarize([run])
        assert summary["strategy"] == run.strategy
        assert summary["label"] == ("fake:model" if run.strategy == "react" else f"fake:model [{run.strategy}]")


def test_grounding_and_score():
    assert extract_facts("The answer is 1,200 apples from Paris.") == ["1,200", "Paris"]
    report = check_grounding("1,200 apples from Paris, sold by Bob", ["We sold 1200.0 apples in Paris"])
    assert report.ungrounded == ["Bob"] and report.fraction == pytest.approx(2 / 3)
    assert check_grounding("yes", []).fraction == 1.0
    assert score_attempt(True, 1.0, 0, 10) == 100
    assert score_attempt(False, 0.5, 9, None) == 15
    assert score_attempt(True, 2.0, 0, 99) == 100  # clamped
