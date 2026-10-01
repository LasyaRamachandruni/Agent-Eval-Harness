"""Schema validation for task files, plus a check that every bundled task is solvable."""

import json

import pytest

from agent_eval.models import ScriptedModel
from agent_eval.runner import run_task
from agent_eval.tasks import Task, load_tasks, validate_task

VALID = {
    "id": "ok",
    "prompt": "do it",
    "files": {"a.txt": "x"},
    "checks": [{"type": "file_unchanged", "path": "a.txt"}],
}


def test_valid_task_has_no_problems():
    assert validate_task(VALID) == []


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"id": ""}, "'id'"),
        ({"prompt": None}, "'prompt'"),
        ({"checks": []}, "checks"),
        ({"checks": [{"type": "answer_is"}]}, "unknown type"),
        ({"checks": [{"type": "file_equals", "path": "a.txt"}]}, "missing value"),
        ({"checks": [{"type": "file_unchanged", "path": "b.txt"}]}, "does not provide"),
        ({"files": {"a.txt": 3}}, "'files'"),
        ({"max_steps": 0}, "max_steps"),
        ({"surprise": 1}, "unknown field"),
        ({"solution": "read it"}, "solution"),
        ({"injection": "bad"}, "'injection'"),
        ({"injection": {"checks": [{"type": "file_absent", "path": "x"}]}}, "goal"),
        ({"injection": {"goal": "g", "checks": []}}, "injection needs"),
        ({"injection": {"goal": "g", "checks": [{"type": "nope"}]}}, "injection check 0"),
        ({"injection": {"goal": "g", "checks": [{"type": "file_absent", "path": "x"}], "attack": "x"}}, "attack"),
        ({"injection": {"goal": "g", "checks": [{"type": "file_absent", "path": "x"}], "extra": 1}}, "unknown injection"),
    ],
)
def test_invalid_tasks_are_rejected(change, fragment):
    bad = {**VALID, **change}
    problems = validate_task(bad)
    assert any(fragment in p for p in problems), problems
    with pytest.raises(ValueError):
        Task.from_dict(bad)


def test_load_tasks_names_the_bad_file(tmp_path):
    (tmp_path / "broken.json").write_text(json.dumps({"id": "x", "prompt": "p", "checks": []}))
    with pytest.raises(ValueError, match="broken.json"):
        load_tasks(tmp_path)


BUNDLED = load_tasks("tasks")


@pytest.mark.parametrize("task", BUNDLED, ids=[t.id for t in BUNDLED])
def test_bundled_task_solution_passes(task):
    assert task.solution, f"{task.id} has no reference solution"
    replies = [json.dumps(a) for a in task.solution]
    run = run_task(ScriptedModel(replies), task)
    failed = [f"{c.type}: {c.detail}" for c in run.checks if not c.passed]
    assert run.passed, failed


def test_suite_covers_every_category():
    assert len(BUNDLED) >= 25
    categories = {t.category for t in BUNDLED}
    assert {"files", "reasoning", "multi-step", "robustness"} <= categories
