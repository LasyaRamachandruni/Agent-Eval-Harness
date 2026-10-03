"""Offline demo: scripted agents that need no API key.

Two pretend agents replay actions stored in the task files:

- scripted:careful  replays each task's reference `solution`.
- scripted:gullible does the same, except on injection tasks, where it replays
                    the `attack` (the actions of an agent that obeys the hidden
                    instructions).

Their results go through the real runner, grader and failure taxonomy and are
saved like any other run, so `agent-eval report` has something to compare. The
numbers show what the harness measures; they say nothing about real models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .models import ScriptedModel
from .runner import TaskRun, run_task, save_results
from .tasks import Task


def _careful(task: Task) -> list[dict]:
    return task.solution or []


def _gullible(task: Task) -> list[dict]:
    if task.is_injection and task.injection.get("attack"):
        return task.injection["attack"]
    return task.solution or []


DEMO_AGENTS: dict[str, Callable[[Task], list[dict]]] = {
    "scripted:careful": _careful,
    "scripted:gullible": _gullible,
}


def run_demo_agent(name: str, tasks: list[Task]) -> list[TaskRun]:
    """Run one demo agent over `tasks`, with a fresh scripted model per task."""
    actions_for = DEMO_AGENTS[name]
    runs = []
    for task in tasks:
        model = ScriptedModel([json.dumps(a) for a in actions_for(task)], name=name)
        runs.append(run_task(model, task))
    return runs


def run_demo(tasks: list[Task], out_dir: str | Path) -> list[Path]:
    """Run every demo agent and save each one's results under `out_dir`."""
    return [save_results(run_demo_agent(name, tasks), out_dir) for name in DEMO_AGENTS]
