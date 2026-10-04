"""Offline demo: scripted agents that need no API key.

Two pretend agents replay actions stored in the task files:

- scripted:careful  replays each task's reference `solution`.
- scripted:gullible does the same, except on injection tasks, where it replays
                    the `attack` (the actions of an agent that obeys the hidden
                    instructions).

Their results go through the real runner, grader and failure taxonomy and are
saved like any other run, so `agent-eval report` has something to compare. The
numbers show what the harness measures; they say nothing about real models.

With defenses (`agent-eval demo --defense all`), the gullible agent checks its
system prompt: if any defense's instructions are there, it replays the
solution instead of the attack. That stands in for a model the defense fully
works on, so `agent-eval compare` has a before/after pair to show; real models
are only partly helped, which is what the real benchmark measures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .defenses import Defense
from .models import ModelClient, ModelReply, ScriptedModel
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


class DefenseAwareModel(ModelClient):
    """Replays `defended` if the system prompt carries any defense's instructions, else `undefended`."""

    def __init__(self, undefended: list[dict], defended: list[dict], defenses: list[Defense], name: str):
        self.name = name
        self._choices = (undefended, defended)
        self._markers = [d.system_suffix for d in defenses if d.system_suffix]
        self._script: ScriptedModel | None = None

    def complete(self, system: str, messages: list[dict]) -> ModelReply:
        if self._script is None:
            defended = bool(self._markers) and any(m in system for m in self._markers)
            actions = self._choices[defended]
            self._script = ScriptedModel([json.dumps(a) for a in actions], name=self.name)
        return self._script.complete(system, messages)


def run_demo_agent(name: str, tasks: list[Task], defenses: list[Defense] | None = None) -> list[TaskRun]:
    """Run one demo agent over `tasks`, with a fresh scripted model per task."""
    actions_for = DEMO_AGENTS[name]
    runs = []
    for task in tasks:
        model = DefenseAwareModel(actions_for(task), _careful(task), defenses or [], name=name)
        runs.append(run_task(model, task, defenses=defenses))
    return runs


def run_demo(tasks: list[Task], out_dir: str | Path, defenses: list[Defense] | None = None) -> list[Path]:
    """Run every demo agent (with the given defenses) and save each one's results under `out_dir`."""
    return [save_results(run_demo_agent(name, tasks, defenses), out_dir) for name in DEMO_AGENTS]
