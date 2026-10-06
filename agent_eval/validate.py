"""Check task files before they go into a benchmark.

Loading a task already validates its schema (see tasks.validate_task). This
module goes one step further and replays the task's scripted actions through
the real runner and grader:

- the reference `solution` must pass every check (the task is solvable and
  its checks accept a correct answer);
- on injection tasks, the solution must also resist the attack, and the
  scripted `attack` must be caught (the injection checks really detect a
  hijacked agent).

A task with no solution is reported as a warning, not an error: it can still
be run, but nothing proves its checks are right.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import ScriptedModel
from .runner import TaskRun, run_task
from .tasks import Task, load_tasks


@dataclass
class TaskReport:
    """What replaying one task's scripted actions showed."""

    task_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing is wrong (warnings are allowed)."""
        return not self.errors


def _replay(task: Task, actions: list[dict]) -> TaskRun:
    return run_task(ScriptedModel([json.dumps(a) for a in actions], name="replay"), task)


def _failed_checks(run: TaskRun) -> str:
    bad = [f"{c.type} ({c.detail})" for c in run.checks + run.injection_checks if not c.passed]
    if run.agent.stop_reason != "final":
        bad.append(f"stopped with {run.agent.stop_reason}")
    return "; ".join(bad) or "unknown reason"


def check_task(task: Task) -> TaskReport:
    """Replay a task's solution (and attack, for injection tasks) and report problems."""
    report = TaskReport(task.id)
    if not task.solution:
        report.warnings.append("no reference solution, so its checks are untested")
    else:
        run = _replay(task, task.solution)
        if not run.passed:
            report.errors.append(f"reference solution fails: {_failed_checks(run)}")
    if task.is_injection:
        attack = task.injection.get("attack")
        if not attack:
            report.warnings.append("no scripted attack, so its injection checks are untested")
        elif _replay(task, attack).resisted is not False:
            report.errors.append("scripted attack is not caught by the injection checks")
    return report


def check_tasks(path: str | Path) -> list[TaskReport]:
    """Load every task under `path` (schema errors raise ValueError) and check each one."""
    return [check_task(t) for t in load_tasks(path)]
