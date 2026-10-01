"""Tasks and checks.

A task is a JSON file:

    {
      "id": "sum-expenses",
      "category": "files",
      "prompt": "What is the total of the amounts in expenses.csv?",
      "files": {"expenses.csv": "item,amount\\ncoffee,4\\nlunch,12\\n"},
      "checks": [{"type": "answer_equals", "value": "16"}],
      "max_steps": 8
    }

A task passes only if every check passes.

Tasks may also carry an optional "solution": a list of actions (the same JSON
objects a model would send) that solves the task. The test suite replays each
solution to prove every bundled task is actually passable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .agent import AgentResult
from .tools import Workspace


@dataclass
class CheckResult:
    type: str
    passed: bool
    detail: str = ""


# Required keys for each check type (besides "type").
CHECK_FIELDS: dict[str, tuple[str, ...]] = {
    "answer_equals": ("value",),
    "answer_contains": ("value",),
    "answer_not_contains": ("value",),
    "file_equals": ("path", "value"),
    "file_contains": ("path", "value"),
    "file_not_contains": ("path", "value"),
    "file_absent": ("path",),
    "file_unchanged": ("path",),
    "max_steps": ("value",),
}

TASK_FIELDS = {"id", "prompt", "category", "files", "checks", "max_steps", "solution"}


def validate_task(d: dict) -> list[str]:
    """Return a list of problems with a task dict (empty if it is valid)."""
    if not isinstance(d, dict):
        return ["task must be a JSON object"]
    problems = []
    tid = d.get("id", "?")
    for key in ("id", "prompt"):
        if not isinstance(d.get(key), str) or not d.get(key, "").strip():
            problems.append(f"{tid}: '{key}' must be a non-empty string")
    unknown = set(d) - TASK_FIELDS
    if unknown:
        problems.append(f"{tid}: unknown field(s): {', '.join(sorted(unknown))}")
    if "category" in d and not (isinstance(d["category"], str) and d["category"]):
        problems.append(f"{tid}: 'category' must be a non-empty string")
    files = d.get("files", {})
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        problems.append(f"{tid}: 'files' must map file names to string contents")
    max_steps = d.get("max_steps", 10)
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        problems.append(f"{tid}: 'max_steps' must be a positive integer")
    checks = d.get("checks")
    if not isinstance(checks, list) or not checks:
        problems.append(f"{tid}: needs a non-empty 'checks' list")
        checks = []
    for i, c in enumerate(checks):
        kind = c.get("type") if isinstance(c, dict) else None
        if kind not in CHECK_FIELDS:
            problems.append(f"{tid}: check {i} has unknown type {kind!r}")
            continue
        missing = [k for k in CHECK_FIELDS[kind] if k not in c]
        if missing:
            problems.append(f"{tid}: check {i} ({kind}) is missing {', '.join(missing)}")
        if kind == "file_unchanged" and isinstance(files, dict) and c.get("path") not in files:
            problems.append(f"{tid}: check {i} (file_unchanged) refers to a file the task does not provide")
    solution = d.get("solution")
    if solution is not None and not (
        isinstance(solution, list) and all(isinstance(a, dict) for a in solution)
    ):
        problems.append(f"{tid}: 'solution' must be a list of action objects")
    return problems


@dataclass
class Task:
    id: str
    prompt: str
    category: str = "general"
    files: dict[str, str] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    max_steps: int = 10
    solution: list[dict] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        problems = validate_task(d)
        if problems:
            raise ValueError("invalid task: " + "; ".join(problems))
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            category=d.get("category", "general"),
            files=dict(d.get("files", {})),
            checks=list(d["checks"]),
            max_steps=int(d.get("max_steps", 10)),
            solution=d.get("solution"),
        )


def load_tasks(path: str | Path) -> list[Task]:
    """Load one task file, or every *.json file under a directory."""
    path = Path(path)
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    tasks = []
    for f in files:
        try:
            tasks.append(Task.from_dict(json.loads(f.read_text())))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{f}: {exc}") from exc
    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate task ids: {', '.join(sorted(dupes))}")
    return tasks


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def run_check(check: dict, result: AgentResult, ws: Workspace) -> CheckResult:
    kind = check.get("type")
    answer = result.final_answer or ""

    if kind == "answer_equals":
        ok = _norm(answer) == _norm(str(check["value"]))
        return CheckResult(kind, ok, f"expected {check['value']!r}, got {answer!r}")

    if kind == "answer_contains":
        ok = _norm(str(check["value"])) in _norm(answer)
        return CheckResult(kind, ok, f"expected answer to contain {check['value']!r}")

    if kind == "answer_not_contains":
        ok = _norm(str(check["value"])) not in _norm(answer)
        return CheckResult(kind, ok, f"answer must not contain {check['value']!r}")

    if kind == "file_equals":
        actual = ws.files.get(check["path"])
        ok = actual is not None and actual.strip() == str(check["value"]).strip()
        return CheckResult(kind, ok, f"{check['path']} = {actual!r}")

    if kind == "file_contains":
        actual = ws.files.get(check["path"]) or ""
        ok = str(check["value"]) in actual
        return CheckResult(kind, ok, f"expected {check['path']} to contain {check['value']!r}")

    if kind == "file_not_contains":
        actual = ws.files.get(check["path"]) or ""
        ok = str(check["value"]) not in actual
        return CheckResult(kind, ok, f"{check['path']} must not contain {check['value']!r}")

    if kind == "file_absent":
        ok = check["path"] not in ws.files
        return CheckResult(kind, ok, f"{check['path']} must not exist")

    if kind == "file_unchanged":
        # Compared against the task's starting files by the runner.
        ok = ws.files.get(check["path"]) == check.get("_original")
        return CheckResult(kind, ok, f"{check['path']} must not be modified")

    if kind == "max_steps":
        ok = result.num_steps <= int(check["value"])
        return CheckResult(kind, ok, f"used {result.num_steps} steps (limit {check['value']})")

    return CheckResult(str(kind), False, f"unknown check type: {kind}")
