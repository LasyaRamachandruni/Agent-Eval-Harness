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


@dataclass
class Task:
    id: str
    prompt: str
    category: str = "general"
    files: dict[str, str] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    max_steps: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        if "id" not in d or "prompt" not in d:
            raise ValueError("task needs at least 'id' and 'prompt'")
        if not d.get("checks"):
            raise ValueError(f"task {d['id']} has no checks")
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            category=d.get("category", "general"),
            files=dict(d.get("files", {})),
            checks=list(d["checks"]),
            max_steps=int(d.get("max_steps", 10)),
        )


def load_tasks(path: str | Path) -> list[Task]:
    """Load one task file, or every *.json file under a directory."""
    path = Path(path)
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    tasks = [Task.from_dict(json.loads(f.read_text())) for f in files]
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

    if kind == "file_unchanged":
        # Compared against the task's starting files by the runner.
        ok = ws.files.get(check["path"]) == check.get("_original")
        return CheckResult(kind, ok, f"{check['path']} must not be modified")

    if kind == "max_steps":
        ok = result.num_steps <= int(check["value"])
        return CheckResult(kind, ok, f"used {result.num_steps} steps (limit {check['value']})")

    return CheckResult(str(kind), False, f"unknown check type: {kind}")
