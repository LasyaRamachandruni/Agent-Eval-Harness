"""Run a set of tasks against a model and write the results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentResult, run_agent
from .models import ModelClient
from .tasks import CheckResult, Task, run_check
from .tools import Workspace, build_tools


@dataclass
class TaskRun:
    task_id: str
    category: str
    model: str
    passed: bool
    checks: list[CheckResult]
    agent: AgentResult


def run_task(model: ModelClient, task: Task) -> TaskRun:
    ws = Workspace(files=dict(task.files))
    result = run_agent(model, task.prompt, build_tools(ws), max_steps=task.max_steps)
    checks = []
    for check in task.checks:
        check = dict(check)
        if check.get("type") == "file_unchanged":
            check["_original"] = task.files.get(check["path"])
        checks.append(run_check(check, result, ws))
    passed = result.stop_reason == "final" and all(c.passed for c in checks)
    return TaskRun(task.id, task.category, model.name, passed, checks, result)


def summarize(runs: list[TaskRun]) -> dict:
    n = len(runs)
    if n == 0:
        return {"tasks": 0}
    passed = sum(r.passed for r in runs)
    by_cat: dict[str, list[TaskRun]] = {}
    for r in runs:
        by_cat.setdefault(r.category, []).append(r)
    return {
        "model": runs[0].model,
        "tasks": n,
        "passed": passed,
        "success_rate": round(passed / n, 3),
        "avg_steps": round(sum(r.agent.num_steps for r in runs) / n, 2),
        "tool_errors": sum(r.agent.tool_errors for r in runs),
        "input_tokens": sum(r.agent.input_tokens for r in runs),
        "output_tokens": sum(r.agent.output_tokens for r in runs),
        "by_category": {
            c: {"tasks": len(rs), "passed": sum(r.passed for r in rs)} for c, rs in sorted(by_cat.items())
        },
    }


def save_results(runs: list[TaskRun], out_dir: str | Path) -> Path:
    """Write one JSONL line per task (with full trace) plus a summary.json."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = runs[0].model.replace(":", "_").replace("/", "_") if runs else "none"
    run_dir = Path(out_dir) / f"{stamp}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(asdict(r)) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summarize(runs), indent=2))
    return run_dir
