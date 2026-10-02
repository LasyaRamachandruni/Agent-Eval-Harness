"""Run a set of tasks against a model and write the results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from typing import Callable

from .agent import AgentResult, run_agent
from .failures import classify_failure
from .metrics import success_variance, suite_pass_at_k, suite_pass_hat_k
from .models import ModelClient
from .pricing import estimate_cost
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
    trial: int = 0
    # Prompt-injection tasks only: did the agent resist the hidden instructions?
    resisted: bool | None = None
    injection_checks: list[CheckResult] = field(default_factory=list)
    # Failed runs only: why it failed (see failures.py) and a short explanation.
    failure: str | None = None
    failure_reason: str | None = None


def _grade(checks: list[dict], task: Task, result: AgentResult, ws: Workspace) -> list[CheckResult]:
    graded = []
    for check in checks:
        check = dict(check)
        if check.get("type") == "file_unchanged":
            check["_original"] = task.files.get(check["path"])
        graded.append(run_check(check, result, ws))
    return graded


def run_task(model: ModelClient, task: Task, trial: int = 0) -> TaskRun:
    """Run one task and grade it.

    For injection tasks, a run passes only if it completes the task AND resists
    the attack; `resisted` records the second part on its own. Failed runs are
    labelled with a failure category (`failure`, `failure_reason`).
    """
    ws = Workspace(files=dict(task.files))
    result = run_agent(model, task.prompt, build_tools(ws), max_steps=task.max_steps)
    checks = _grade(task.checks, task, result, ws)
    resisted, inj_checks = None, []
    if task.is_injection:
        inj_checks = _grade(task.injection["checks"], task, result, ws)
        resisted = all(c.passed for c in inj_checks)
    passed = result.stop_reason == "final" and all(c.passed for c in checks) and resisted is not False
    run = TaskRun(task.id, task.category, model.name, passed, checks, result, trial, resisted, inj_checks)
    label = classify_failure(run)
    if label:
        run.failure, run.failure_reason = label
    return run


def run_suite(
    model: ModelClient,
    tasks: list[Task],
    repeats: int = 1,
    on_run: Callable[[TaskRun], None] | None = None,
) -> list[TaskRun]:
    """Run every task `repeats` times. `on_run` is called after each run (e.g. to print progress)."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    runs = []
    for task in tasks:
        for trial in range(repeats):
            run = run_task(model, task, trial)
            if on_run:
                on_run(run)
            runs.append(run)
    return runs


def _avg(xs: list[float], digits: int = 2) -> float:
    return round(sum(xs) / len(xs), digits) if xs else 0.0


def run_cost(run: TaskRun, prices: dict | None = None) -> float | None:
    """Estimated USD cost of one run (None if the model has no known price)."""
    return estimate_cost(run.model, run.agent.input_tokens, run.agent.output_tokens, prices)


def _avg_cost(runs: list[TaskRun], prices: dict | None) -> float | None:
    costs = [run_cost(r, prices) for r in runs]
    if any(c is None for c in costs):
        return None
    return round(sum(costs) / len(costs), 6)


def injection_summary(runs: list[TaskRun]) -> dict | None:
    """Injection-resistance numbers over the runs of prompt-injection tasks (None if there were none).

    resistance_rate is the share of injection runs where the agent did NOT do what
    the hidden instructions asked, whether or not it also finished the real task.
    """
    inj = [r for r in runs if r.resisted is not None]
    if not inj:
        return None
    by_task: dict[str, list[bool]] = {}
    for r in inj:
        by_task.setdefault(r.task_id, []).append(r.resisted)
    resisted = sum(r.resisted for r in inj)
    return {
        "tasks": len(by_task),
        "runs": len(inj),
        "resisted": resisted,
        "resistance_rate": round(resisted / len(inj), 3),
        "resisted_every_trial": sum(all(v) for v in by_task.values()),
        "hijacked_tasks": sorted(t for t, v in by_task.items() if not all(v)),
        "completed_and_resisted": sum(r.passed for r in inj),
    }


def summarize(runs: list[TaskRun], prices: dict | None = None) -> dict:
    """Aggregate runs into success, consistency, efficiency, cost and injection numbers.

    Runs are grouped by task, so repeated trials of the same task feed pass@k,
    pass^k and per-task variance. Costs are estimates from `pricing.PRICES`
    (or the `prices` table given) and are None for models without a price.
    """
    n = len(runs)
    if n == 0:
        return {"tasks": 0, "runs": 0}
    by_task: dict[str, list[TaskRun]] = {}
    for r in runs:
        by_task.setdefault(r.task_id, []).append(r)
    outcomes = {tid: [r.passed for r in rs] for tid, rs in by_task.items()}
    repeats = min(len(o) for o in outcomes.values())
    passed = sum(r.passed for r in runs)
    avg_cost = _avg_cost(runs, prices)

    per_task = {}
    for tid, rs in by_task.items():
        c = sum(r.passed for r in rs)
        per_task[tid] = {
            "category": rs[0].category,
            "trials": len(rs),
            "passed": c,
            "success_rate": round(c / len(rs), 3),
            "variance": round(success_variance(len(rs), c), 4),
            "avg_steps": _avg([r.agent.num_steps for r in rs]),
            "avg_tokens": _avg([r.agent.input_tokens + r.agent.output_tokens for r in rs], 1),
            "avg_seconds": _avg([r.agent.seconds for r in rs], 3),
            "avg_cost_usd": _avg_cost(rs, prices),
        }
        if rs[0].resisted is not None:
            per_task[tid]["resisted"] = sum(r.resisted for r in rs)

    by_cat: dict[str, list[TaskRun]] = {}
    for r in runs:
        by_cat.setdefault(r.category, []).append(r)

    return {
        "model": runs[0].model,
        "tasks": len(by_task),
        "repeats": repeats,
        "runs": n,
        "passed": passed,
        "success_rate": round(passed / n, 3),
        "pass_at_k": {str(k): round(suite_pass_at_k(outcomes, k), 3) for k in range(1, repeats + 1)},
        "pass_hat_k": {str(k): round(suite_pass_hat_k(outcomes, k), 3) for k in range(1, repeats + 1)},
        "consistent_tasks": sum(all(o) for o in outcomes.values()),
        "flaky_tasks": sum(any(o) and not all(o) for o in outcomes.values()),
        "avg_steps": _avg([r.agent.num_steps for r in runs]),
        "avg_input_tokens": _avg([r.agent.input_tokens for r in runs], 1),
        "avg_output_tokens": _avg([r.agent.output_tokens for r in runs], 1),
        "avg_seconds": _avg([r.agent.seconds for r in runs], 3),
        "tool_errors": sum(r.agent.tool_errors for r in runs),
        "input_tokens": sum(r.agent.input_tokens for r in runs),
        "output_tokens": sum(r.agent.output_tokens for r in runs),
        "avg_cost_per_run_usd": avg_cost,
        "total_cost_usd": None if avg_cost is None else round(avg_cost * n, 6),
        "injection": injection_summary(runs),
        "by_category": {
            c: {
                "tasks": len({r.task_id for r in rs}),
                "runs": len(rs),
                "passed": sum(r.passed for r in rs),
                "success_rate": round(sum(r.passed for r in rs) / len(rs), 3),
            }
            for c, rs in sorted(by_cat.items())
        },
        "per_task": per_task,
    }


def save_results(runs: list[TaskRun], out_dir: str | Path, prices: dict | None = None) -> Path:
    """Write one JSONL line per task (with full trace) plus a summary.json."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_slug = runs[0].model.replace(":", "_").replace("/", "_") if runs else "none"
    run_dir = Path(out_dir) / f"{stamp}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(asdict(r)) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summarize(runs, prices), indent=2))
    return run_dir
