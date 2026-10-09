"""Run a set of tasks against a model and write the results."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import AgentResult
from .cache import sample_scope
from .defenses import Defense
from .environment import EvalItem
from .failures import LABELS, classify_failure
from .grading import CheckResult
from .metrics import success_variance, suite_pass_at_k, suite_pass_hat_k
from .models import ModelClient
from .pricing import estimate_cost
from .strategies import Strategy, get_strategy


@dataclass
class TaskRun:
    """One graded attempt at one task: the outcome, the checks and the agent's trace."""

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
    # Names of the injection defenses the agent ran with (empty = baseline).
    defenses: list[str] = field(default_factory=list)
    # Benchmark metrics that are not pass/fail, e.g. {"em": 1.0, "f1": 1.0} for HotpotQA.
    scores: dict[str, float] = field(default_factory=dict)
    strategy: str = "react"
    benchmark: str = "builtin"

    @classmethod
    def from_record(cls, d: dict[str, Any]) -> TaskRun:
        """Rebuild a run from one line of runs.jsonl (used to resume and to re-summarise)."""
        data = dict(d)
        data["checks"] = [CheckResult(**c) for c in data.get("checks", [])]
        data["injection_checks"] = [CheckResult(**c) for c in data.get("injection_checks", [])]
        data["agent"] = AgentResult.from_dict(data["agent"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def run_label(model: str, defenses: list[str] | None, strategy: str = "react") -> str:
    """How a run is named in summaries and reports: the model, its strategy (if not react) and any defenses."""
    label = model if strategy == "react" else f"{model} [{strategy}]"
    return f"{label} +{'+'.join(defenses)}" if defenses else label


def run_task(
    model: ModelClient,
    task: EvalItem,
    trial: int = 0,
    defenses: list[Defense] | None = None,
    strategy: Strategy | None = None,
    benchmark: str = "builtin",
) -> TaskRun:
    """Run one task (or benchmark item) with `strategy` (default: react) and grade it.

    For injection tasks, a run passes only if it completes the task AND resists
    the attack; `resisted` records the second part on its own. Failed runs are
    labelled with a failure category (`failure`, `failure_reason`). `defenses`
    are passed to the agent (see defenses.py). Model calls are cached under
    sample index `trial`, so repeated trials are independent samples.
    """
    defenses = defenses or []
    strategy = strategy or get_strategy("react")
    env = task.make_env()
    with sample_scope(trial):
        result = strategy.run(model, task.prompt, env, max_steps=task.max_steps, defenses=defenses)
    grade = task.grade(result, env)
    run = TaskRun(
        task.id,
        task.category,
        model.name,
        grade.passed,
        grade.checks,
        result,
        trial,
        grade.resisted,
        grade.injection_checks,
        defenses=[d.name for d in defenses],
        scores=dict(grade.scores),
        strategy=strategy.label,
        benchmark=benchmark,
    )
    label = classify_failure(run)
    if label:
        run.failure, run.failure_reason = label
    return run


def run_suite(
    model: ModelClient,
    tasks: list[Any],
    repeats: int = 1,
    on_run: Callable[[TaskRun], None] | None = None,
    defenses: list[Defense] | None = None,
    strategy: Strategy | None = None,
    benchmark: str = "builtin",
) -> list[TaskRun]:
    """Run every task `repeats` times, one at a time. `on_run` is called after each run.

    For concurrency, rate-limit handling, caching and resume, see
    `executor.run_suite_async`.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    runs = []
    for task in tasks:
        for trial in range(repeats):
            run = run_task(model, task, trial, defenses, strategy, benchmark)
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
    known = [c for c in costs if c is not None]
    if len(known) != len(costs) or not known:
        return None
    return round(sum(known) / len(known), 6)


def _mean_scores(runs: list[TaskRun]) -> dict[str, float]:
    """Average of each benchmark score over the runs that report it."""
    totals: dict[str, list[float]] = {}
    for r in runs:
        for key, value in r.scores.items():
            totals.setdefault(key, []).append(float(value))
    return {key: round(sum(v) / len(v), 4) for key, v in totals.items()}


def _billed_cost(runs: list[TaskRun], prices: dict | None) -> float | None:
    """What the runs actually cost: tokens of calls not served from the response cache."""
    costs = [estimate_cost(r.model, r.agent.billed_input_tokens, r.agent.billed_output_tokens, prices) for r in runs]
    known = [c for c in costs if c is not None]
    if len(known) != len(costs):
        return None
    return round(sum(known), 6)


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
        by_task.setdefault(r.task_id, []).append(bool(r.resisted))
    resisted = sum(bool(r.resisted) for r in inj)
    return {
        "tasks": len(by_task),
        "runs": len(inj),
        "resisted": resisted,
        "resistance_rate": round(resisted / len(inj), 3),
        "resisted_every_trial": sum(all(v) for v in by_task.values()),
        "hijacked_tasks": sorted(t for t, v in by_task.items() if not all(v)),
        "completed_and_resisted": sum(r.passed for r in inj),
    }


def failure_summary(runs: list[TaskRun]) -> dict:
    """Count failed runs by failure label, overall and per category.

    Labels are listed most common first; labels that never occurred are left out.
    `examples` gives one task id per label so a failure is easy to look up.
    """
    failed = [r for r in runs if not r.passed]
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    by_cat: dict[str, dict[str, int]] = {}
    for r in failed:
        label = r.failure or "unlabelled"
        counts[label] = counts.get(label, 0) + 1
        examples.setdefault(label, r.task_id)
        cat = by_cat.setdefault(r.category, {})
        cat[label] = cat.get(label, 0) + 1
    order = {label: i for i, label in enumerate(LABELS)}

    def ranked(d: dict[str, int]) -> dict[str, int]:
        return dict(sorted(d.items(), key=lambda kv: (-kv[1], order.get(kv[0], len(order)))))

    return {
        "failed_runs": len(failed),
        "counts": ranked(counts),
        "examples": examples,
        "by_category": {c: ranked(d) for c, d in sorted(by_cat.items())},
    }


def summarize(runs: list[TaskRun], prices: dict | None = None) -> dict:
    """Aggregate runs into success, consistency, efficiency, cost, injection and failure numbers.

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
        if rs[0].scores:
            per_task[tid]["scores"] = _mean_scores(rs)
        if rs[0].resisted is not None:
            per_task[tid]["resisted"] = sum(bool(r.resisted) for r in rs)
        labels = [r.failure for r in rs if r.failure]
        if labels:
            per_task[tid]["failures"] = {lab: labels.count(lab) for lab in dict.fromkeys(labels)}

    by_cat: dict[str, list[TaskRun]] = {}
    for r in runs:
        by_cat.setdefault(r.category, []).append(r)

    return {
        "model": runs[0].model,
        "defenses": list(runs[0].defenses),
        "label": run_label(runs[0].model, runs[0].defenses, runs[0].strategy),
        "strategy": runs[0].strategy,
        "benchmark": runs[0].benchmark,
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
        "model_calls": sum(r.agent.model_calls for r in runs),
        "cached_calls": sum(r.agent.cached_calls for r in runs),
        "billed_cost_usd": _billed_cost(runs, prices),
        "scores": _mean_scores(runs),
        "avg_cost_per_run_usd": avg_cost,
        "total_cost_usd": None if avg_cost is None else round(avg_cost * n, 6),
        "injection": injection_summary(runs),
        "failures": failure_summary(runs),
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


def slugify(label: str) -> str:
    """A label made safe for a directory name, e.g. "openai:gpt-4o [reflexion]" -> "openai_gpt-4o_reflexion"."""
    slug = label.replace(":", "_").replace("/", "_").replace(" ", "")
    slug = re.sub(r"[^A-Za-z0-9._+-]+", "_", slug).strip("_")
    return slug or "run"


def new_run_dir(out_dir: str | Path, label: str) -> Path:
    """Create `<out_dir>/<UTC timestamp>_<label slug>` and return it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(out_dir) / f"{stamp}_{slugify(label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_summary(run_dir: str | Path, runs: list[TaskRun], prices: dict | None = None) -> dict:
    """Write summary.json for `runs` into `run_dir` and return the summary."""
    summary = summarize(runs, prices)
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def save_results(runs: list[TaskRun], out_dir: str | Path, prices: dict | None = None) -> Path:
    """Write one JSONL line per task (with full trace) plus a summary.json."""
    label = run_label(runs[0].model, runs[0].defenses, runs[0].strategy) if runs else "none"
    run_dir = new_run_dir(out_dir, label)
    with open(run_dir / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(asdict(r)) + "\n")
    write_summary(run_dir, runs, prices)
    return run_dir


def find_run_dir(path: str | Path) -> Path:
    """Resolve `path` to a run directory: itself if it holds runs.jsonl, else its newest subdirectory that does."""
    path = Path(path)
    if (path / "runs.jsonl").exists():
        return path
    candidates = sorted(p.parent for p in path.glob("*/runs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no runs.jsonl found in {path}")
    return candidates[-1]  # directory names start with a UTC timestamp


def load_run_records(run_dir: str | Path) -> list[dict]:
    """Read the saved runs (one dict per run, as written by save_results)."""
    lines = (Path(run_dir) / "runs.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]
