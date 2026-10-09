"""Run a benchmark concurrently, safely and resumably.

`run_suite_async` is what `agent-eval run` uses:

* Concurrency: up to `concurrency` runs in flight at once (asyncio + worker
  threads, since provider SDK calls block).
* Rate limits: the model is wrapped in `RetryingModel` (exponential backoff,
  honours retry-after) by `prepare_model`.
* Caching: and in `CachingModel`, keyed by (model, prompt, params, sample),
  so re-running or resuming never pays twice for the same call.
* Resume: every finished run is appended to runs.jsonl immediately (one line,
  flushed and fsynced). After a crash or Ctrl-C, `--resume <run dir>` loads
  the finished (task, trial) pairs and runs only what is missing. The run's
  configuration is saved in meta.json and must match on resume.
* Cost: billed tokens (cache misses) are priced as runs finish; with a budget,
  no new runs start once it is spent and the run can be resumed later.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .cache import CachingModel, ResponseCache
from .defenses import Defense
from .logs import get_logger
from .models import ModelClient
from .pricing import estimate_cost
from .resilience import RetryingModel
from .runner import TaskRun, run_task
from .strategies import Strategy

log = get_logger(__name__)


class ResumeError(RuntimeError):
    """The run directory cannot be resumed with this configuration."""


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a run; saved as meta.json and checked on resume."""

    model: str
    strategy: str
    benchmark: str
    repeats: int
    params: dict[str, Any]
    defenses: list[str] = field(default_factory=list)
    items: int = 0
    harness_version: str = __version__

    def comparable(self) -> dict[str, Any]:
        """The fields that must match for a resume to be valid (not the version)."""
        data = asdict(self)
        data.pop("harness_version")
        return data


def git_commit(cwd: str | Path = ".") -> str | None:
    """The current git commit, if the harness runs from a checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class ResultLog:
    """Append-only runs.jsonl plus meta.json in one run directory."""

    def __init__(self, run_dir: str | Path, config: RunConfig):
        self.run_dir = Path(run_dir)
        self.config = config
        self.path = self.run_dir / "runs.jsonl"
        meta_path = self.run_dir / "meta.json"
        if meta_path.exists():
            saved = json.loads(meta_path.read_text())
            saved_config = {k: v for k, v in saved.get("config", {}).items() if k != "harness_version"}
            if saved_config != config.comparable():
                diff = sorted(k for k in config.comparable() if saved_config.get(k) != config.comparable()[k])
                raise ResumeError(f"{self.run_dir} was started with a different configuration ({', '.join(diff)})")
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            meta = {"config": asdict(config), "git_commit": git_commit()}
            meta_path.write_text(json.dumps(meta, indent=2))

    def completed(self) -> list[TaskRun]:
        """Runs already saved; a half-written last line (from a crash) is ignored."""
        if not self.path.exists():
            return []
        runs = []
        for n, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                runs.append(TaskRun.from_record(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning("skipping unreadable result line", extra={"fields": {"line": n, "error": str(exc)}})
        return runs

    def rewrite(self, runs: list[TaskRun]) -> None:
        """Replace runs.jsonl with exactly `runs` (drops a torn line before appending)."""
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(asdict(r)) + "\n" for r in runs))
        tmp.replace(self.path)

    def append(self, run: TaskRun) -> None:
        """Write one finished run durably."""
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(run)) + "\n")
            f.flush()
            os.fsync(f.fileno())


def prepare_model(
    model: ModelClient, cache_path: str | Path | None, max_retries: int = 6
) -> tuple[ModelClient, ResponseCache | None]:
    """Wrap a model with retries and (if `cache_path` is given) the response cache."""
    wrapped: ModelClient = RetryingModel(model, max_retries=max_retries)
    cache = None
    if cache_path is not None:
        cache = ResponseCache(cache_path)
        wrapped = CachingModel(wrapped, cache)
    return wrapped, cache


@dataclass
class Progress:
    """Live totals while a suite runs."""

    total: int
    done: int = 0
    skipped: int = 0
    spent_usd: float = 0.0
    unpriced: bool = False


async def run_suite_async(
    model: ModelClient,
    items: list[Any],
    *,
    strategy: Strategy,
    repeats: int = 1,
    concurrency: int = 4,
    defenses: list[Defense] | None = None,
    benchmark: str = "builtin",
    result_log: ResultLog | None = None,
    on_run: Callable[[TaskRun, Progress], None] | None = None,
    budget_usd: float | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
) -> list[TaskRun]:
    """Run every item `repeats` times with up to `concurrency` runs in flight.

    Returns all runs (resumed and new) in item order, then trial order.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    order = {item.id: i for i, item in enumerate(items)}
    previous = result_log.completed() if result_log else []
    previous = [r for r in previous if r.task_id in order and r.trial < repeats]
    if result_log and previous:
        result_log.rewrite(previous)
        log.info("resuming", extra={"fields": {"finished": len(previous), "run_dir": str(result_log.run_dir)}})
    done = {(r.task_id, r.trial) for r in previous}
    todo = [(item, t) for item in items for t in range(repeats) if (item.id, t) not in done]
    progress = Progress(total=len(items) * repeats, done=len(previous))
    for r in previous:
        cost = estimate_cost(r.model, r.agent.billed_input_tokens, r.agent.billed_output_tokens, prices)
        progress.spent_usd += cost or 0.0
    semaphore = asyncio.Semaphore(concurrency)
    new_runs: list[TaskRun] = []

    async def one(item: Any, trial: int) -> None:
        async with semaphore:
            if budget_usd is not None and progress.spent_usd >= budget_usd:
                progress.skipped += 1
                return
            run = await asyncio.to_thread(run_task, model, item, trial, defenses, strategy, benchmark)
            cost = estimate_cost(run.model, run.agent.billed_input_tokens, run.agent.billed_output_tokens, prices)
            if cost is None:
                progress.unpriced = True
            progress.spent_usd += cost or 0.0
            progress.done += 1
            if result_log:
                result_log.append(run)
            new_runs.append(run)
            if on_run:
                on_run(run, progress)

    await asyncio.gather(*(one(item, t) for item, t in todo))
    if progress.skipped:
        log.warning(
            "budget reached; runs skipped",
            extra={"fields": {"skipped": progress.skipped, "spent_usd": round(progress.spent_usd, 4)}},
        )
    runs = previous + new_runs
    runs.sort(key=lambda r: (order[r.task_id], r.trial))
    return runs


def run_suite_concurrently(model: ModelClient, items: list[Any], **kwargs: Any) -> list[TaskRun]:
    """Synchronous wrapper around `run_suite_async`."""
    return asyncio.run(run_suite_async(model, items, **kwargs))
