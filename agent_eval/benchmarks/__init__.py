"""Benchmarks the runner can evaluate, besides the built-in task suite.

    builtin    the 40 tasks in tasks/ (files, reasoning, multi-step, robustness, injection)
    hotpotqa   200 seeded HotpotQA dev (distractor) questions, exact match + F1
    humaneval  164 HumanEval problems, pass@1 with sandboxed hidden tests

Data for hotpotqa and humaneval is downloaded with `agent-eval download`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tasks import load_tasks
from . import hotpotqa, humaneval
from .download import DATASETS, HOTPOT_SUBSET

BENCHMARKS = ("builtin", "hotpotqa", "humaneval")


class MissingDataError(FileNotFoundError):
    """The benchmark's data has not been downloaded yet."""


def _require(path: Path, name: str) -> Path:
    if not path.exists():
        raise MissingDataError(f"{name} data not found at {path}; run: agent-eval download {name}")
    return path


def load_benchmark(
    name: str, data_dir: str | Path = "data", tasks_dir: str | Path = "tasks", limit: int | None = None
) -> list[Any]:
    """The evaluation items of benchmark `name` (optionally only the first `limit`)."""
    data_dir = Path(data_dir)
    if name == "builtin":
        return load_tasks(tasks_dir)[:limit]
    if name == "hotpotqa":
        return hotpotqa.load_items(_require(data_dir / HOTPOT_SUBSET, name), limit)
    if name == "humaneval":
        return humaneval.load_items(_require(data_dir / DATASETS["humaneval"].filename, name), limit)
    raise ValueError(f"unknown benchmark {name!r} (choose from: {', '.join(BENCHMARKS)})")


__all__ = ["BENCHMARKS", "MissingDataError", "load_benchmark"]
