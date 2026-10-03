"""Compare saved runs: a leaderboard of models, and a static HTML report.

Every `agent-eval run` writes a directory with a `summary.json`. This module
reads a folder of those directories and lines the runs up side by side:

    runs = collect_runs("results/")
    rows = leaderboard(runs)          # one row per model, best first

By default only the newest run of each model is kept, so re-running a model
replaces its old row instead of crowding the board.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SavedRun:
    """One run directory: its name (timestamp_model) and its summary.json."""

    name: str
    path: Path
    summary: dict

    @property
    def model(self) -> str:
        return self.summary.get("model", "unknown")


def collect_runs(path: str | Path) -> list[SavedRun]:
    """Load every run under `path`, oldest first.

    `path` may be a single run directory or a folder of them. Directories
    without a readable summary.json are skipped.
    """
    path = Path(path)
    dirs = [path] if (path / "summary.json").exists() else sorted(p.parent for p in path.glob("*/summary.json"))
    runs = []
    for d in dirs:
        try:
            summary = json.loads((d / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("runs"):
            runs.append(SavedRun(d.name, d, summary))
    return runs


def latest_per_model(runs: list[SavedRun]) -> list[SavedRun]:
    """Keep only the newest run of each model (run names start with a UTC timestamp)."""
    newest: dict[str, SavedRun] = {}
    for r in sorted(runs, key=lambda r: r.name):
        newest[r.model] = r
    return list(newest.values())


def leaderboard_row(run: SavedRun) -> dict:
    """The headline numbers of one run, flattened for a table."""
    s = run.summary
    k = s.get("repeats", 1)
    inj = s.get("injection") or {}
    fails = s.get("failures") or {}
    return {
        "model": run.model,
        "run": run.name,
        "tasks": s["tasks"],
        "runs": s["runs"],
        "repeats": k,
        "success_rate": s["success_rate"],
        # pass^k for the run's own k: the share of tasks it can be trusted to pass every time.
        "pass_hat_k": s.get("pass_hat_k", {}).get(str(k), s["success_rate"]),
        "consistent_tasks": s.get("consistent_tasks", 0),
        "injection_resistance": inj.get("resistance_rate"),
        "injection_runs": inj.get("runs", 0),
        "avg_steps": s.get("avg_steps", 0.0),
        "avg_tokens": round(s.get("avg_input_tokens", 0) + s.get("avg_output_tokens", 0), 1),
        "avg_cost_per_run_usd": s.get("avg_cost_per_run_usd"),
        "total_cost_usd": s.get("total_cost_usd"),
        "failures": dict(fails.get("counts", {})),
    }


def _rank_key(row: dict) -> tuple:
    # Higher success first, then more dependable, then harder to hijack, then cheaper.
    resistance = row["injection_resistance"]
    cost = row["avg_cost_per_run_usd"]
    return (
        -row["success_rate"],
        -row["pass_hat_k"],
        -(resistance if resistance is not None else -1),
        cost if cost is not None else float("inf"),
        row["model"],
    )


def leaderboard(runs: list[SavedRun], latest_only: bool = True) -> list[dict]:
    """Leaderboard rows sorted best first, each with a 1-based `rank`.

    Ranking: success rate, then pass^k (consistency), then injection
    resistance, then cost per run (cheaper wins ties).
    """
    if latest_only:
        runs = latest_per_model(runs)
    rows = sorted((leaderboard_row(r) for r in runs), key=_rank_key)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows
