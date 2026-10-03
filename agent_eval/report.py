"""Compare saved runs: a leaderboard of models, and a static HTML report.

Every `agent-eval run` writes a directory with a `summary.json`. This module
reads a folder of those directories and lines the runs up side by side:

    runs = collect_runs("results/")
    rows = leaderboard(runs)          # one row per model, best first
    page = render_html(runs)          # self-contained HTML comparing them

By default only the newest run of each model is kept, so re-running a model
replaces its old row instead of crowding the board.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .failures import LABELS


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


# ---------------------------------------------------------------- HTML report

CSS = """
:root { --bg: #ffffff; --fg: #1d2330; --muted: #5d6675; --line: #e3e6eb; --head: #f4f6f9;
        --good: #1f8a4c; --mid: #b7791f; --bad: #c53030; --bar: #3b6fd8; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171c; --fg: #e6e9ee; --muted: #9aa3b2; --line: #2b313b; --head: #1c2027;
          --good: #48bb78; --mid: #ecc94b; --bad: #fc8181; --bar: #6b9bff; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg);
       font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; }
p.sub, p.note { color: var(--muted); margin: 0 0 8px; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th { background: var(--head); font-weight: 600; }
tr:last-child td { border-bottom: none; }
th:first-child, td:first-child, .leaderboard td:nth-child(2), .leaderboard th:nth-child(2) { text-align: left; }
.bar { display: inline-block; width: 80px; height: 8px; margin-left: 8px; border-radius: 4px;
       background: var(--line); vertical-align: middle; overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--bar); }
.good { color: var(--good); } .mid { color: var(--mid); } .bad { color: var(--bad); }
.na { color: var(--muted); }
summary { cursor: pointer; color: var(--muted); margin-bottom: 8px; }
"""


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _pct(x: float | None, bar: bool = False) -> str:
    if x is None:
        return '<span class="na">n/a</span>'
    cls = "good" if x >= 0.9 else "mid" if x >= 0.6 else "bad"
    text = f'<span class="{cls}">{x:.0%}</span>'
    if bar:
        text += f'<span class="bar"><i style="width:{x * 100:.0f}%"></i></span>'
    return text


def _usd(x: float | None, digits: int = 5) -> str:
    return '<span class="na">n/a</span>' if x is None else f"${x:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    """An HTML table; cells are already-escaped HTML."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    attr = f' class="{cls}"' if cls else ""
    return f'<div class="scroll"><table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _leaderboard_section(rows: list[dict]) -> str:
    headers = ["#", "Model", "Success", "pass^k", "Consistent", "Injection resistance",
               "Avg steps", "Avg tokens", "Cost / run", "Total cost"]
    body = [
        [
            str(r["rank"]),
            f'<span title="{_esc(r["run"])}">{_esc(r["model"])}</span>',
            _pct(r["success_rate"], bar=True),
            f'{_pct(r["pass_hat_k"])} <span class="na">k={r["repeats"]}</span>',
            f'{r["consistent_tasks"]}/{r["tasks"]}',
            _pct(r["injection_resistance"], bar=r["injection_resistance"] is not None),
            f'{r["avg_steps"]:.1f}',
            f'{r["avg_tokens"]:.0f}',
            _usd(r["avg_cost_per_run_usd"]),
            _usd(r["total_cost_usd"], 4),
        ]
        for r in rows
    ]
    return "<h2>Leaderboard</h2>" + _table(headers, body, cls="leaderboard")


def _category_section(runs: list[SavedRun]) -> str:
    cats = sorted({c for r in runs for c in r.summary.get("by_category", {})})
    if not cats:
        return ""
    body = []
    for cat in cats:
        row = [_esc(cat)]
        for r in runs:
            c = r.summary.get("by_category", {}).get(cat)
            if c:
                row.append(f'{_pct(c["success_rate"])} <span class="na">{c["passed"]}/{c["runs"]}</span>')
            else:
                row.append(_pct(None))
        body.append(row)
    return "<h2>Success by category</h2>" + _table(["Category"] + [_esc(r.model) for r in runs], body)


def _failure_section(runs: list[SavedRun]) -> str:
    counts = [(r.summary.get("failures") or {}).get("counts", {}) for r in runs]
    labels = [lab for lab in LABELS if any(lab in c for c in counts)]
    labels += sorted({lab for c in counts for lab in c} - set(labels))  # e.g. "unlabelled"
    if not labels:
        return "<h2>Failure breakdown</h2><p class=\"note\">No failed runs.</p>"
    body = [[_esc(lab)] + [str(c.get(lab, 0)) if c.get(lab) else '<span class="na">0</span>' for c in counts]
            for lab in labels]
    body.append(["<b>total failed</b>"] + [f"<b>{(r.summary.get('failures') or {}).get('failed_runs', 0)}</b>" for r in runs])
    return ("<h2>Failure breakdown</h2><p class=\"note\">Failed runs by cause "
            "(see the failure taxonomy in the README).</p>" + _table(["Label"] + [_esc(r.model) for r in runs], body))


def _task_cell(t: dict | None) -> str:
    if t is None:
        return '<span class="na">not run</span>'
    rate = t["success_rate"]
    cls = "good" if rate == 1 else "bad" if rate == 0 else "mid"
    cell = f'<span class="{cls}">{t["passed"]}/{t["trials"]}</span>'
    if t.get("resisted") is not None and t["resisted"] < t["trials"]:
        cell += ' <span class="bad">hijacked</span>'
    elif t.get("failures"):
        cell += f' <span class="na">{_esc(", ".join(t["failures"]))}</span>'
    return cell


def _task_section(runs: list[SavedRun]) -> str:
    """Task x model grid: passed trials per task, with the failure label when it failed."""
    tasks: dict[str, str] = {}
    for r in runs:
        for tid, t in r.summary.get("per_task", {}).items():
            tasks.setdefault(tid, t.get("category", ""))
    if not tasks:
        return ""
    body = [
        [f"{_esc(tid)} <span class=\"na\">{_esc(cat)}</span>"]
        + [_task_cell(r.summary.get("per_task", {}).get(tid)) for r in runs]
        for tid, cat in sorted(tasks.items(), key=lambda kv: (kv[1], kv[0]))
    ]
    return (
        f"<h2>Per-task results</h2><details open><summary>{len(tasks)} tasks: trials passed per model, "
        "with the failure label (or <span class=\"bad\">hijacked</span>) when a task failed</summary>"
        + _table(["Task"] + [_esc(r.model) for r in runs], body)
        + "</details>"
    )


def render_html(runs: list[SavedRun], title: str = "Agent Eval Report", latest_only: bool = True) -> str:
    """A complete, self-contained HTML page comparing the given runs.

    The page has no scripts and no external resources, so it can be opened
    from disk, attached to an email or published as a static file.
    """
    if latest_only:
        runs = latest_per_model(runs)
    rows = leaderboard(runs, latest_only=False)
    ordered = [next(r for r in runs if r.name == row["run"]) for row in rows]  # columns in rank order
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = sum(r.summary["runs"] for r in ordered)
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>",
        f"<h1>{_esc(title)}</h1>",
        f'<p class="sub">{len(ordered)} model(s), {total} runs. Generated {stamp}.</p>',
    ]
    if not ordered:
        parts.append('<p class="note">No saved runs found.</p>')
    else:
        parts += [_leaderboard_section(rows), _category_section(ordered), _failure_section(ordered),
                  _task_section(ordered)]
        parts.append(
            '<p class="note">Success: share of runs that passed. pass^k: chance that all k trials of a task pass. '
            "Injection resistance: share of injection runs where the agent did not follow the hidden "
            "instructions. Costs are estimates from a per-model price table.</p>"
        )
    parts.append("</main></body></html>")
    return "\n".join(parts)
