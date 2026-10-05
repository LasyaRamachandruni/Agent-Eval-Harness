"""Before/after comparison of two saved runs, e.g. a model without and with a defense.

    before = find_saved_run("openai:gpt-4o", "results/")
    after = find_saved_run("openai:gpt-4o +hardened_prompt", "results/")
    diff = compare_runs(before.summary, after.summary)

The headline numbers (success, consistency, injection resistance, steps,
tokens, cost) are compared as a whole, and tasks are compared one by one to
show exactly which ones a change fixed or broke. A defense that stops
hijacks but also breaks ordinary tasks shows up in both lists.
"""

from __future__ import annotations

from pathlib import Path

from .report import SavedRun, collect_runs

# (key, label, kind): kind "rate" is shown as a percentage, "num" as a number, "usd" as dollars.
METRICS = (
    ("success_rate", "success", "rate"),
    ("pass_hat_k", "pass^k", "rate"),
    ("injection_resistance", "injection resistance", "rate"),
    ("avg_steps", "avg steps", "num"),
    ("avg_tokens", "avg tokens", "num"),
    ("avg_cost_per_run_usd", "cost per run", "usd"),
)


def find_saved_run(ref: str, results: str | Path = "results") -> SavedRun:
    """Resolve `ref` to a saved run.

    `ref` may be a run directory (anything holding a summary.json) or a run
    label such as "openai:gpt-4o +hardened_prompt", in which case the newest
    run with that label under `results` is used.
    """
    path = Path(ref)
    if (path / "summary.json").exists():
        runs = collect_runs(path)
        if runs:
            return runs[0]
    matches = [r for r in collect_runs(results) if r.model == ref]
    if not matches:
        raise LookupError(f"no saved run found for {ref!r} (not a run directory, and no run with that label in {results})")
    return matches[-1]  # collect_runs returns oldest first


def headline(summary: dict) -> dict:
    """The comparable headline numbers of one summary (None when not measured)."""
    k = summary.get("repeats", 1)
    inj = summary.get("injection") or {}
    return {
        "success_rate": summary.get("success_rate"),
        "pass_hat_k": (summary.get("pass_hat_k") or {}).get(str(k), summary.get("success_rate")),
        "injection_resistance": inj.get("resistance_rate"),
        "avg_steps": summary.get("avg_steps"),
        "avg_tokens": round(summary.get("avg_input_tokens", 0) + summary.get("avg_output_tokens", 0), 1),
        "avg_cost_per_run_usd": summary.get("avg_cost_per_run_usd"),
    }


def _hijacked(t: dict) -> bool:
    return t.get("resisted") is not None and t["resisted"] < t["trials"]


def compare_runs(before: dict, after: dict) -> dict:
    """Compare two summaries (as saved in summary.json).

    Returns the headline numbers of each side with their change, plus the
    tasks both runs share, split into: fixed (success rate went up), broken
    (went down), newly_resisted / newly_hijacked (injection tasks whose
    hijacked status changed). Tasks only one side ran are listed separately.
    """
    b, a = headline(before), headline(after)
    metrics = {}
    for key, _, _ in METRICS:
        change = None if b[key] is None or a[key] is None else round(a[key] - b[key], 6)
        metrics[key] = {"before": b[key], "after": a[key], "change": change}

    bt, at = before.get("per_task", {}), after.get("per_task", {})
    shared = sorted(set(bt) & set(at))
    return {
        "before": before.get("label", before.get("model")),
        "after": after.get("label", after.get("model")),
        "metrics": metrics,
        "shared_tasks": len(shared),
        "fixed": [t for t in shared if at[t]["success_rate"] > bt[t]["success_rate"]],
        "broken": [t for t in shared if at[t]["success_rate"] < bt[t]["success_rate"]],
        "newly_resisted": [t for t in shared if _hijacked(bt[t]) and not _hijacked(at[t])],
        "newly_hijacked": [t for t in shared if not _hijacked(bt[t]) and _hijacked(at[t])],
        "only_before": sorted(set(bt) - set(at)),
        "only_after": sorted(set(at) - set(bt)),
    }


def _fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "rate":
        return f"{value:.0%}"
    if kind == "usd":
        return f"${value:.5f}"
    return f"{value:g}"


def _fmt_change(change, kind: str) -> str:
    if change is None:
        return "-"
    if change == 0:
        return "="
    if kind == "rate":
        return f"{change * 100:+.0f} pts"
    if kind == "usd":
        return f"{'+' if change > 0 else '-'}${abs(change):.5f}"
    return f"{change:+g}"


def format_comparison(diff: dict) -> str:
    """Plain-text before/after table and task lists for the terminal."""
    lines = [f"before: {diff['before']}", f"after:  {diff['after']}", ""]
    lines.append(f"{'metric':<22}{'before':>10}{'after':>10}{'change':>12}")
    for key, label, kind in METRICS:
        m = diff["metrics"][key]
        lines.append(f"{label:<22}{_fmt(m['before'], kind):>10}{_fmt(m['after'], kind):>10}"
                     f"{_fmt_change(m['change'], kind):>12}")
    lines.append("")
    lines.append(f"{diff['shared_tasks']} tasks in both runs")
    for key, label in (("fixed", "fixed"), ("broken", "broken"),
                       ("newly_resisted", "no longer hijacked"), ("newly_hijacked", "newly hijacked")):
        tasks = diff[key]
        lines.append(f"{label} ({len(tasks)}){': ' + ', '.join(tasks) if tasks else ''}")
    if diff["only_before"] or diff["only_after"]:
        lines.append(f"note: {len(diff['only_before'])} task(s) only in before, "
                     f"{len(diff['only_after'])} only in after; headline numbers cover different task sets")
    return "\n".join(lines)


def format_comparison_markdown(diff: dict) -> str:
    """The same comparison as Markdown, for a write-up such as docs/BENCHMARK.md."""
    lines = [f"**Before:** `{diff['before']}`  ", f"**After:** `{diff['after']}`", "",
             "| Metric | Before | After | Change |", "| --- | ---: | ---: | ---: |"]
    for key, label, kind in METRICS:
        m = diff["metrics"][key]
        lines.append(f"| {label} | {_fmt(m['before'], kind)} | {_fmt(m['after'], kind)} | "
                     f"{_fmt_change(m['change'], kind)} |")
    lines += ["", f"{diff['shared_tasks']} tasks in both runs.", ""]
    for key, label in (("fixed", "Fixed"), ("broken", "Broken"),
                       ("newly_resisted", "No longer hijacked"), ("newly_hijacked", "Newly hijacked")):
        tasks = diff[key]
        lines.append(f"- {label} ({len(tasks)}){': ' + ', '.join(f'`{t}`' for t in tasks) if tasks else ''}")
    if diff["only_before"] or diff["only_after"]:
        lines += ["", f"_Note: {len(diff['only_before'])} task(s) only in before, {len(diff['only_after'])} only "
                      "in after; headline numbers cover different task sets._"]
    return "\n".join(lines) + "\n"
