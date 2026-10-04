"""Command line entry point.

    agent-eval run --model anthropic:claude-sonnet-4-5 --tasks tasks/
    agent-eval run --model ollama:llama3.1 --category injection
    agent-eval run --model ollama:llama3.1 --category injection --defense all
    agent-eval list --tasks tasks/
    agent-eval failures results/            # why did the latest run's tasks fail?
    agent-eval report results/              # leaderboard + results/report.html
    agent-eval compare MODEL "MODEL +hardened_prompt"   # before/after a defense
    agent-eval demo                         # offline scripted agents, no API key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .compare import compare_runs, find_saved_run, format_comparison
from .defenses import resolve_defenses
from .demo import run_demo
from .models import load_model
from .pricing import load_prices
from .report import collect_runs, leaderboard, render_html
from .runner import TaskRun, find_run_dir, load_run_records, run_suite, save_results, summarize
from .tasks import Task, load_tasks


def select_tasks(args: argparse.Namespace) -> list[Task]:
    """Load tasks and apply the --category / --only filters."""
    tasks = load_tasks(args.tasks)
    if getattr(args, "category", None):
        cats = set(args.category.split(","))
        tasks = [t for t in tasks if t.category in cats]
    if getattr(args, "only", None):
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t.id in wanted]
    if not tasks:
        raise SystemExit("no tasks matched the filters")
    return tasks


def _cmd_list(args: argparse.Namespace) -> int:
    for t in select_tasks(args):
        extra = f"  injection: {t.injection['goal']}" if t.is_injection else ""
        print(f"{t.id:<28} {t.category:<12} {len(t.checks)} check(s){extra}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    tasks = select_tasks(args)
    prices = load_prices(args.prices) if args.prices else None
    try:
        defenses = resolve_defenses(args.defense)
    except ValueError as exc:
        raise SystemExit(str(exc))
    model = load_model(args.model)
    if defenses:
        print(f"defenses: {', '.join(d.name for d in defenses)}")

    def show(run: TaskRun) -> None:
        mark = "PASS" if run.passed else "FAIL"
        if run.resisted is False:
            mark = "HIJACKED"
        trial = f" #{run.trial + 1}" if args.repeats > 1 else ""
        why = f"  ({run.failure})" if run.failure else ""
        print(f"[{mark}] {run.task_id + trial:<32} steps={run.agent.num_steps:<3} stop={run.agent.stop_reason}{why}")
        if not run.passed and args.verbose:
            print(f"        why: {run.failure_reason}")
            for c in run.checks + run.injection_checks:
                if not c.passed:
                    print(f"        - {c.type}: {c.detail}")

    runs = run_suite(model, tasks, repeats=args.repeats, on_run=show, defenses=defenses)
    s = summarize(runs, prices)
    out = save_results(runs, args.out, prices)
    print()
    print(format_summary(s))
    print(f"results: {out}")
    return 0


def _cmd_failures(args: argparse.Namespace) -> int:
    try:
        run_dir = find_run_dir(args.results)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    records = load_run_records(run_dir)
    failed = [r for r in records if not r["passed"]]
    if args.label:
        failed = [r for r in failed if r.get("failure") == args.label]
    print(f"{run_dir}: {sum(not r['passed'] for r in records)}/{len(records)} runs failed")
    if not failed:
        return 0
    groups: dict[str, list[dict]] = {}
    for r in failed:
        groups.setdefault(r.get("failure") or "unlabelled", []).append(r)
    for label, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{label} ({len(rs)})")
        for r in rs:
            trial = f" #{r['trial'] + 1}" if r.get("trial") else ""
            print(f"  {r['task_id'] + trial:<32} {r.get('failure_reason') or ''}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    runs = collect_runs(args.results)
    if not runs:
        raise SystemExit(f"no saved runs (summary.json) found in {args.results}")
    latest = not args.all
    rows = leaderboard(runs, latest_only=latest)
    print(format_leaderboard(rows))
    out = Path(args.out) if args.out else Path(args.results) / "report.html"
    out.write_text(render_html(runs, title=args.title, latest_only=latest))
    print(f"\nreport: {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        before = find_saved_run(args.before, args.results)
        after = find_saved_run(args.after, args.results)
    except LookupError as exc:
        raise SystemExit(str(exc))
    print(format_comparison(compare_runs(before.summary, after.summary)))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    tasks = select_tasks(args)
    try:
        defenses = resolve_defenses(args.defense)
    except ValueError as exc:
        raise SystemExit(str(exc))
    for path in run_demo(tasks, args.out, defenses):
        print(f"saved {path}")
    print(f"\nnext: agent-eval report {args.out}")
    if defenses:
        after = "scripted:gullible +" + "+".join(d.name for d in defenses)
        print(f'      agent-eval compare scripted:gullible "{after}" --results {args.out}')
    return 0


def format_leaderboard(rows: list[dict]) -> str:
    """Plain-text leaderboard for the terminal."""
    lines = [f"{'#':<3}{'model':<34}{'success':>8}{'pass^k':>8}{'inj.res':>8}{'cost/run':>11}"]
    for r in rows:
        inj = "-" if r["injection_resistance"] is None else f"{r['injection_resistance']:.0%}"
        cost = "-" if r["avg_cost_per_run_usd"] is None else f"${r['avg_cost_per_run_usd']:.5f}"
        lines.append(
            f"{r['rank']:<3}{r['model'][:33]:<34}{r['success_rate']:>8.0%}{r['pass_hat_k']:>8.0%}{inj:>8}{cost:>11}"
        )
    return "\n".join(lines)


def format_summary(s: dict) -> str:
    """Human-readable summary lines for the end of a run."""
    k = s["repeats"]
    lines = [
        f"{s['passed']}/{s['runs']} runs passed ({s['success_rate']:.0%}) "
        f"across {s['tasks']} tasks x {k} trial(s)",
    ]
    if k > 1:
        lines.append(
            f"pass@{k} {s['pass_at_k'][str(k)]:.0%}  pass^{k} {s['pass_hat_k'][str(k)]:.0%}  "
            f"consistent {s['consistent_tasks']}/{s['tasks']}  flaky {s['flaky_tasks']}"
        )
    lines.append(
        f"avg {s['avg_steps']} steps, {s['avg_input_tokens'] + s['avg_output_tokens']:.0f} tokens, "
        f"{s['avg_seconds']}s per run; {s['tool_errors']} tool errors"
    )
    inj = s.get("injection")
    if inj:
        line = (
            f"injection resistance {inj['resisted']}/{inj['runs']} runs ({inj['resistance_rate']:.0%}) "
            f"on {inj['tasks']} injection tasks"
        )
        if inj["hijacked_tasks"]:
            line += f"; hijacked by: {', '.join(inj['hijacked_tasks'])}"
        lines.append(line)
    fails = s.get("failures")
    if fails and fails["failed_runs"]:
        parts = ", ".join(f"{label} {n}" for label, n in fails["counts"].items())
        lines.append(f"failures: {parts}")
    if s.get("total_cost_usd") is not None:
        lines.append(f"estimated cost ${s['total_cost_usd']:.4f} (${s['avg_cost_per_run_usd']:.5f} per run)")
    else:
        lines.append("estimated cost: unknown for this model (pass --prices to set one)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent-eval", description="Evaluate LLM agents on tool-using tasks.")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="list tasks")
    pl.add_argument("--tasks", default="tasks")
    pl.add_argument("--category", help="comma-separated categories, e.g. injection")
    pl.set_defaults(fn=_cmd_list)

    pr = sub.add_parser("run", help="run tasks against a model")
    pr.add_argument("--model", required=True, help="e.g. anthropic:claude-sonnet-4-5, openai:gpt-4o-mini, ollama:llama3.1")
    pr.add_argument("--tasks", default="tasks")
    pr.add_argument("--category", help="comma-separated categories to run, e.g. injection")
    pr.add_argument("--only", help="comma-separated task ids to run")
    pr.add_argument("--repeats", type=int, default=1, help="run each task N times to measure consistency")
    pr.add_argument("--prices", help="JSON file of per-model prices, e.g. {\"openai:gpt-4o\": [2.5, 10]}")
    pr.add_argument("--defense", help="comma-separated injection defenses: hardened_prompt, tag_untrusted, all")
    pr.add_argument("--out", default="results")
    pr.add_argument("-v", "--verbose", action="store_true", help="show failed checks")
    pr.set_defaults(fn=_cmd_run)

    pf = sub.add_parser("failures", help="group the failed runs of a saved result by failure label")
    pf.add_argument("results", nargs="?", default="results", help="a run directory, or a folder of them (newest is used)")
    pf.add_argument("--label", help="only show runs with this failure label, e.g. wrong_answer")
    pf.set_defaults(fn=_cmd_failures)

    pp = sub.add_parser("report", help="compare saved runs: print a leaderboard and write an HTML report")
    pp.add_argument("results", nargs="?", default="results", help="a folder of run directories (or a single one)")
    pp.add_argument("-o", "--out", help="where to write the HTML (default: <results>/report.html)")
    pp.add_argument("--all", action="store_true", help="include every run, not just the newest per model")
    pp.add_argument("--title", default="Agent Eval Report")
    pp.set_defaults(fn=_cmd_report)

    pc = sub.add_parser("compare", help="compare two saved runs, e.g. before and after a defense")
    pc.add_argument("before", help="a run directory, or a run label such as openai:gpt-4o")
    pc.add_argument("after", help="a run directory, or a run label such as 'openai:gpt-4o +hardened_prompt'")
    pc.add_argument("--results", default="results", help="where to look up run labels (default: results)")
    pc.set_defaults(fn=_cmd_compare)

    pd = sub.add_parser("demo", help="run two scripted demo agents offline (no API key) to try the report")
    pd.add_argument("--tasks", default="tasks")
    pd.add_argument("--category", help="comma-separated categories to run")
    pd.add_argument("--defense", help="run the demo agents with these defenses (try: all)")
    pd.add_argument("--out", default="results/demo")
    pd.set_defaults(fn=_cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
