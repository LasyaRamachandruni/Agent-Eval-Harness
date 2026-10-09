"""Command line entry point.

agent-eval run --model anthropic:claude-sonnet-4-5 --tasks tasks/
agent-eval run --model ollama:llama3.1 --category injection
agent-eval run --model ollama:llama3.1 --category injection --defense all
agent-eval list --tasks tasks/
agent-eval failures results/            # why did the latest run's tasks fail?
agent-eval report results/              # leaderboard + results/report.html
agent-eval compare MODEL "MODEL +hardened_prompt"   # before/after a defense
agent-eval demo                         # offline scripted agents, no API key
agent-eval check --model openai:gpt-4o  # is the SDK installed and the key set?
agent-eval validate tasks/my_suite/     # schema + replay each task's solution
agent-eval download all                 # fetch HotpotQA and HumanEval (checksummed)
agent-eval estimate --model openai:gpt-4o-mini --benchmark humaneval --repeats 5
agent-eval run --model openai:gpt-4o-mini --benchmark hotpotqa --strategy reflexion --repeats 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .benchmarks import BENCHMARKS, MissingDataError, load_benchmark
from .benchmarks.download import DownloadError, download
from .compare import compare_runs, find_saved_run, format_comparison, format_comparison_markdown
from .config import ConfigError, Settings
from .defenses import resolve_defenses
from .demo import run_demo
from .estimate import estimate_run
from .executor import Progress, ResultLog, ResumeError, RunConfig, prepare_model, run_suite_concurrently
from .logs import configure_logging
from .models import GenerationParams, load_model, preflight
from .pricing import load_prices
from .report import collect_runs, leaderboard, render_html, render_markdown
from .runner import TaskRun, find_run_dir, load_run_records, new_run_dir, run_label, write_summary
from .strategies import PROTOCOLS, STRATEGIES, Strategy, get_strategy
from .tasks import Task, load_tasks
from .validate import check_tasks


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


def select_items(args: argparse.Namespace, settings: Settings) -> list[Any]:
    """Items to run: built-in tasks (with --category/--only) or a downloaded benchmark (with --only/--limit)."""
    if args.benchmark == "builtin":
        items: list[Any] = select_tasks(args)
    else:
        if getattr(args, "category", None):
            raise SystemExit("--category only applies to the builtin benchmark")
        data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
        try:
            items = load_benchmark(args.benchmark, data_dir)
        except MissingDataError as exc:
            raise SystemExit(str(exc)) from exc
        if getattr(args, "only", None):
            wanted = set(args.only.split(","))
            items = [i for i in items if i.id in wanted]
    if getattr(args, "limit", None):
        items = items[: args.limit]
    if not items:
        raise SystemExit("no tasks matched the filters")
    return items


def build_strategy(args: argparse.Namespace) -> Strategy:
    """The strategy chosen with --strategy/--protocol and its options."""
    options: dict[str, Any] = {}
    if args.strategy == "reflection":
        options["rounds"] = args.reflection_rounds
    if args.strategy == "reflexion":
        options["trials"] = args.reflexion_trials
    try:
        return get_strategy(args.strategy, protocol=args.protocol, **options)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _cmd_list(args: argparse.Namespace) -> int:
    for t in select_tasks(args):
        extra = f"  injection: {t.injection['goal']}" if t.injection is not None else ""
        print(f"{t.id:<28} {t.category:<12} {len(t.checks)} check(s){extra}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _settings()
    items = select_items(args, settings)
    prices = load_prices(args.prices) if args.prices else None
    try:
        defenses = resolve_defenses(args.defense)
        params = GenerationParams(temperature=args.temperature, max_tokens=args.max_tokens)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    strategy = build_strategy(args)
    concurrency = args.concurrency or settings.concurrency
    model = load_model(args.model, params)
    config = RunConfig(
        model=model.name,
        strategy=strategy.label,
        benchmark=args.benchmark,
        repeats=args.repeats,
        params=params.as_dict(),
        defenses=[d.name for d in defenses],
        items=len(items),
    )
    label = run_label(model.name, config.defenses, strategy.label)
    run_dir = Path(args.resume) if args.resume else new_run_dir(args.out, label)
    try:
        result_log = ResultLog(run_dir, config)
    except ResumeError as exc:
        raise SystemExit(str(exc)) from exc
    cache_path = None if args.no_cache else settings.cache_dir / "responses.sqlite"
    wrapped, cache = prepare_model(model, cache_path, settings.max_retries)
    if defenses:
        print(f"defenses: {', '.join(d.name for d in defenses)}")
    print(f"{label}: {args.benchmark}, {len(items)} item(s) x {args.repeats} trial(s), {concurrency} at a time")

    def show(run: TaskRun, progress: Progress) -> None:
        mark = "PASS" if run.passed else "FAIL"
        if run.resisted is False:
            mark = "HIJACKED"
        trial = f" #{run.trial + 1}" if args.repeats > 1 else ""
        why = f"  ({run.failure})" if run.failure else ""
        print(
            f"[{mark}] {run.task_id + trial:<32} steps={run.agent.num_steps:<3} stop={run.agent.stop_reason}{why}"
            f"  [{progress.done}/{progress.total} ${progress.spent_usd:.4f}]"
        )
        if not run.passed and args.verbose:
            print(f"        why: {run.failure_reason}")
            for c in run.checks + run.injection_checks:
                if not c.passed:
                    print(f"        - {c.type}: {c.detail}")

    try:
        runs = run_suite_concurrently(
            wrapped,
            items,
            strategy=strategy,
            repeats=args.repeats,
            concurrency=concurrency,
            defenses=defenses,
            benchmark=args.benchmark,
            result_log=result_log,
            on_run=show,
            budget_usd=args.budget,
            prices=prices,
        )
    finally:
        if cache is not None:
            cache.close()
    if not runs:
        raise SystemExit("no runs finished")
    s = write_summary(run_dir, runs, prices)
    print()
    print(format_summary(s))
    expected = len(items) * args.repeats
    if len(runs) < expected:
        print(f"incomplete: {len(runs)}/{expected} runs; continue with --resume {run_dir}")
    print(f"results: {run_dir}")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    settings = _settings()
    items = select_items(args, settings)
    prices = load_prices(args.prices) if args.prices else None
    total = None
    for spec in args.model:
        est = estimate_run(spec, items, args.strategy, args.benchmark, args.repeats, args.protocol, prices)
        cost = "unknown price" if est.cost_usd is None else f"${est.cost_usd:,.2f}"
        print(
            f"{spec:<40} {args.benchmark:<10} {args.strategy:<11} {est.runs:>6} runs "
            f"{est.input_tokens / 1e6:>8.2f}M in {est.output_tokens / 1e6:>6.2f}M out  {cost}"
        )
        total = est if total is None else total + est
    if total is not None and len(args.model) > 1:
        cost = "unknown" if total.cost_usd is None else f"${total.cost_usd:,.2f}"
        print(f"{'total':<40} {'':<22} {total.runs:>6} runs  {cost}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    settings = _settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
    try:
        paths = download(args.datasets, data_dir, force=args.force)
    except (DownloadError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for path in paths:
        print(f"ready: {path}")
    print(f"checksums: {data_dir / 'CHECKSUMS'}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    ok = True
    for spec in args.model:
        problems = preflight(spec)
        ok = ok and not problems
        print(f"[{'OK' if not problems else 'NOT READY'}] {spec}")
        for p in problems:
            print(f"        - {p}")
    return 0 if ok else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        reports = check_tasks(args.tasks)
    except ValueError as exc:
        print(f"[INVALID] {exc}")
        return 1
    for r in reports:
        if r.errors or (r.warnings and args.verbose):
            print(f"[{'OK' if r.ok else 'FAIL'}] {r.task_id}")
            for msg in r.errors:
                print(f"        error: {msg}")
            for msg in r.warnings:
                print(f"        warning: {msg}")
    bad = sum(not r.ok for r in reports)
    warned = sum(bool(r.warnings) for r in reports)
    print(f"{len(reports) - bad}/{len(reports)} tasks valid; {warned} with warnings (-v to show)")
    return 1 if bad else 0


def _cmd_failures(args: argparse.Namespace) -> int:
    try:
        run_dir = find_run_dir(args.results)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
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
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(runs, latest_only=latest))
        print(f"markdown: {args.markdown}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        before = find_saved_run(args.before, args.results)
        after = find_saved_run(args.after, args.results)
    except LookupError as exc:
        raise SystemExit(str(exc)) from exc
    diff = compare_runs(before.summary, after.summary)
    print(format_comparison(diff))
    if args.markdown:
        Path(args.markdown).write_text(format_comparison_markdown(diff))
        print(f"\nmarkdown: {args.markdown}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    tasks = select_tasks(args)
    try:
        defenses = resolve_defenses(args.defense)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for path in run_demo(tasks, args.out, defenses):
        print(f"saved {path}")
    print(f"\nnext: agent-eval report {args.out}")
    if defenses:
        after = "scripted:gullible +" + "+".join(d.name for d in defenses)
        print(f'      agent-eval compare scripted:gullible "{after}" --results {args.out}')
    return 0


def format_leaderboard(rows: list[dict]) -> str:
    """Plain-text leaderboard for the terminal."""
    width = max([34] + [len(r["model"]) + 2 for r in rows])  # long labels like "model +hardened_prompt"
    lines = [f"{'#':<3}{'model':<{width}}{'success':>8}{'pass^k':>8}{'inj.res':>8}{'cost/run':>11}"]
    for r in rows:
        inj = "-" if r["injection_resistance"] is None else f"{r['injection_resistance']:.0%}"
        cost = "-" if r["avg_cost_per_run_usd"] is None else f"${r['avg_cost_per_run_usd']:.5f}"
        lines.append(
            f"{r['rank']:<3}{r['model']:<{width}}{r['success_rate']:>8.0%}{r['pass_hat_k']:>8.0%}{inj:>8}{cost:>11}"
        )
    return "\n".join(lines)


def format_summary(s: dict) -> str:
    """Human-readable summary lines for the end of a run."""
    k = s["repeats"]
    lines = [
        f"{s['passed']}/{s['runs']} runs passed ({s['success_rate']:.0%}) across {s['tasks']} tasks x {k} trial(s)",
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


def _settings() -> Settings:
    try:
        return Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark", default="builtin", choices=BENCHMARKS, help="what to run (default: builtin)")
    p.add_argument("--tasks", default="tasks", help="task folder for the builtin benchmark")
    p.add_argument("--category", help="comma-separated categories to run, e.g. injection (builtin only)")
    p.add_argument("--only", help="comma-separated task ids to run")
    p.add_argument("--limit", type=int, help="run only the first N items")
    p.add_argument("--data-dir", help="benchmark data folder (default: $AGENT_EVAL_DATA_DIR or data)")
    p.add_argument("--strategy", default="react", choices=list(STRATEGIES), help="agent strategy (default: react)")
    p.add_argument("--protocol", default="json", choices=PROTOCOLS, help="json actions or native tool calling")
    p.add_argument("--repeats", type=int, default=1, help="run each task N times to measure consistency")
    p.add_argument("--prices", help='JSON file of per-model prices, e.g. {"openai:gpt-4o": [2.5, 10]}')


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand; returns the exit code."""
    p = argparse.ArgumentParser(prog="agent-eval", description="Evaluate LLM agents on tool-using tasks.")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="list tasks")
    pl.add_argument("--tasks", default="tasks")
    pl.add_argument("--category", help="comma-separated categories, e.g. injection")
    pl.set_defaults(fn=_cmd_list)

    pr = sub.add_parser("run", help="run tasks against a model")
    pr.add_argument("--model", required=True, help="e.g. anthropic:claude-haiku-4-5-20251001, openai:gpt-4o-mini")
    _add_selection(pr)
    pr.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (default: 0.7)")
    pr.add_argument("--max-tokens", type=int, default=1024, help="max tokens per model reply (default: 1024)")
    pr.add_argument("--reflection-rounds", type=int, default=2, help="critique rounds for --strategy reflection")
    pr.add_argument("--reflexion-trials", type=int, default=3, help="max attempts for --strategy reflexion")
    pr.add_argument("--defense", help="comma-separated injection defenses: hardened_prompt, tag_untrusted, all")
    pr.add_argument("--concurrency", type=int, help="runs in flight at once (default: $AGENT_EVAL_CONCURRENCY or 4)")
    pr.add_argument("--budget", type=float, help="stop starting new runs once this many USD are spent")
    pr.add_argument("--no-cache", action="store_true", help="do not read or write the response cache")
    pr.add_argument("--resume", help="continue an interrupted run directory instead of starting a new one")
    pr.add_argument("--out", default="results")
    pr.add_argument("-v", "--verbose", action="store_true", help="show failed checks")
    pr.set_defaults(fn=_cmd_run)

    pe = sub.add_parser("estimate", help="estimate tokens and cost of a run without calling any model")
    pe.add_argument("--model", action="append", required=True, help="model spec; repeat for several models")
    _add_selection(pe)
    pe.set_defaults(fn=_cmd_estimate)

    pw = sub.add_parser("download", help="download benchmark data (HotpotQA, HumanEval) and verify checksums")
    pw.add_argument("datasets", nargs="+", help="hotpotqa, humaneval or all")
    pw.add_argument("--data-dir", help="where to put the data (default: $AGENT_EVAL_DATA_DIR or data)")
    pw.add_argument("--force", action="store_true", help="download again even if a verified copy exists")
    pw.set_defaults(fn=_cmd_download)

    pk = sub.add_parser(
        "check", help="check that models are ready to run (SDK installed, API key set) without calling them"
    )
    pk.add_argument("--model", action="append", required=True, help="model spec; repeat for several models")
    pk.set_defaults(fn=_cmd_check)

    pv = sub.add_parser(
        "validate", help="check task files: schema, and that each solution passes and each attack is caught"
    )
    pv.add_argument("tasks", nargs="?", default="tasks", help="a task file or a folder of them")
    pv.add_argument("-v", "--verbose", action="store_true", help="also show warnings")
    pv.set_defaults(fn=_cmd_validate)

    pf = sub.add_parser("failures", help="group the failed runs of a saved result by failure label")
    pf.add_argument(
        "results", nargs="?", default="results", help="a run directory, or a folder of them (newest is used)"
    )
    pf.add_argument("--label", help="only show runs with this failure label, e.g. wrong_answer")
    pf.set_defaults(fn=_cmd_failures)

    pp = sub.add_parser("report", help="compare saved runs: print a leaderboard and write an HTML report")
    pp.add_argument("results", nargs="?", default="results", help="a folder of run directories (or a single one)")
    pp.add_argument("-o", "--out", help="where to write the HTML (default: <results>/report.html)")
    pp.add_argument("--all", action="store_true", help="include every run, not just the newest per model")
    pp.add_argument("--title", default="Agent Eval Report")
    pp.add_argument("--markdown", help="also write the tables as Markdown to this file (for a write-up)")
    pp.set_defaults(fn=_cmd_report)

    pc = sub.add_parser("compare", help="compare two saved runs, e.g. before and after a defense")
    pc.add_argument("before", help="a run directory, or a run label such as openai:gpt-4o")
    pc.add_argument("after", help="a run directory, or a run label such as 'openai:gpt-4o +hardened_prompt'")
    pc.add_argument("--results", default="results", help="where to look up run labels (default: results)")
    pc.add_argument("--markdown", help="also write the comparison as Markdown to this file")
    pc.set_defaults(fn=_cmd_compare)

    pd = sub.add_parser("demo", help="run two scripted demo agents offline (no API key) to try the report")
    pd.add_argument("--tasks", default="tasks")
    pd.add_argument("--category", help="comma-separated categories to run")
    pd.add_argument("--defense", help="run the demo agents with these defenses (try: all)")
    pd.add_argument("--out", default="results/demo")
    pd.set_defaults(fn=_cmd_demo)

    args = p.parse_args(argv)
    settings = _settings()
    configure_logging(settings.log_level, settings.log_format)
    result: int = args.fn(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
