"""Command line entry point.

    agent-eval run --model anthropic:claude-sonnet-4-5 --tasks tasks/
    agent-eval list --tasks tasks/
"""

from __future__ import annotations

import argparse
import sys

from .models import load_model
from .pricing import load_prices
from .runner import TaskRun, run_suite, save_results, summarize
from .tasks import load_tasks


def _cmd_list(args: argparse.Namespace) -> int:
    for t in load_tasks(args.tasks):
        print(f"{t.id:<28} {t.category:<12} {len(t.checks)} check(s)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t.id in wanted]
    prices = load_prices(args.prices) if args.prices else None
    model = load_model(args.model)

    def show(run: TaskRun) -> None:
        mark = "PASS" if run.passed else "FAIL"
        if run.resisted is False:
            mark = "HIJACKED"
        trial = f" #{run.trial + 1}" if args.repeats > 1 else ""
        print(f"[{mark}] {run.task_id + trial:<32} steps={run.agent.num_steps:<3} stop={run.agent.stop_reason}")
        if not run.passed and args.verbose:
            for c in run.checks + run.injection_checks:
                if not c.passed:
                    print(f"        - {c.type}: {c.detail}")

    runs = run_suite(model, tasks, repeats=args.repeats, on_run=show)
    s = summarize(runs, prices)
    out = save_results(runs, args.out, prices)
    print()
    print(format_summary(s))
    print(f"results: {out}")
    return 0


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
    pl.set_defaults(fn=_cmd_list)

    pr = sub.add_parser("run", help="run tasks against a model")
    pr.add_argument("--model", required=True, help="e.g. anthropic:claude-sonnet-4-5, openai:gpt-4o-mini, ollama:llama3.1")
    pr.add_argument("--tasks", default="tasks")
    pr.add_argument("--only", help="comma-separated task ids to run")
    pr.add_argument("--repeats", type=int, default=1, help="run each task N times to measure consistency")
    pr.add_argument("--prices", help="JSON file of per-model prices, e.g. {\"openai:gpt-4o\": [2.5, 10]}")
    pr.add_argument("--out", default="results")
    pr.add_argument("-v", "--verbose", action="store_true", help="show failed checks")
    pr.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
