"""Command line entry point.

    agent-eval run --model anthropic:claude-sonnet-4-5 --tasks tasks/
    agent-eval list --tasks tasks/
"""

from __future__ import annotations

import argparse
import sys

from .models import load_model
from .runner import run_task, save_results, summarize
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
    model = load_model(args.model)
    runs = []
    for t in tasks:
        run = run_task(model, t)
        mark = "PASS" if run.passed else "FAIL"
        print(f"[{mark}] {t.id:<28} steps={run.agent.num_steps:<3} stop={run.agent.stop_reason}")
        if not run.passed and args.verbose:
            for c in run.checks:
                if not c.passed:
                    print(f"        - {c.type}: {c.detail}")
        runs.append(run)
    s = summarize(runs)
    out = save_results(runs, args.out)
    print(f"\n{s['passed']}/{s['tasks']} passed ({s['success_rate']:.0%}), "
          f"avg {s['avg_steps']} steps, {s['tool_errors']} tool errors")
    print(f"results: {out}")
    return 0


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
    pr.add_argument("--out", default="results")
    pr.add_argument("-v", "--verbose", action="store_true", help="show failed checks")
    pr.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
