"""Use the harness as a Python library: your own model, your own task, no API key.

Run it from the repository root:

    python examples/library_usage.py

It defines a task in code, plugs in a toy "model" (a plain Python function that
reads the conversation and decides the next action), runs it three times and
prints the same summary the CLI prints. Swap `RuleBasedModel` for
`load_model("anthropic:claude-sonnet-4-5")` to evaluate a real model.
"""

from __future__ import annotations

import json

from agent_eval.cli import format_summary
from agent_eval.models import ModelClient, ModelReply
from agent_eval.runner import run_suite, summarize
from agent_eval.tasks import Task

TASK = Task.from_dict({
    "id": "count-cities",
    "category": "files",
    "prompt": "How many cities are listed in cities.txt? Reply with just the number.",
    "files": {"cities.txt": "Paris\nLagos\nLima\nOsaka\n"},
    "checks": [{"type": "answer_equals", "value": "4"}],
    "max_steps": 4,
})


class RuleBasedModel(ModelClient):
    """A toy agent: read cities.txt, then count its lines. Any object with
    `name` and `complete(system, messages)` can be evaluated the same way."""

    name = "example:rule-based"

    def complete(self, system: str, messages: list[dict]) -> ModelReply:
        last = messages[-1]["content"]
        if last.startswith("Result:"):
            lines = [line for line in last.splitlines()[1:] if line.strip()]
            action = {"thought": "count the lines", "final": str(len(lines))}
        else:
            action = {"thought": "read the list", "tool": "read_file", "args": {"path": "cities.txt"}}
        return ModelReply(json.dumps(action), input_tokens=len(system) // 4, output_tokens=20)


def main() -> dict:
    runs = run_suite(RuleBasedModel(), [TASK], repeats=3)
    for run in runs:
        print(f"trial {run.trial + 1}: {'PASS' if run.passed else 'FAIL'} "
              f"answer={run.agent.final_answer!r} steps={run.agent.num_steps}")
    summary = summarize(runs)
    print()
    print(format_summary(summary))
    return summary


if __name__ == "__main__":
    main()
