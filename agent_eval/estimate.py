"""Estimate tokens and cost before running anything.

The estimate is built from the actual prompts of the items to be run, with a
simple model of how a run unfolds:

* tokens are approximated as characters / 4;
* each agent turn re-sends the system prompt, the task and the conversation
  so far, and adds one reply plus one tool observation;
* the expected number of turns and the size of a reply and an observation
  depend on the benchmark (see TURN_MODEL);
* reflection and reflexion add reviewer calls and retries, modelled as
  multipliers on a ReAct run (see STRATEGY_FACTOR).

It is meant to answer "is this $5 or $500?" before spending anything, not to
predict a bill: expect real costs within roughly half to double the estimate.
After a real run, `summary.json` has the measured tokens and cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent import SYSTEM_TEMPLATE
from .pricing import estimate_cost
from .strategies.base import NATIVE_TEMPLATE
from .tools import describe_tools

CHARS_PER_TOKEN = 4

# benchmark -> (expected agent turns, tokens per reply, tokens per tool observation)
TURN_MODEL: dict[str, tuple[float, int, int]] = {
    "builtin": (3.5, 70, 90),
    "hotpotqa": (4.0, 60, 140),
    "humaneval": (2.5, 220, 60),
}

# strategy -> (input multiplier, output multiplier) relative to a ReAct run
STRATEGY_FACTOR: dict[str, tuple[float, float]] = {
    "react": (1.0, 1.0),
    "reflection": (1.7, 1.5),
    "reflexion": (2.3, 2.0),
}


def _tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Estimate:
    """Estimated totals for one (model, strategy, benchmark) run."""

    runs: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    def __add__(self, other: Estimate) -> Estimate:
        cost = None if self.cost_usd is None or other.cost_usd is None else self.cost_usd + other.cost_usd
        return Estimate(
            self.runs + other.runs,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            cost,
        )


def item_tokens(item: Any, strategy: str, benchmark: str, protocol: str = "json") -> tuple[float, float]:
    """Expected (input, output) tokens for one run of `item`."""
    turns, reply, observation = TURN_MODEL.get(benchmark, TURN_MODEL["builtin"])
    env = item.make_env()
    if strategy == "direct":
        prompt = _tokens(item.prompt) + _tokens(env.describe()) + 40
        return float(prompt), float(reply if benchmark != "humaneval" else 220)
    system = NATIVE_TEMPLATE if protocol == "native" else SYSTEM_TEMPLATE.format(tools=describe_tools(env.tools))
    base = _tokens(system) + _tokens(item.prompt)
    per_turn = reply + observation
    turns = min(turns, float(item.max_steps))
    whole_turns = int(turns)
    inputs = float(sum(base + i * per_turn for i in range(whole_turns)))
    inputs += (turns - whole_turns) * (base + whole_turns * per_turn)
    outputs = turns * reply
    f_in, f_out = STRATEGY_FACTOR.get(strategy, (1.0, 1.0))
    return inputs * f_in, outputs * f_out


def estimate_run(
    model: str,
    items: list[Any],
    strategy: str,
    benchmark: str,
    repeats: int,
    protocol: str = "json",
    prices: dict[str, tuple[float, float]] | None = None,
) -> Estimate:
    """Estimate one run of `items` x `repeats` with `strategy` on `model`."""
    total_in = total_out = 0.0
    for item in items:
        i, o = item_tokens(item, strategy, benchmark, protocol)
        total_in += i
        total_out += o
    total_in *= repeats
    total_out *= repeats
    cost = estimate_cost(model, int(total_in), int(total_out), prices)
    return Estimate(len(items) * repeats, int(total_in), int(total_out), cost)
