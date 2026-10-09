"""The agent's trace types and the classic entry point `run_agent`.

How the agent actually works is decided by a strategy (see strategies/):
`direct`, `react`, `reflection` or `reflexion`. Each one can talk to the model
in one of two protocols:

JSON-action protocol (works with any chat model): on every turn the model
replies with ONE JSON object, either a tool call

    {"thought": "...", "tool": "read_file", "args": {"path": "notes.txt"}}

or a final answer

    {"thought": "...", "final": "the answer"}

Native protocol: the provider's own tool-calling API; the model calls tools
natively and finishes by replying with plain text.

Every step is recorded in a trace so failures can be inspected afterwards.
Optional defenses (see defenses.py) can harden the system prompt and mark tool
output as untrusted data; the trace keeps the raw tool output either way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .defenses import Defense
from .models import ModelClient, ModelReply
from .tools import Tool

if TYPE_CHECKING:  # pragma: no cover
    from .strategies import Protocol

SYSTEM_TEMPLATE = """You are an agent that completes tasks by calling tools.

Available tools:
{tools}

On every turn, reply with exactly one JSON object and nothing else.
To call a tool:  {{"thought": "<short reasoning>", "tool": "<tool name>", "args": {{...}}}}
To finish:       {{"thought": "<short reasoning>", "final": "<your final answer>"}}

Only use the tools listed above. Keep final answers short and exact."""


@dataclass
class Step:
    """One turn of the agent: what the model said and what happened next.

    `action` is "tool", "final" or "invalid" for agent turns, and "reflect" for
    the extra model calls that reflection strategies make to critique or score
    an answer. `attempt` numbers the Reflexion trial the step belongs to.
    """

    index: int
    model_output: str
    action: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    observation: str | None = None
    error: str | None = None
    attempt: int = 0


@dataclass
class AgentResult:
    """Everything one agent run produced: the answer, the full trace and its cost."""

    final_answer: str | None
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""  # "final", "max_steps", "model_error"
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    strategy: str = "react"
    model_calls: int = 0
    cached_calls: int = 0
    # Tokens of calls that were not served from the response cache (what this run actually cost).
    billed_input_tokens: int = 0
    billed_output_tokens: int = 0
    attempts: int = 1
    reflections: list[str] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        """Agent turns used, including invalid replies (critique/scoring calls are not counted)."""
        return sum(1 for s in self.steps if s.action != "reflect")

    @property
    def tool_errors(self) -> int:
        """Steps that ended in an error (bad format, unknown tool, failed call)."""
        return sum(1 for s in self.steps if s.error)

    def record(self, reply: ModelReply) -> None:
        """Add one model reply's token usage to the totals."""
        self.model_calls += 1
        self.input_tokens += reply.input_tokens
        self.output_tokens += reply.output_tokens
        if reply.cached:
            self.cached_calls += 1
        else:
            self.billed_input_tokens += reply.input_tokens
            self.billed_output_tokens += reply.output_tokens

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentResult:
        """Rebuild a result from its saved JSON form (see runner.save_results)."""
        data = dict(d)
        steps = [Step(**s) for s in data.pop("steps", [])]
        known = set(cls.__dataclass_fields__)
        return cls(steps=steps, **{k: v for k, v in data.items() if k in known})


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> Any:
    """Pull the JSON action out of a model reply, tolerating code fences or extra prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("reply did not contain a JSON object")
    return json.loads(match.group(0))


def run_agent(
    model: ModelClient,
    task_prompt: str,
    tools: dict[str, Tool],
    max_steps: int = 10,
    defenses: list[Defense] | None = None,
    protocol: Protocol = "json",
) -> AgentResult:
    """Let `model` work on `task_prompt` with `tools` (ReAct loop) until it answers or runs out of steps.

    Errors (bad JSON, unknown tools, failed calls) are fed back to the model so
    it can recover; they cost a step but never crash the run. A failed model
    API call ends the run with stop_reason "model_error".
    """
    from .environment import Environment
    from .strategies import get_strategy

    strategy = get_strategy("react", protocol=protocol)
    return strategy.run(model, task_prompt, Environment(tools), max_steps=max_steps, defenses=defenses or [])
