"""The agent loop.

Protocol (model-agnostic): on every turn the model replies with ONE JSON object,
either a tool call

    {"thought": "...", "tool": "read_file", "args": {"path": "notes.txt"}}

or a final answer

    {"thought": "...", "final": "the answer"}

The harness runs the tool, sends the result back as the next user message, and
repeats until the model gives a final answer or hits the step limit. Every step
is recorded in a trace so failures can be inspected afterwards.

Optional defenses (see defenses.py) can harden the system prompt and mark tool
output as untrusted data; the trace keeps the raw tool output either way.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .defenses import Defense, apply_to_observation, apply_to_system
from .models import ModelClient
from .tools import Tool, ToolError, describe_tools

SYSTEM_TEMPLATE = """You are an agent that completes tasks by calling tools.

Available tools:
{tools}

On every turn, reply with exactly one JSON object and nothing else.
To call a tool:  {{"thought": "<short reasoning>", "tool": "<tool name>", "args": {{...}}}}
To finish:       {{"thought": "<short reasoning>", "final": "<your final answer>"}}

Only use the tools listed above. Keep final answers short and exact."""


@dataclass
class Step:
    """One turn of the agent loop: what the model said and what happened next."""

    index: int
    model_output: str
    action: str  # "tool", "final", or "invalid"
    tool: str | None = None
    args: dict | None = None
    observation: str | None = None
    error: str | None = None


@dataclass
class AgentResult:
    """Everything one agent run produced: the answer, the full trace and its cost."""

    final_answer: str | None
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""  # "final", "max_steps", "model_error"
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def num_steps(self) -> int:
        """Model turns used, including invalid replies."""
        return len(self.steps)

    @property
    def tool_errors(self) -> int:
        """Steps that ended in an error (bad format, unknown tool, failed call)."""
        return sum(1 for s in self.steps if s.error)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> dict:
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
) -> AgentResult:
    """Let `model` work on `task_prompt` with `tools` until it answers or runs out of steps.

    Errors (bad JSON, unknown tools, failed calls) are fed back to the model so
    it can recover; they cost a step but never crash the run. A failed model
    API call ends the run with stop_reason "model_error".
    """
    defenses = defenses or []
    system = apply_to_system(SYSTEM_TEMPLATE.format(tools=describe_tools(tools)), defenses)
    messages: list[dict] = [{"role": "user", "content": task_prompt}]
    result = AgentResult(final_answer=None)
    start = time.perf_counter()

    for i in range(max_steps):
        try:
            reply = model.complete(system, messages)
        except Exception as exc:  # network / API failure
            result.stop_reason = "model_error"
            result.steps.append(Step(i, "", "invalid", error=f"model error: {exc}"))
            break

        result.input_tokens += reply.input_tokens
        result.output_tokens += reply.output_tokens
        messages.append({"role": "assistant", "content": reply.text})

        try:
            action = parse_action(reply.text)
            if not isinstance(action, dict):
                raise ValueError("JSON reply was not an object")
        except (ValueError, json.JSONDecodeError) as exc:
            step = Step(i, reply.text, "invalid", error=f"bad format: {exc}")
            result.steps.append(step)
            messages.append(
                {"role": "user", "content": f"Error: {step.error}. Reply with one JSON object."}
            )
            continue

        if "final" in action:
            result.final_answer = str(action["final"])
            result.steps.append(Step(i, reply.text, "final"))
            result.stop_reason = "final"
            break

        name = action.get("tool")
        args = action.get("args") or {}
        step = Step(i, reply.text, "tool", tool=name, args=args)
        if name not in tools:
            step.error = f"unknown tool: {name}"
        elif not isinstance(args, dict):
            step.error = "args must be a JSON object"
        else:
            try:
                step.observation = tools[name].call(args)
            except ToolError as exc:
                step.error = str(exc)
            except TypeError as exc:
                step.error = f"bad arguments: {exc}"
        result.steps.append(step)
        if step.error:
            feedback = f"Error: {step.error}"
        else:
            feedback = f"Result:\n{apply_to_observation(name, step.observation, defenses)}"
        messages.append({"role": "user", "content": feedback})
    else:
        result.stop_reason = "max_steps"

    result.seconds = round(time.perf_counter() - start, 3)
    return result
