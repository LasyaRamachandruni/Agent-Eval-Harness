"""Helpers for strategies that ask the model to review an attempt."""

from __future__ import annotations

from typing import Any

from ..agent import AgentResult, Step, parse_action
from ..models import ModelClient

MAX_OBSERVATION_CHARS = 600
MAX_STEPS_SHOWN = 12


def _clip(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more characters]"


def render_trajectory(steps: list[Step], attempt: int, answer: str | None) -> str:
    """A compact, readable account of one attempt for a reviewer prompt."""
    lines = []
    shown = [s for s in steps if s.attempt == attempt and s.action != "reflect"][-MAX_STEPS_SHOWN:]
    for s in shown:
        if s.action == "tool":
            lines.append(f"- called {s.tool}({s.args})")
            lines.append(f"  -> {_clip(s.error or s.observation or '')}")
        elif s.action == "invalid":
            lines.append(f"- invalid reply: {s.error}")
    lines.append(f"Final answer: {answer!r}" if answer is not None else "Final answer: (none, ran out of steps)")
    return "\n".join(lines)


def ask_json(model: ModelClient, system: str, prompt: str, result: AgentResult, attempt: int) -> dict[str, Any] | None:
    """One reviewer call that should return a JSON object; recorded as a "reflect" step.

    Returns None if the call fails or the reply is not a JSON object, so a bad
    review never crashes the run.
    """
    index = len(result.steps)
    try:
        reply = model.complete(system, [{"role": "user", "content": prompt}])
    except Exception as exc:
        result.steps.append(Step(index, "", "reflect", error=f"model error: {exc}", attempt=attempt))
        return None
    result.record(reply)
    try:
        data = parse_action(reply.text)
    except ValueError:
        data = None
    ok = isinstance(data, dict)
    result.steps.append(
        Step(
            index,
            reply.text,
            "reflect",
            error=None if ok else "bad format: review was not a JSON object",
            attempt=attempt,
        )
    )
    return data if ok else None
