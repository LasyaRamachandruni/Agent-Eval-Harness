"""Shared pieces of every strategy: the tool-using agent loop in both protocols."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from ..agent import SYSTEM_TEMPLATE, AgentResult, Step, parse_action
from ..defenses import Defense, apply_to_observation, apply_to_system
from ..environment import Environment
from ..logs import get_logger
from ..models import Message, ModelClient, ModelReply
from ..tools import Tool, ToolError, describe_tools

log = get_logger(__name__)

Protocol = Literal["json", "native"]
PROTOCOLS: tuple[Protocol, ...] = ("json", "native")
StopReason = Literal["final", "max_steps", "model_error"]

NATIVE_TEMPLATE = """You are an agent that completes tasks by calling the tools you are given.

Call tools whenever you need information or need to change something. When you
are done, reply with only your final answer as plain text, without calling a
tool. Keep final answers short and exact."""


def execute_tool(tools: dict[str, Tool], name: object, args: object) -> tuple[str | None, str | None]:
    """Run one tool call; returns (observation, error). Never raises for bad calls."""
    if not isinstance(name, str) or name not in tools:
        return None, f"unknown tool: {name}"
    if not isinstance(args, dict):
        return None, "args must be a JSON object"
    try:
        return tools[name].call(args), None
    except ToolError as exc:
        return None, str(exc)
    except TypeError as exc:
        return None, f"bad arguments: {exc}"


def json_system_prompt(tools: dict[str, Tool], defenses: list[Defense], extra: str = "") -> str:
    """The JSON-action system prompt for these tools, with defenses and any extra text appended."""
    system = apply_to_system(SYSTEM_TEMPLATE.format(tools=describe_tools(tools)), defenses)
    return f"{system}\n\n{extra}" if extra else system


def native_system_prompt(defenses: list[Defense], extra: str = "") -> str:
    """The native-protocol system prompt, with defenses and any extra text appended."""
    system = apply_to_system(NATIVE_TEMPLATE, defenses)
    return f"{system}\n\n{extra}" if extra else system


class AgentLoop:
    """One conversation between a model and an environment's tools.

    `run(budget)` lets the model take up to `budget` turns and stops at the
    first final answer. The conversation can be continued afterwards (the
    reflection strategy adds a critique and calls `run` again).
    """

    def __init__(
        self,
        model: ModelClient,
        env: Environment,
        result: AgentResult,
        *,
        protocol: Protocol = "json",
        defenses: list[Defense] | None = None,
        system_extra: str = "",
        attempt: int = 0,
    ):
        if protocol == "native" and not model.supports_native_tools:
            raise ValueError(f"{model.name} does not support native tool calling; use --protocol json")
        self.model = model
        self.env = env
        self.result = result
        self.protocol = protocol
        self.defenses = defenses or []
        self.attempt = attempt
        if protocol == "json":
            self.system = json_system_prompt(env.tools, self.defenses, system_extra)
        else:
            self.system = native_system_prompt(self.defenses, system_extra)
        self.specs = [t.spec() for t in env.tools.values()]
        self.messages: list[Message] = []
        self.turn = 0

    def add_user(self, text: str) -> None:
        """Append a user message (the task, or feedback such as a critique)."""
        self.messages.append({"role": "user", "content": text})

    def observations(self) -> list[str]:
        """Tool outputs seen in this loop, for grounding checks."""
        return [s.observation for s in self.result.steps if s.attempt == self.attempt and s.observation]

    def _call(self) -> ModelReply | None:
        try:
            if self.protocol == "native":
                reply = self.model.complete_with_tools(self.system, self.messages, self.specs)
            else:
                reply = self.model.complete(self.system, self.messages)
        except Exception as exc:  # network / API failure after retries
            log.warning("model call failed", extra={"fields": {"model": self.model.name, "error": repr(exc)}})
            self.result.steps.append(Step(self.turn, "", "invalid", error=f"model error: {exc}", attempt=self.attempt))
            return None
        self.result.record(reply)
        return reply

    def run(self, budget: int) -> StopReason:
        """Let the model take up to `budget` turns; sets result.final_answer when it finishes."""
        for _ in range(budget):
            reply = self._call()
            if reply is None:
                return "model_error"
            done = self._json_turn(reply) if self.protocol == "json" else self._native_turn(reply)
            self.turn += 1
            if done:
                return "final"
        return "max_steps"

    def _observe(self, name: str, observation: str) -> str:
        return apply_to_observation(name, observation, self.defenses)

    def _json_turn(self, reply: ModelReply) -> bool:
        self.messages.append({"role": "assistant", "content": reply.text})
        try:
            action = parse_action(reply.text)
            if not isinstance(action, dict):
                raise ValueError("JSON reply was not an object")
        except (ValueError, json.JSONDecodeError) as exc:
            step = Step(self.turn, reply.text, "invalid", error=f"bad format: {exc}", attempt=self.attempt)
            self.result.steps.append(step)
            self.add_user(f"Error: {step.error}. Reply with one JSON object.")
            return False
        if "final" in action:
            self.result.final_answer = str(action["final"])
            self.result.steps.append(Step(self.turn, reply.text, "final", attempt=self.attempt))
            return True
        name, args = action.get("tool"), action.get("args") or {}
        observation, error = execute_tool(self.env.tools, name, args)
        step = Step(
            self.turn,
            reply.text,
            "tool",
            tool=name if isinstance(name, str) else str(name),
            args=args if isinstance(args, dict) else None,
            observation=observation,
            error=error,
            attempt=self.attempt,
        )
        self.result.steps.append(step)
        if error:
            self.add_user(f"Error: {error}")
        else:
            self.add_user(f"Result:\n{self._observe(str(name), observation or '')}")
        return False

    def _native_turn(self, reply: ModelReply) -> bool:
        if not reply.tool_calls:
            text = reply.text.strip()
            self.messages.append({"role": "assistant", "content": reply.text})
            if not text:
                self.result.steps.append(
                    Step(self.turn, "", "invalid", error="bad format: empty reply", attempt=self.attempt)
                )
                self.add_user("Error: empty reply. Call a tool or reply with your final answer.")
                return False
            self.result.final_answer = text
            self.result.steps.append(Step(self.turn, reply.text, "final", attempt=self.attempt))
            return True
        calls: list[dict[str, Any]] = [{"id": c.id, "name": c.name, "args": c.args} for c in reply.tool_calls]
        self.messages.append({"role": "assistant", "content": reply.text, "tool_calls": calls})
        output = json.dumps({"text": reply.text, "tool_calls": calls}) if reply.text else json.dumps(calls)
        for call in reply.tool_calls:
            observation: str | None
            error: str | None
            if call.error:
                observation, error = None, call.error
            else:
                observation, error = execute_tool(self.env.tools, call.name, call.args)
            self.result.steps.append(
                Step(
                    self.turn,
                    output,
                    "tool",
                    tool=call.name,
                    args=call.args,
                    observation=observation,
                    error=error,
                    attempt=self.attempt,
                )
            )
            content = f"Error: {error}" if error else self._observe(call.name, observation or "")
            self.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
        return False


class Strategy:
    """How an agent goes from a task to an answer. Subclasses implement `_run`."""

    name = "base"

    def __init__(self, protocol: Protocol = "json"):
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r} (choose from: {', '.join(PROTOCOLS)})")
        self.protocol: Protocol = protocol

    @property
    def label(self) -> str:
        """How this strategy is named in results, e.g. "react" or "reflexion/native"."""
        return self.name if self.protocol == "json" else f"{self.name}/{self.protocol}"

    def run(
        self,
        model: ModelClient,
        task_prompt: str,
        env: Environment,
        *,
        max_steps: int = 10,
        defenses: list[Defense] | None = None,
    ) -> AgentResult:
        """Work on `task_prompt` in `env` and return the answer with its full trace."""
        result = AgentResult(final_answer=None, strategy=self.label)
        start = time.perf_counter()
        result.stop_reason = self._run(model, task_prompt, env, result, max_steps, defenses or [])
        result.seconds = round(time.perf_counter() - start, 3)
        return result

    def _run(
        self,
        model: ModelClient,
        task_prompt: str,
        env: Environment,
        result: AgentResult,
        max_steps: int,
        defenses: list[Defense],
    ) -> StopReason:  # pragma: no cover
        raise NotImplementedError
