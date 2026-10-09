"""ReAct: reason, call a tool, observe, repeat until a final answer.

Port of Basic_-React_Agent_Langchain (LangChain's ZERO_SHOT_REACT agent),
written against the harness's own model interface instead of LangChain, so
it runs on every provider and in both protocols.
"""

from __future__ import annotations

from ..agent import AgentResult
from ..defenses import Defense
from ..environment import Environment
from ..models import ModelClient
from .base import AgentLoop, StopReason, Strategy


class ReactStrategy(Strategy):
    """The tool-using agent loop with error feedback."""

    name = "react"

    def _run(
        self,
        model: ModelClient,
        task_prompt: str,
        env: Environment,
        result: AgentResult,
        max_steps: int,
        defenses: list[Defense],
    ) -> StopReason:
        loop = AgentLoop(model, env, result, protocol=self.protocol, defenses=defenses)
        loop.add_user(task_prompt)
        return loop.run(max_steps)
