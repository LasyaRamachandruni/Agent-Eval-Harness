"""Direct: one model call, no tools.

The baseline every agent strategy should beat. The model gets the task plus
whatever the environment can show inline (the workspace files, the HotpotQA
paragraphs) and must answer in one reply. It cannot change files, so tasks
that require a write fail by design. Defenses still apply: the inline content
is treated as untrusted data, like a `read_file` result.
"""

from __future__ import annotations

from ..agent import AgentResult, Step, parse_action
from ..defenses import Defense, apply_to_observation, apply_to_system
from ..environment import Environment
from ..models import ModelClient
from .base import StopReason, Strategy

DIRECT_SYSTEM = """Answer the user's task directly in one reply. You cannot call tools or change files.
Reply with only the final answer, short and exact."""


def final_from_text(text: str) -> str:
    """Accept either plain text or a JSON {"final": ...} object as the answer."""
    try:
        action = parse_action(text)
    except ValueError:
        return text.strip()
    if isinstance(action, dict) and "final" in action:
        return str(action["final"])
    return text.strip()


class DirectStrategy(Strategy):
    """Answer in a single model call without tools."""

    name = "direct"

    @property
    def label(self) -> str:
        """Direct answers never call tools, so the protocol does not apply."""
        return self.name

    def _run(
        self,
        model: ModelClient,
        task_prompt: str,
        env: Environment,
        result: AgentResult,
        max_steps: int,
        defenses: list[Defense],
    ) -> StopReason:
        context = env.describe()
        prompt = task_prompt
        if context:
            prompt += "\n\n" + apply_to_observation("read_file", context, defenses)
        try:
            reply = model.complete(apply_to_system(DIRECT_SYSTEM, defenses), [{"role": "user", "content": prompt}])
        except Exception as exc:
            result.steps.append(Step(0, "", "invalid", error=f"model error: {exc}"))
            return "model_error"
        result.record(reply)
        result.final_answer = final_from_text(reply.text)
        result.steps.append(Step(0, reply.text, "final"))
        return "final"
