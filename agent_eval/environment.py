"""Environments: the tools and state one run works against.

A strategy never touches files or datasets directly; it gets an Environment
with a set of tools. Environments can be snapshotted and restored, which is
what lets the Reflexion strategy start each trial from the task's original
state and keep the end state of its best trial for grading.
"""

from __future__ import annotations

from typing import Protocol

from .agent import AgentResult
from .grading import Grade
from .tools import Tool, Workspace, build_tools


class Environment:
    """Base environment: a fixed tool set and no state."""

    def __init__(self, tools: dict[str, Tool] | None = None):
        self.tools: dict[str, Tool] = tools or {}

    def snapshot(self) -> object:
        """Capture the current state (default: stateless)."""
        return None

    def restore(self, state: object) -> None:
        """Return to a state captured by `snapshot`."""

    def describe(self) -> str:
        """Everything a tool-less agent should see inline (used by the direct strategy)."""
        return ""

    def feedback(self, answer: str) -> tuple[float, str] | None:
        """Optional visible self-check of an answer: (score 0..1, explanation), or None.

        This must only use information the agent is allowed to see (for
        HumanEval, the examples in the docstring, never the hidden tests).
        """
        return None


class WorkspaceEnv(Environment):
    """The in-memory file workspace used by the built-in task suite."""

    def __init__(self, files: dict[str, str]):
        self.workspace = Workspace(files=dict(files))
        super().__init__(build_tools(self.workspace))

    def snapshot(self) -> object:
        return dict(self.workspace.files)

    def restore(self, state: object) -> None:
        if not isinstance(state, dict):
            raise TypeError("workspace state must be a dict of files")
        self.workspace.files.clear()
        self.workspace.files.update(state)

    def describe(self) -> str:
        files = self.workspace.files
        if not files:
            return "The workspace has no files."
        parts = [f"--- {path} ---\n{files[path]}" for path in sorted(files)]
        return "Files in the workspace:\n\n" + "\n\n".join(parts)


class EvalItem(Protocol):
    """Anything the runner can evaluate: a built-in task or a benchmark question."""

    id: str
    category: str
    prompt: str
    max_steps: int

    def make_env(self) -> Environment: ...

    def grade(self, result: AgentResult, env: Environment) -> Grade: ...
