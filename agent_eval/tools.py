"""Tools the agent can call.

Every task runs inside its own Workspace: an in-memory set of files. Nothing
the agent does touches the real filesystem, which keeps runs repeatable and
makes it easy to check the end state after a run.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(Exception):
    """Raised when a tool is called with bad arguments or fails."""


def normalize_path(path: str) -> str:
    """Map the spellings a model might use for one file to a single key.

    "notes.txt", "./notes.txt" and "/notes.txt" are the same workspace file, so
    an agent is not marked wrong for a harmless "./" prefix. Paths that try to
    leave the workspace ("../x") or are not strings are rejected.
    """
    if not isinstance(path, str) or not path.strip():
        raise ToolError("path must be a non-empty string")
    parts = []
    for part in path.strip().replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ToolError(f"path must stay inside the workspace: {path}")
        parts.append(part)
    if not parts:
        raise ToolError(f"not a file path: {path}")
    return "/".join(parts)


@dataclass
class Workspace:
    """The in-memory files one task run can see and change (path -> contents)."""

    files: dict[str, str] = field(default_factory=dict)

    def list_files(self) -> str:
        if not self.files:
            return "(no files)"
        return "\n".join(sorted(self.files))

    def read_file(self, path: str) -> str:
        key = normalize_path(path)
        if key not in self.files:
            raise ToolError(f"file not found: {path}")
        return self.files[key]

    def write_file(self, path: str, content: str) -> str:
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        key = normalize_path(path)
        self.files[key] = content
        return f"wrote {len(content)} characters to {key}"


# --- calculator -------------------------------------------------------------
# A tiny safe evaluator: numbers and + - * / // % ** only. No eval().

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ToolError("calculator only supports numbers and + - * / // % **")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {expression}") from exc
    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --- registry ---------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # param name -> short description
    fn: Callable[..., str]

    def call(self, args: dict[str, Any]) -> str:
        missing = [p for p in self.params if p not in args]
        if missing:
            raise ToolError(f"missing argument(s) for {self.name}: {', '.join(missing)}")
        extra = [a for a in args if a not in self.params]
        if extra:
            raise ToolError(f"unknown argument(s) for {self.name}: {', '.join(extra)}")
        return self.fn(**args)


def build_tools(ws: Workspace) -> dict[str, Tool]:
    tools = [
        Tool("list_files", "List the files in the workspace.", {}, ws.list_files),
        Tool("read_file", "Read a file from the workspace.", {"path": "file path"}, ws.read_file),
        Tool(
            "write_file",
            "Create or overwrite a file in the workspace.",
            {"path": "file path", "content": "full file contents"},
            ws.write_file,
        ),
        Tool(
            "calculator",
            "Evaluate an arithmetic expression, e.g. '(12 + 3) * 4'.",
            {"expression": "arithmetic expression"},
            calculator,
        ),
    ]
    return {t.name: t for t in tools}


def describe_tools(tools: dict[str, Tool]) -> str:
    lines = []
    for t in tools.values():
        params = ", ".join(f"{k}: {v}" for k, v in t.params.items()) or "no arguments"
        lines.append(f"- {t.name}({params}): {t.description}")
    return "\n".join(lines)
