"""Check the benchmark adapters themselves, without any model.

    humaneval  every canonical solution, sent through the real runner and
               grader (sandbox included), must pass its hidden tests; and the
               docstring-example self-check used by Reflexion is run on the
               canonical solutions to show how often it agrees.
    hotpotqa   every gold answer must score EM 1.0 through the grader; and the
               offline `search` tool is measured: how often a search with the
               question text returns both supporting paragraphs in its top 3.

These numbers say the harness grades correctly; they say nothing about models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import ModelClient, ModelReply
from ..runner import run_task
from ..strategies import get_strategy
from .hotpotqa import HotpotItem
from .humaneval import HumanEvalItem


class OracleModel(ModelClient):
    """Always replies with one fixed reference answer."""

    name = "oracle"

    def __init__(self, answer: str):
        self.answer = answer

    def complete(self, system: str, messages: list[dict[str, Any]]) -> ModelReply:
        return ModelReply(self.answer)


@dataclass
class SelfCheck:
    """One measured property of an adapter."""

    name: str
    passed: int
    total: int
    failures: list[str]

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def line(self) -> str:
        """e.g. "canonical solutions pass hidden tests: 164/164 (100.0%)"."""
        text = f"{self.name}: {self.passed}/{self.total} ({self.rate:.1%})"
        return text + (f"; not passing: {', '.join(self.failures[:10])}" if self.failures else "")


def _oracle_run(item: Any, answer: str) -> bool:
    return run_task(OracleModel(answer), item, strategy=get_strategy("direct")).passed


def check_humaneval(items: list[HumanEvalItem]) -> list[SelfCheck]:
    """Canonical solutions through the grader, and the docstring self-check on them."""
    bad = [i.id for i in items if not _oracle_run(i, i.problem.prompt + i.problem.canonical_solution)]
    with_examples, agree, disagree = 0, 0, []
    for item in items:
        feedback = item.make_env().feedback(item.problem.prompt + item.problem.canonical_solution)
        if feedback is None:
            continue
        with_examples += 1
        if feedback[0] == 1.0:
            agree += 1
        else:
            disagree.append(item.id)
    return [
        SelfCheck("canonical solutions pass hidden tests", len(items) - len(bad), len(items), bad),
        SelfCheck("docstring examples agree with the canonical solution", agree, with_examples, disagree),
    ]


def check_hotpotqa(items: list[HotpotItem]) -> list[SelfCheck]:
    """Gold answers through the grader, and recall@3 of the offline search tool."""
    bad = [i.id for i in items if not _oracle_run(i, i.question.answer)]
    found, missed = 0, []
    for item in items:
        gold = {title for title, _ in item.question.supporting_facts}
        top = {line.split(":", 1)[0] for line in item.make_env().search(item.question.question).splitlines()}
        if gold and gold <= top:
            found += 1
        else:
            missed.append(item.id)
    return [
        SelfCheck("gold answers score exact match", len(items) - len(bad), len(items), bad),
        SelfCheck("search(question) returns both supporting paragraphs in top 3", found, len(items), missed),
    ]
