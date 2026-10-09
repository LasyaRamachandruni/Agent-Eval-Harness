"""Grading results shared by every kind of evaluation item."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """Whether one check passed, with a short human-readable detail."""

    type: str
    passed: bool
    detail: str = ""


@dataclass
class Grade:
    """The outcome of grading one run.

    `passed` decides success. `scores` holds benchmark metrics that are not
    pass/fail (for example HotpotQA exact match and F1); they are averaged in
    the summary. `resisted` and `injection_checks` are set for prompt-injection
    tasks only.
    """

    passed: bool
    checks: list[CheckResult]
    resisted: bool | None = None
    injection_checks: list[CheckResult] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
