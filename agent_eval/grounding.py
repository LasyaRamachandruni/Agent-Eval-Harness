"""Grounding check and attempt score, ported from Neural-Reflexion-Agent.

Neural-Reflexion-Agent keeps a citation only if its URL was actually returned
by a search in that run, and scores each revision with a fixed checklist to
decide when to stop. The same two ideas, adapted to tool-using agents:

* Grounding: the checkable facts in an answer (numbers and proper-noun-like
  words) should appear in something the agent actually saw: the task prompt
  or a tool result in this attempt. Facts that appear nowhere are reported as
  ungrounded. Numbers match numerically, so "1,200" matches "1200.0".
* Attempt score (0-100), a hand-written checklist, not a learned reward:

    finished with a final answer          40
    evidence                         up to 30  (visible self-check if the
                                               environment has one, else the
                                               grounded share of the facts)
    clean run                        up to 10  (minus 2 per tool error)
    self-evaluation                  up to 20  (the model's own 0-10 rating x 2)

Reflexion keeps the best-scoring attempt and stops when a new attempt does
not beat it, exactly like the stop rule in Neural-Reflexion-Agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"\b[A-Z][a-zA-Z'\-]{2,}\b")
_STOP = frozenset(
    {
        "The", "This", "That", "These", "Those", "There", "Yes", "Not", "And", "But", "For", "With",
        "Answer", "Final", "None", "True", "False", "From", "Its", "His", "Her", "Their", "They", "She",
        "What", "Which", "When", "Where", "Who", "How", "Why", "Total", "Result",
    }
)  # fmt: skip


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def extract_facts(answer: str) -> list[str]:
    """Numbers and capitalised words in `answer` that can be checked against sources."""
    facts: list[str] = []
    for token in _NUMBER.findall(answer):
        if token not in facts:
            facts.append(token)
    for word in _WORD.findall(answer):
        if word not in _STOP and word not in facts:
            facts.append(word)
    return facts


@dataclass
class GroundingReport:
    """Which facts in an answer were found in the sources, and the grounded share."""

    facts: list[str] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        """Share of facts found in a source (1.0 if the answer has no checkable facts)."""
        if not self.facts:
            return 1.0
        return 1 - len(self.ungrounded) / len(self.facts)


def check_grounding(answer: str, sources: list[str]) -> GroundingReport:
    """Report which facts of `answer` appear in `sources` (prompt and tool outputs)."""
    corpus = "\n".join(sources)
    lowered = corpus.lower()
    numbers = {v for v in (_to_float(t) for t in _NUMBER.findall(corpus)) if v is not None}
    report = GroundingReport(facts=extract_facts(answer))
    for fact in report.facts:
        value = _to_float(fact)
        found = value in numbers if value is not None else fact.lower() in lowered
        if not found:
            report.ungrounded.append(fact)
    return report


def score_attempt(
    finished: bool,
    evidence: float,
    tool_errors: int,
    self_score: float | None,
) -> float:
    """The 0-100 attempt score described above. `evidence` and the self-score are clamped."""
    score = 40.0 if finished else 0.0
    score += 30.0 * min(1.0, max(0.0, evidence))
    score += max(0.0, 10.0 - 2.0 * tool_errors)
    if self_score is not None:
        score += 2.0 * min(10.0, max(0.0, self_score))
    return round(score, 2)
