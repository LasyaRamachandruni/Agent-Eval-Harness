"""HotpotQA (Yang et al., 2018), distractor setting: multi-hop questions over 10 paragraphs.

Each question comes with 10 Wikipedia paragraphs: the 2 that hold the answer
and 8 distractors. Instead of calling a live search API, the agent searches
and reads only these paragraphs, through two offline tools, so every run sees
exactly the same evidence and results are reproducible:

    search(query)          the 3 best-matching paragraph titles, with their first sentence
    read_paragraph(title)  the full paragraph

The direct strategy gets all 10 paragraphs inline instead. Answers are scored
with the official HotpotQA exact match and F1 (lower-case, strip punctuation
and articles, compare tokens). A run passes on exact match; F1 is averaged in
the summary as `scores.f1`.

The evaluation set is a fixed, seeded sample of 200 dev questions, chosen by
`select_subset` from the ids in sorted order, so it is the same on every machine.
"""

from __future__ import annotations

import collections
import json
import random
import re
import string
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent import AgentResult
from ..environment import Environment
from ..grading import CheckResult, Grade
from ..tools import Tool, ToolError

CATEGORY = "hotpotqa"
SUBSET_SIZE = 200
SUBSET_SEED = 1234

PROMPT_TEMPLATE = """Answer the question using the Wikipedia paragraphs you can search and read with your tools.
Your final answer must be only the answer itself: a short span such as a name, a date, a number, or yes/no. \
No explanation.

Question: {question}"""

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "who",
        "what",
        "which",
        "when",
        "where",
        "how",
        "did",
        "does",
        "do",
        "that",
        "this",
        "as",
        "its",
        "it",
        "his",
        "her",
        "their",
    }
)


# --- official metric -------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation, articles and extra whitespace (official HotpotQA normalisation)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if the normalised strings are equal, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 as in the official HotpotQA evaluation script."""
    pred_norm, gold_norm = normalize_answer(prediction), normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (pred_norm in special or gold_norm in special) and pred_norm != gold_norm:
        return 0.0
    pred_tokens, gold_tokens = pred_norm.split(), gold_norm.split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred_tokens), same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --- data --------------------------------------------------------------------


@dataclass(frozen=True)
class HotpotQuestion:
    """One question with its 10 paragraphs (title, sentences)."""

    id: str
    question: str
    answer: str
    context: list[tuple[str, list[str]]]
    type: str = ""
    level: str = ""
    supporting_facts: list[tuple[str, int]] = field(default_factory=list)

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> HotpotQuestion:
        """Build from one record of the official JSON (or our subset JSONL)."""
        return cls(
            id=r["_id"],
            question=r["question"],
            answer=r["answer"],
            context=[(str(title), [str(s) for s in sents]) for title, sents in r["context"]],
            type=r.get("type", ""),
            level=r.get("level", ""),
            supporting_facts=[(str(t), int(i)) for t, i in r.get("supporting_facts", [])],
        )


def select_subset(
    records: Iterable[dict[str, Any]], n: int = SUBSET_SIZE, seed: int = SUBSET_SEED
) -> list[dict[str, Any]]:
    """A fixed sample of `n` records: ids are sorted, then sampled with random.Random(seed)."""
    by_id = {r["_id"]: r for r in records}
    if n > len(by_id):
        raise ValueError(f"asked for {n} questions but only {len(by_id)} are available")
    chosen = random.Random(seed).sample(sorted(by_id), n)
    return [by_id[i] for i in chosen]


def load_questions(path: str | Path, limit: int | None = None) -> list[HotpotQuestion]:
    """Read the official JSON array, or a JSONL subset written by the download step."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    return [HotpotQuestion.from_record(r) for r in records[:limit]]


# --- environment -------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


class HotpotEnv(Environment):
    """Offline search over one question's 10 paragraphs. Read-only, so it has no state."""

    def __init__(self, question: HotpotQuestion):
        self.paragraphs = {title: " ".join(sents).strip() for title, sents in question.context}
        tools = [
            Tool(
                "search",
                "Search the Wikipedia paragraphs for this question. Returns the 3 best-matching titles "
                "with their first sentence.",
                {"query": "search words"},
                self.search,
            ),
            Tool(
                "read_paragraph",
                "Read the full paragraph with this exact title.",
                {"title": "paragraph title, as returned by search"},
                self.read_paragraph,
            ),
        ]
        super().__init__({t.name: t for t in tools})
        self._first = {title: (sents[0].strip() if sents else "") for title, sents in question.context}

    def search(self, query: str) -> str:
        """Rank titles by word overlap with the query (title words count double)."""
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        words = _tokens(query)
        scored = []
        for i, (title, text) in enumerate(self.paragraphs.items()):
            score = 2 * len(words & _tokens(title)) + len(words & _tokens(text))
            scored.append((-score, i, title))
        top = [title for neg, _, title in sorted(scored)[:3] if neg < 0]
        if not top:
            return "No paragraph matches those words. Try other words from the question."
        return "\n".join(f"{title}: {self._first[title]}" for title in top)

    def read_paragraph(self, title: str) -> str:
        """The paragraph whose title matches (case-insensitive)."""
        if not isinstance(title, str):
            raise ToolError("title must be a string")
        for name, text in self.paragraphs.items():
            if name.lower() == title.strip().lower():
                return text
        raise ToolError(f"no paragraph titled {title!r}; available titles: {', '.join(self.paragraphs)}")

    def describe(self) -> str:
        return "Wikipedia paragraphs:\n\n" + "\n\n".join(f"## {t}\n{p}" for t, p in self.paragraphs.items())


@dataclass
class HotpotItem:
    """One HotpotQA question as something the runner can evaluate."""

    question: HotpotQuestion
    max_steps: int = 8
    category: str = CATEGORY

    @property
    def id(self) -> str:
        return self.question.id

    @property
    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(question=self.question.question)

    def make_env(self) -> HotpotEnv:
        return HotpotEnv(self.question)

    def grade(self, result: AgentResult, env: Environment) -> Grade:
        """Official exact match decides pass/fail; F1 is recorded as a score."""
        prediction = result.final_answer or ""
        em = exact_match(prediction, self.question.answer)
        f1 = f1_score(prediction, self.question.answer)
        passed = bool(em) and result.stop_reason == "final"
        detail = f"expected {self.question.answer!r}, got {prediction!r} (F1 {f1:.2f})"
        return Grade(passed, [CheckResult("exact_match", passed, detail)], scores={"em": em, "f1": round(f1, 4)})


def load_items(path: str | Path, limit: int | None = None) -> list[HotpotItem]:
    """HotpotQA questions as evaluation items."""
    return [HotpotItem(q) for q in load_questions(path, limit)]
