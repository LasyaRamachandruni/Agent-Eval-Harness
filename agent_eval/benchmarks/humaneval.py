"""HumanEval (Chen et al., 2021): 164 Python functions, graded by hidden unit tests.

The model sees the function signature and docstring and must return the full
function. Grading runs the completion against the problem's hidden `check`
tests in the sandbox (see sandbox.py): no network, a timeout and resource
limits. A run passes when every test passes, so success rate is pass@1, and
`--repeats k` gives pass@k and pass^k.

Agent strategies get one tool, `run_python`, to try their code in the same
sandbox. They never see the hidden tests. Reflexion's visible self-check runs
only the examples written in the docstring, as in the Reflexion paper's
"internal tests".
"""

from __future__ import annotations

import ast
import doctest
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..agent import AgentResult
from ..environment import Environment
from ..grading import CheckResult, Grade
from ..sandbox import run_python
from ..tools import Tool

CATEGORY = "humaneval"
TEST_TIMEOUT_S = 10.0

PROMPT_TEMPLATE = """Complete this Python function. Your final answer must be the complete function \
(signature and body, plus any imports it needs) in one ```python code block, and nothing else.

```python
{prompt}```"""

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class HumanEvalProblem:
    """One problem as published in HumanEval.jsonl."""

    task_id: str
    prompt: str
    entry_point: str
    test: str
    canonical_solution: str = ""


def load_problems(path: str | Path) -> list[HumanEvalProblem]:
    """Read HumanEval.jsonl or HumanEval.jsonl.gz."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fields = HumanEvalProblem.__dataclass_fields__
    return [HumanEvalProblem(**{k: v for k, v in r.items() if k in fields}) for r in rows]


def extract_code(answer: str, entry_point: str) -> str:
    """The code in a model's answer: the fenced block defining `entry_point`, else the last block, else the text."""
    blocks = _FENCE.findall(answer)
    for block in blocks:
        if f"def {entry_point}" in block:
            return block
    if blocks:
        return blocks[-1]
    return answer.strip("\n")


def build_program(problem: HumanEvalProblem, answer: str) -> str:
    """Prompt + completion, as a runnable module.

    If the answer redefines the function, it is appended after the prompt (the
    prompt's stub, which is just a docstring, is replaced; helper functions and
    imports from the prompt stay available). Otherwise the answer is treated as
    the function body and appended to the prompt directly.
    """
    code = extract_code(answer, problem.entry_point)
    if f"def {problem.entry_point}" in code:
        return f"{problem.prompt}\n    pass\n\n{code}\n"
    return problem.prompt + code + "\n"


def grading_program(problem: HumanEvalProblem, answer: str) -> str:
    """The program that grades `answer`: the completion plus the hidden tests."""
    return f"{build_program(problem, answer)}\n\n{problem.test}\n\ncheck({problem.entry_point})\n"


def docstring_examples(prompt: str, entry_point: str) -> list[doctest.Example]:
    """The `>>>` examples in the docstring of `entry_point` (the only tests an agent may see)."""
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            doc = ast.get_docstring(node) or ""
            try:
                return doctest.DocTestParser().get_examples(doc)
            except ValueError:
                return []
    return []


def examples_program(code: str, examples: list[doctest.Example]) -> str:
    """A script that runs each docstring example against `code` and prints how many passed."""
    cases = [(ex.source.strip(), ex.want.strip()) for ex in examples]
    return (
        f"{code}\n\n"
        "def _same(src, want):\n"
        "    got = eval(src)\n"
        "    if want == '':\n"
        "        return got is None or got is True\n"
        "    if repr(got) == want or str(got) == want:\n"
        "        return True\n"
        "    try:\n"
        "        return got == eval(want)\n"
        "    except Exception:\n"
        "        return False\n\n"
        "_passed = 0\n"
        f"for _src, _want in {cases!r}:\n"
        "    try:\n"
        "        _passed += bool(_same(_src, _want))\n"
        "    except Exception:\n"
        "        pass\n"
        "print(_passed)\n"
    )


class HumanEvalEnv(Environment):
    """A `run_python` tool backed by the sandbox, and a docstring-example self-check."""

    def __init__(self, problem: HumanEvalProblem, timeout: float = TEST_TIMEOUT_S):
        self.problem = problem
        self.timeout = timeout
        tool = Tool(
            "run_python",
            "Run a Python program in a sandbox (no network, 10 s limit) and see its exit code and output. "
            "Use it to test your function before answering.",
            {"code": "a complete Python program"},
            self._run,
        )
        super().__init__({tool.name: tool})

    def _run(self, code: str) -> str:
        return run_python(code, timeout=self.timeout).summary()

    def feedback(self, answer: str) -> tuple[float, str] | None:
        examples = docstring_examples(self.problem.prompt, self.problem.entry_point)
        if not examples:
            return None
        result = run_python(examples_program(build_program(self.problem, answer), examples), timeout=self.timeout)
        lines = result.stdout.strip().splitlines()
        passed = int(lines[-1]) if result.ok and lines and lines[-1].isdigit() else 0
        detail = f"{passed}/{len(examples)} docstring examples pass"
        if not result.ok:
            detail += f" (the code failed: {result.summary(400)})"
        return passed / len(examples), detail


@dataclass
class HumanEvalItem:
    """One HumanEval problem as something the runner can evaluate."""

    problem: HumanEvalProblem
    max_steps: int = 6
    category: str = CATEGORY
    timeout: float = TEST_TIMEOUT_S

    @property
    def id(self) -> str:
        return self.problem.task_id

    @property
    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(prompt=self.problem.prompt)

    def make_env(self) -> HumanEvalEnv:
        return HumanEvalEnv(self.problem, self.timeout)

    def grade(self, result: AgentResult, env: Environment) -> Grade:
        """Run the hidden tests against the final answer in the sandbox."""
        if result.final_answer is None:
            check = CheckResult("hidden_tests", False, "no final answer")
            return Grade(False, [check], scores={"pass": 0.0})
        outcome = run_python(grading_program(self.problem, result.final_answer), timeout=self.timeout)
        passed = outcome.ok and result.stop_reason == "final"
        detail = "all hidden tests pass" if outcome.ok else outcome.summary(600)
        return Grade(passed, [CheckResult("hidden_tests", passed, detail)], scores={"pass": float(passed)})


def load_items(path: str | Path, limit: int | None = None) -> list[HumanEvalItem]:
    """HumanEval problems as evaluation items, in file order (optionally the first `limit`)."""
    problems = load_problems(path)
    return [HumanEvalItem(p) for p in problems[:limit]]
