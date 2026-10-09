"""HotpotQA and HumanEval adapters and the download step, offline with tiny fixtures."""

import gzip
import hashlib
import io
import json

import pytest

from agent_eval.benchmarks import MissingDataError, load_benchmark
from agent_eval.benchmarks import download as dl
from agent_eval.benchmarks.hotpotqa import (
    HotpotEnv,
    HotpotItem,
    HotpotQuestion,
    exact_match,
    f1_score,
    normalize_answer,
    select_subset,
)
from agent_eval.benchmarks.humaneval import (
    HumanEvalEnv,
    HumanEvalItem,
    HumanEvalProblem,
    build_program,
    docstring_examples,
    extract_code,
    load_problems,
)
from agent_eval.runner import run_task, summarize
from agent_eval.strategies import get_strategy
from agent_eval.tools import ToolError
from tests.fakes import FunctionModel, act, final, is_review, turns

# --- HotpotQA ---------------------------------------------------------------

RECORD = {
    "_id": "q1",
    "question": "Which city is the birthplace of the composer of the opera Nixon in China?",
    "answer": "Worcester, Massachusetts",
    "type": "bridge",
    "level": "hard",
    "supporting_facts": [["Nixon in China", 0], ["John Adams (composer)", 0]],
    "context": [
        ["Nixon in China", ["Nixon in China is an opera by John Adams.", " It premiered in 1987."]],
        [
            "John Adams (composer)",
            ["John Adams (born 1947) is an American composer.", " He was born in Worcester, Massachusetts."],
        ],
        ["Richard Nixon", ["Richard Nixon was the 37th president."]],
    ],
}


def test_official_metric():
    assert normalize_answer("The  Beatles!") == "beatles"
    assert exact_match("the Worcester, Massachusetts.", "Worcester, Massachusetts") == 1.0
    assert f1_score("Worcester", "Worcester, Massachusetts") == pytest.approx(2 / 3)
    assert f1_score("no", "yes") == 0.0 and f1_score("yes", "yes") == 1.0
    assert f1_score("", "x") == 0.0


def test_subset_is_seeded_and_order_independent():
    records = [{"_id": f"id{i:03d}"} for i in range(50)]
    a = [r["_id"] for r in select_subset(records, n=10, seed=7)]
    b = [r["_id"] for r in select_subset(list(reversed(records)), n=10, seed=7)]
    assert a == b and len(set(a)) == 10
    assert a != [r["_id"] for r in select_subset(records, n=10, seed=8)]
    with pytest.raises(ValueError):
        select_subset(records, n=51)


def test_hotpot_tools_search_and_read():
    env = HotpotEnv(HotpotQuestion.from_record(RECORD))
    hits = env.tools["search"].call({"query": "composer John Adams born"}).splitlines()
    assert hits[0].startswith("John Adams (composer):")
    assert "Worcester" in env.tools["read_paragraph"].call({"title": "john adams (composer)"})
    assert "No paragraph matches" in env.search("zzz qqq")
    with pytest.raises(ToolError, match="available titles"):
        env.read_paragraph("Nixon")
    assert "## Richard Nixon" in env.describe()
    assert env.snapshot() is None


def hotpot_agent(system, messages, tools):
    if turns(messages) == 0:
        return act("search", query="Nixon in China opera composer")
    if turns(messages) == 1:
        return act("read_paragraph", title="John Adams (composer)")
    return final("Worcester, Massachusetts")


def test_hotpot_item_with_react_and_direct():
    item = HotpotItem(HotpotQuestion.from_record(RECORD))
    run = run_task(FunctionModel(hotpot_agent, native=False), item, benchmark="hotpotqa")
    assert run.passed and run.scores == {"em": 1.0, "f1": 1.0} and run.benchmark == "hotpotqa"

    model = FunctionModel(lambda s, m, t: "Worcester", native=False)
    run = run_task(model, item, strategy=get_strategy("direct"), benchmark="hotpotqa")
    assert not run.passed and run.scores["f1"] == pytest.approx(0.6667, abs=1e-4)
    assert "## John Adams (composer)" in model.calls[0]["messages"][0]["content"]
    summary = summarize([run])
    assert summary["scores"] == {"em": 0.0, "f1": 0.6667} and summary["benchmark"] == "hotpotqa"


# --- HumanEval --------------------------------------------------------------

PROBLEM = HumanEvalProblem(
    task_id="Test/0",
    prompt='def add(a: int, b: int) -> int:\n    """Add two numbers.\n    >>> add(2, 3)\n    5\n    """\n',
    entry_point="add",
    test="def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
    canonical_solution="    return a + b\n",
)
GOOD = "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_extract_and_build_program():
    assert extract_code("text\n```py\nx = 1\n```\n```python\ndef add(a, b):\n    pass\n```", "add").startswith(
        "def add"
    )
    assert extract_code("```\nprint(1)\n```", "add") == "print(1)\n"
    assert "def add" in build_program(PROBLEM, GOOD)
    assert build_program(PROBLEM, PROBLEM.canonical_solution).endswith("    return a + b\n")


def test_docstring_examples_and_feedback():
    assert [e.source.strip() for e in docstring_examples(PROBLEM.prompt, "add")] == ["add(2, 3)"]
    assert docstring_examples("def broken(:", "add") == []
    env = HumanEvalEnv(PROBLEM)
    assert env.feedback(GOOD) == (1.0, "1/1 docstring examples pass")
    assert env.feedback(BAD)[0] == 0.0
    assert "exit code 0" in env.tools["run_python"].call({"code": "print('hi')"})


@pytest.mark.parametrize("answer, passed", [(GOOD, True), (PROBLEM.canonical_solution, True), (BAD, False)])
def test_humaneval_grading(answer, passed):
    run = run_task(FunctionModel(lambda s, m, t: final(answer), native=False), HumanEvalItem(PROBLEM))
    assert run.passed is passed and run.scores == {"pass": float(passed)}


def test_humaneval_no_answer_and_infinite_loop():
    item = HumanEvalItem(PROBLEM, max_steps=1, timeout=1)
    run = run_task(FunctionModel(lambda s, m, t: act("run_python", code="print(1)"), native=False), item)
    assert not run.passed and run.checks[0].detail == "no final answer"
    loop = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    grade = item.grade(run.agent.__class__(final_answer=loop, stop_reason="final"), item.make_env())
    assert not grade.passed and "timed out" in grade.checks[0].detail


def test_reflexion_on_humaneval_uses_docstring_examples():
    def fn(system, messages, tools):
        if is_review(system):
            assert "docstring examples pass" in messages[0]["content"]
            return json.dumps({"score": 10 if "0/1" not in messages[0]["content"] else 1, "reflection": "use +"})
        return final(GOOD if "Lessons" in system else BAD)

    run = run_task(FunctionModel(fn, native=False), HumanEvalItem(PROBLEM), strategy=get_strategy("reflexion"))
    assert run.passed and run.agent.attempts == 2


# --- download and loading -----------------------------------------------------


def gz_lines(rows):
    buf = io.BytesIO()
    with gzip.open(buf, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return buf.getvalue()


def opener_for(payload: bytes):
    return lambda url: io.BytesIO(payload)


def test_download_verifies_checksum_and_count(tmp_path, monkeypatch):
    payload = gz_lines([PROBLEM.__dict__] * 2)
    ds = dl.Dataset(
        "humaneval", "http://x", "humaneval/HumanEval.jsonl.gz", records=2, sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setitem(dl.DATASETS, "humaneval", ds)
    [path] = dl.download(["humaneval"], tmp_path, opener=opener_for(payload))
    assert len(load_problems(path)) == 2
    assert ds.sha256 in (tmp_path / "CHECKSUMS").read_text()
    # Present and valid: no second download.
    dl.download(["humaneval"], tmp_path, opener=lambda url: (_ for _ in ()).throw(AssertionError("refetched")))
    items = load_benchmark("humaneval", tmp_path)
    assert [i.id for i in items] == ["Test/0", "Test/0"]

    bad = dl.Dataset("humaneval", "http://x", "other.jsonl.gz", records=2, sha256="0" * 64)
    with pytest.raises(dl.DownloadError, match="SHA-256 mismatch"):
        dl.fetch(bad, tmp_path, opener=opener_for(payload))
    assert not (tmp_path / "other.jsonl.gz").exists()

    wrong_count = dl.Dataset("humaneval", "http://x", "third.jsonl.gz", records=5)
    with pytest.raises(dl.DownloadError, match="expected 5 records"):
        dl.fetch(wrong_count, tmp_path, opener=opener_for(payload))


def test_hotpot_download_writes_the_seeded_subset(tmp_path, monkeypatch):
    records = [{**RECORD, "_id": f"q{i}"} for i in range(250)]
    payload = json.dumps(records).encode()
    monkeypatch.setitem(dl.DATASETS, "hotpotqa", dl.Dataset("hotpotqa", "http://x", "hotpotqa/dev.json", records=250))
    [subset] = dl.download(["hotpotqa"], tmp_path, opener=opener_for(payload))
    assert subset.name == "dev_distractor_200_seed1234.jsonl"
    items = load_benchmark("hotpotqa", tmp_path, limit=5)
    assert len(load_benchmark("hotpotqa", tmp_path)) == 200 and len(items) == 5
    assert "dev_distractor_200_seed1234.jsonl" in (tmp_path / "CHECKSUMS").read_text()


def test_download_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        dl.download(["imagenet"], tmp_path)

    def refuse(url):
        raise OSError("blocked by proxy")

    with pytest.raises(dl.DownloadError, match="blocked by proxy"):
        dl.fetch(dl.Dataset("x", "http://x", "x.json", records=1), tmp_path, opener=refuse)
    with pytest.raises(MissingDataError, match="agent-eval download hotpotqa"):
        load_benchmark("hotpotqa", tmp_path / "empty")
    with pytest.raises(ValueError, match="unknown benchmark"):
        load_benchmark("mmlu")
    assert len(load_benchmark("builtin", limit=3)) == 3
