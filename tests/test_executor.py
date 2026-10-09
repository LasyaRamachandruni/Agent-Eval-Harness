"""Concurrent runner: concurrency limit, resume after a crash, caching, budget, and the CLI around it."""

import asyncio
import json
import threading
import time

import pytest

from agent_eval import cli
from agent_eval.estimate import estimate_run, item_tokens
from agent_eval.executor import (
    ResultLog,
    ResumeError,
    RunConfig,
    prepare_model,
    run_suite_async,
    run_suite_concurrently,
)
from agent_eval.models import ModelReply
from agent_eval.runner import find_run_dir, load_run_records
from agent_eval.strategies import get_strategy
from agent_eval.tasks import Task, load_tasks
from tests.fakes import FunctionModel, act, final, turns

TASKS = [
    Task.from_dict(
        {
            "id": f"t{i}",
            "prompt": f"What number is in n{i}.txt?",
            "files": {f"n{i}.txt": str(i)},
            "checks": [{"type": "answer_equals", "value": str(i)}],
        }
    )
    for i in range(6)
]


def reader(delay=0.0, price_tokens=(1000, 100)):
    """Reads the one file in the workspace, then answers with its content."""

    def fn(system, messages, tools):
        time.sleep(delay)
        if turns(messages) == 0:
            name = messages[0]["content"].split()[-1].rstrip("?")
            return ModelReply(act("read_file", path=name), *price_tokens)
        return ModelReply(final(messages[-1]["content"].split("\n", 1)[1]), *price_tokens)

    return FunctionModel(fn, name="openai:gpt-4o-mini", native=False)


def config(**kw):
    base = dict(
        model="openai:gpt-4o-mini", strategy="react", benchmark="builtin", repeats=2, params={"temperature": 0.7}
    )
    return RunConfig(**{**base, **kw})


def test_respects_the_concurrency_limit_and_keeps_order():
    live, peak, lock = [0], [0], threading.Lock()
    inner = reader(delay=0.02)

    def fn(system, messages, tools):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        try:
            return inner.fn(system, messages, tools)
        finally:
            with lock:
                live[0] -= 1

    model = FunctionModel(fn, name="m", native=False)
    runs = run_suite_concurrently(model, TASKS, strategy=get_strategy("react"), repeats=2, concurrency=3)
    assert all(r.passed for r in runs)
    assert [(r.task_id, r.trial) for r in runs] == [(t.id, k) for t in TASKS for k in range(2)]
    assert 1 < peak[0] <= 3


def test_resume_runs_only_what_is_missing(tmp_path):
    log = ResultLog(tmp_path / "run", config())
    first = run_suite_concurrently(reader(), TASKS[:2], strategy=get_strategy("react"), repeats=2, result_log=log)
    assert len(first) == 4
    # Simulate a crash mid-write: a torn last line.
    with open(log.path, "a") as f:
        f.write('{"task_id": "t2", "trial"')
    model = reader()
    calls_before = len(model.calls)
    log2 = ResultLog(tmp_path / "run", config())
    runs = run_suite_concurrently(model, TASKS[:3], strategy=get_strategy("react"), repeats=2, result_log=log2)
    assert [(r.task_id, r.trial) for r in runs] == [(t.id, k) for t in TASKS[:3] for k in range(2)]
    assert len(model.calls) - calls_before == 4  # only t2 x 2 trials, two calls each
    lines = [json.loads(line) for line in log2.path.read_text().splitlines()]
    assert len(lines) == 6 and all(r["passed"] for r in lines)


def test_resume_refuses_a_different_configuration(tmp_path):
    ResultLog(tmp_path / "run", config())
    with pytest.raises(ResumeError, match="repeats"):
        ResultLog(tmp_path / "run", config(repeats=5))


def test_budget_stops_starting_new_runs():
    # 1000 in + 100 out tokens per call at gpt-4o-mini prices = $0.00021 per call, 2 calls per run.
    runs = run_suite_concurrently(
        reader(), TASKS, strategy=get_strategy("react"), repeats=1, concurrency=1, budget_usd=0.0008
    )
    assert len(runs) == 2


def test_cache_makes_a_rerun_free(tmp_path):
    model = reader()
    wrapped, cache = prepare_model(model, tmp_path / "c.sqlite", max_retries=0)
    first = run_suite_concurrently(wrapped, TASKS[:2], strategy=get_strategy("react"), repeats=2)
    calls = len(model.calls)
    again = run_suite_concurrently(wrapped, TASKS[:2], strategy=get_strategy("react"), repeats=2)
    cache.close()
    assert len(model.calls) == calls == 8  # 2 tasks x 2 trials x 2 calls, nothing new the second time
    assert all(r.agent.cached_calls == 2 and r.agent.billed_input_tokens == 0 for r in again)
    assert [r.agent.final_answer for r in again] == [r.agent.final_answer for r in first]


def test_argument_validation():
    with pytest.raises(ValueError):
        asyncio.run(run_suite_async(reader(), TASKS, strategy=get_strategy("react"), repeats=0))
    with pytest.raises(ValueError):
        asyncio.run(run_suite_async(reader(), TASKS, strategy=get_strategy("react"), concurrency=0))


def test_estimates_scale_with_strategy_and_repeats():
    tasks = load_tasks("tasks")
    react = estimate_run("openai:gpt-4o-mini", tasks, "react", "builtin", 1)
    five = estimate_run("openai:gpt-4o-mini", tasks, "react", "builtin", 5)
    reflexion = estimate_run("openai:gpt-4o-mini", tasks, "reflexion", "builtin", 1)
    assert five.runs == 5 * react.runs == 200
    assert five.input_tokens == pytest.approx(5 * react.input_tokens, rel=1e-3)
    assert reflexion.input_tokens > react.input_tokens
    assert react.cost_usd and react.cost_usd > 0
    assert estimate_run("mystery:model", tasks, "react", "builtin", 1).cost_usd is None
    direct_in, _ = item_tokens(tasks[0], "direct", "builtin")
    assert direct_in > 0
    total = react + reflexion
    assert total.runs == 80


# --- CLI ----------------------------------------------------------------------


def test_cli_run_strategy_resume_and_estimate(tmp_path, monkeypatch, capsys):
    model = reader()
    monkeypatch.setattr(cli, "load_model", lambda spec, params=None: model)
    out = tmp_path / "results"
    argv = ["run", "--model", "openai:gpt-4o-mini", "--only", "arith-simple,find-owner", "--out", str(out)]
    assert cli.main([*argv, "--strategy", "reflexion", "--repeats", "2", "--temperature", "0.2"]) == 0
    text = capsys.readouterr().out
    assert "openai:gpt-4o-mini [reflexion]: builtin, 2 item(s) x 2 trial(s)" in text
    run_dir = find_run_dir(out)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["config"]["params"]["temperature"] == 0.2 and meta["config"]["strategy"] == "reflexion"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["strategy"] == "reflexion" and summary["runs"] == 4

    # Resuming a finished run does nothing new.
    calls = len(model.calls)
    resume = [*argv, "--strategy", "reflexion", "--repeats", "2", "--temperature", "0.2", "--resume", str(run_dir)]
    assert cli.main(resume) == 0
    assert len(model.calls) == calls and len(load_run_records(run_dir)) == 4
    with pytest.raises(SystemExit, match="different configuration"):
        cli.main([*argv, "--repeats", "3", "--resume", str(run_dir)])

    assert cli.main(["estimate", "--model", "openai:gpt-4o-mini", "--model", "ollama:llama3.1", "--repeats", "5"]) == 0
    text = capsys.readouterr().out
    assert "200 runs" in text and "total" in text


def test_cli_rejects_bad_options(tmp_path, capsys):
    with pytest.raises(SystemExit, match="agent-eval download humaneval"):
        cli.main(["run", "--model", "openai:x", "--benchmark", "humaneval", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="only applies to the builtin"):
        cli.main(["estimate", "--model", "openai:x", "--benchmark", "hotpotqa", "--category", "files"])
    with pytest.raises(SystemExit, match="temperature"):
        cli.main(["run", "--model", "openai:x", "--temperature", "5", "--out", str(tmp_path)])


def test_cli_download_reports_errors(tmp_path, monkeypatch):
    def refuse(names, data_dir, force=False):
        from agent_eval.benchmarks.download import DownloadError

        raise DownloadError("could not download hotpotqa: blocked")

    monkeypatch.setattr(cli, "download", refuse)
    with pytest.raises(SystemExit, match="blocked"):
        cli.main(["download", "hotpotqa"])
    monkeypatch.setattr(cli, "download", lambda names, data_dir, force=False: [tmp_path / "x.jsonl"])
    assert cli.main(["download", "all", "--data-dir", str(tmp_path)]) == 0


def test_cli_reports_config_errors(monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_CONCURRENCY", "zero")
    with pytest.raises(SystemExit, match="configuration error"):
        cli.main(["list"])


def test_report_shows_strategy_and_f1(tmp_path):
    from agent_eval.agent import AgentResult
    from agent_eval.grading import CheckResult
    from agent_eval.report import collect_runs, leaderboard, render_html, render_markdown
    from agent_eval.runner import TaskRun, save_results

    def run(em, strategy):
        agent = AgentResult(final_answer="x", stop_reason="final", strategy=strategy)
        check = CheckResult("exact_match", bool(em))
        return TaskRun(
            "q1",
            "hotpotqa",
            "openai:gpt-4o-mini",
            bool(em),
            [check],
            agent,
            scores={"em": em, "f1": 0.5},
            strategy=strategy,
            benchmark="hotpotqa",
        )

    save_results([run(1.0, "react")], tmp_path)
    save_results([run(0.0, "reflexion")], tmp_path)
    runs = collect_runs(tmp_path)
    rows = leaderboard(runs)
    assert [r["model"] for r in rows] == ["openai:gpt-4o-mini", "openai:gpt-4o-mini [reflexion]"]
    assert rows[0]["f1"] == 0.5 and rows[1]["benchmark"] == "hotpotqa"
    assert "<th>F1</th>" in render_html(runs) and "| F1 |" in render_markdown(runs)
