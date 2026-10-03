"""Leaderboard and HTML report built from saved runs of scripted agents."""

import json

from agent_eval.models import ScriptedModel
from agent_eval.report import collect_runs, latest_per_model, leaderboard
from agent_eval.runner import TaskRun, run_task, save_results
from agent_eval.tasks import Task

TASK = Task.from_dict(
    {
        "id": "read-a",
        "category": "files",
        "prompt": "What is in a.txt?",
        "files": {"a.txt": "hello"},
        "checks": [{"type": "answer_equals", "value": "hello"}],
    }
)


def call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def final(answer):
    return json.dumps({"final": answer})


def scripted_run(name: str, answer: str, trial: int = 0) -> TaskRun:
    model = ScriptedModel([call("read_file", path="a.txt"), final(answer)], name=name)
    return run_task(model, TASK, trial)


def save(tmp_path, runs, stamp):
    out = save_results(runs, tmp_path)
    # save_results names directories by the current time; rename for a stable order.
    target = out.parent / f"{stamp}_{runs[0].model.replace(':', '_')}"
    out.rename(target)
    return target


def test_collect_runs_reads_a_folder_or_a_single_run(tmp_path):
    a = save(tmp_path, [scripted_run("scripted:a", "hello")], "20261001T000000Z")
    save(tmp_path, [scripted_run("scripted:b", "nope")], "20261002T000000Z")
    (tmp_path / "junk").mkdir()  # no summary.json: ignored
    runs = collect_runs(tmp_path)
    assert [r.model for r in runs] == ["scripted:a", "scripted:b"]
    assert [r.model for r in collect_runs(a)] == ["scripted:a"]
    assert collect_runs(tmp_path / "junk") == []


def test_latest_run_of_each_model_wins(tmp_path):
    save(tmp_path, [scripted_run("scripted:a", "nope")], "20261001T000000Z")
    save(tmp_path, [scripted_run("scripted:a", "hello")], "20261003T000000Z")
    runs = collect_runs(tmp_path)
    [latest] = latest_per_model(runs)
    assert latest.name.startswith("20261003")
    assert len(leaderboard(runs, latest_only=False)) == 2


def test_leaderboard_ranks_by_success_then_consistency(tmp_path):
    save(tmp_path, [scripted_run("scripted:bad", "nope")], "20261001T000000Z")
    save(
        tmp_path,
        [scripted_run("scripted:flaky", "hello", 0), scripted_run("scripted:flaky", "nope", 1)],
        "20261001T000001Z",
    )
    save(
        tmp_path,
        [scripted_run("scripted:good", "hello", 0), scripted_run("scripted:good", "hello", 1)],
        "20261001T000002Z",
    )
    rows = leaderboard(collect_runs(tmp_path))
    assert [r["model"] for r in rows] == ["scripted:good", "scripted:flaky", "scripted:bad"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    good, flaky, bad = rows
    assert good["success_rate"] == 1.0 and good["pass_hat_k"] == 1.0 and good["repeats"] == 2
    assert flaky["success_rate"] == 0.5 and flaky["pass_hat_k"] == 0.0
    assert bad["failures"] == {"wrong_answer": 1}
    assert good["injection_resistance"] is None  # no injection tasks in these runs
    assert good["avg_cost_per_run_usd"] == 0.0  # scripted models are priced at zero
