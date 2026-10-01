"""Injection-resistance scoring, using the bundled injection suite and scripted agents."""

import json

from agent_eval import cli
from agent_eval.cli import format_summary
from agent_eval.models import ScriptedModel
from agent_eval.runner import injection_summary, run_task, summarize
from agent_eval.tasks import Task, load_tasks

INJECTION = load_tasks("tasks/injection")


def replies(actions):
    return [json.dumps(a) for a in actions]


def test_resistant_agent_scores_100_percent():
    runs = [run_task(ScriptedModel(replies(t.solution)), t) for t in INJECTION]
    inj = summarize(runs)["injection"]
    assert inj["runs"] == len(INJECTION)
    assert inj["resistance_rate"] == 1.0
    assert inj["hijacked_tasks"] == []
    assert inj["completed_and_resisted"] == len(INJECTION)


def test_obedient_agent_is_hijacked_everywhere():
    runs = [run_task(ScriptedModel(replies(t.injection["attack"])), t) for t in INJECTION]
    inj = injection_summary(runs)
    assert inj["resisted"] == 0 and inj["resistance_rate"] == 0.0
    assert inj["hijacked_tasks"] == sorted(t.id for t in INJECTION)


def test_resistance_counts_trials_separately():
    task = INJECTION[0]
    runs = [
        run_task(ScriptedModel(replies(task.solution)), task, 0),
        run_task(ScriptedModel(replies(task.injection["attack"])), task, 1),
    ]
    s = summarize(runs)
    assert s["injection"]["resistance_rate"] == 0.5
    assert s["injection"]["resisted_every_trial"] == 0
    assert s["per_task"][task.id]["resisted"] == 1


def test_no_injection_section_without_injection_tasks():
    task = Task.from_dict({"id": "t", "prompt": "p", "checks": [{"type": "answer_equals", "value": "1"}]})
    s = summarize([run_task(ScriptedModel(['{"final": "1"}']), task)])
    assert s["injection"] is None
    assert "injection" not in format_summary(s)


def test_cli_reports_hijacked_runs(tmp_path, monkeypatch, capsys):
    task = next(t for t in INJECTION if t.id == "inj-canary-word")
    model = ScriptedModel(replies(task.injection["attack"]))
    monkeypatch.setattr(cli, "load_model", lambda spec: model)
    code = cli.main([
        "run", "--model", "scripted", "--tasks", "tasks/injection/canary-word.json",
        "--out", str(tmp_path), "-v",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "[HIJACKED] inj-canary-word" in out
    assert "answer_not_contains" in out
    assert "injection resistance 0/1 runs (0%)" in out and "hijacked by: inj-canary-word" in out
