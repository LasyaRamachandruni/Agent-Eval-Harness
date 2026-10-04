"""Before/after comparison of saved runs."""

import json

import pytest

from agent_eval import cli
from agent_eval.compare import compare_runs, find_saved_run, format_comparison
from agent_eval.defenses import resolve_defenses
from agent_eval.models import ScriptedModel
from agent_eval.runner import run_task, save_results, summarize
from agent_eval.tasks import load_tasks

INJECTION = {t.id: t for t in load_tasks("tasks/injection")}
FILES = {t.id: t for t in load_tasks("tasks/files")}


def replay(task, actions, defenses=None, name="m"):
    return run_task(ScriptedModel([json.dumps(a) for a in actions], name=name), task, defenses=defenses)


def baseline_runs():
    # Hijacked on the canary task, solves the file task.
    return [
        replay(INJECTION["inj-canary-word"], INJECTION["inj-canary-word"].injection["attack"]),
        replay(FILES["count-errors"], FILES["count-errors"].solution),
    ]


def defended_runs():
    # Resists the canary task, but now gets the file task wrong.
    d = resolve_defenses("hardened_prompt")
    return [
        replay(INJECTION["inj-canary-word"], INJECTION["inj-canary-word"].solution, d),
        replay(FILES["count-errors"], [{"final": "wrong"}], d),
    ]


def test_compare_reports_metric_changes_and_task_lists():
    diff = compare_runs(summarize(baseline_runs()), summarize(defended_runs()))
    assert diff["before"] == "m" and diff["after"] == "m +hardened_prompt"
    inj = diff["metrics"]["injection_resistance"]
    assert (inj["before"], inj["after"], inj["change"]) == (0.0, 1.0, 1.0)
    assert diff["metrics"]["success_rate"]["change"] == 0
    assert diff["fixed"] == ["inj-canary-word"]
    assert diff["broken"] == ["count-errors"]
    assert diff["newly_resisted"] == ["inj-canary-word"] and diff["newly_hijacked"] == []
    text = format_comparison(diff)
    assert "+100 pts" in text and "broken (1): count-errors" in text
    assert "no longer hijacked (1): inj-canary-word" in text


def test_compare_notes_different_task_sets():
    before = summarize(baseline_runs())
    after = summarize(defended_runs()[:1])
    diff = compare_runs(before, after)
    assert diff["only_before"] == ["count-errors"] and diff["shared_tasks"] == 1
    assert "headline numbers cover different task sets" in format_comparison(diff)


def test_find_saved_run_by_path_or_label(tmp_path):
    first = save_results(baseline_runs(), tmp_path)
    save_results(defended_runs(), tmp_path)
    assert find_saved_run(str(first)).model == "m"
    assert find_saved_run("m +hardened_prompt", tmp_path).defenses == ["hardened_prompt"]
    with pytest.raises(LookupError, match="no saved run"):
        find_saved_run("nobody", tmp_path)


def test_cli_compare(tmp_path, capsys):
    save_results(baseline_runs(), tmp_path)
    save_results(defended_runs(), tmp_path)
    assert cli.main(["compare", "m", "m +hardened_prompt", "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "before: m\nafter:  m +hardened_prompt" in out
    assert "injection resistance" in out and "fixed (1): inj-canary-word" in out
    with pytest.raises(SystemExit, match="no saved run"):
        cli.main(["compare", "m", "missing", "--results", str(tmp_path)])
