"""The examples in examples/ must keep working."""

import runpy
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_library_usage_example(capsys):
    module = runpy.run_path(str(EXAMPLES / "library_usage.py"))
    summary = module["main"]()
    assert summary["passed"] == summary["runs"] == 3
    out = capsys.readouterr().out
    assert "trial 3: PASS answer='4'" in out
    assert "3/3 runs passed" in out


def test_example_task_is_valid():
    from agent_eval.validate import check_tasks

    reports = check_tasks(EXAMPLES / "tasks")
    assert reports and all(r.ok and not r.warnings for r in reports)
