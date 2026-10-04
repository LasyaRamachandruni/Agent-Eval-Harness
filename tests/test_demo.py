"""The offline demo agents run the bundled tasks and feed the report."""

from agent_eval import cli
from agent_eval.demo import run_demo_agent
from agent_eval.runner import summarize
from agent_eval.tasks import load_tasks

TASKS = load_tasks("tasks")


def test_careful_agent_passes_everything():
    s = summarize(run_demo_agent("scripted:careful", TASKS))
    assert s["success_rate"] == 1.0
    assert s["injection"]["resistance_rate"] == 1.0


def test_gullible_agent_is_hijacked_only_on_injection_tasks():
    runs = run_demo_agent("scripted:gullible", TASKS)
    assert all(r.passed for r in runs if r.category != "injection")
    inj = [r for r in runs if r.category == "injection"]
    assert inj and all(r.resisted is False and r.failure == "followed_injection" for r in inj)


def test_demo_then_report(tmp_path, capsys):
    out = tmp_path / "demo"
    assert cli.main(["demo", "--out", str(out), "--category", "injection,files"]) == 0
    assert cli.main(["report", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.index("scripted:careful") < text.index("scripted:gullible", text.index("#"))
    page = (out / "report.html").read_text()
    assert "followed_injection" in page and "hijacked" in page


def test_defenses_stop_the_gullible_demo_agent():
    from agent_eval.defenses import resolve_defenses

    inj = [t for t in TASKS if t.category == "injection"]
    for spec in ("hardened_prompt", "tag_untrusted"):
        runs = run_demo_agent("scripted:gullible", inj, resolve_defenses(spec))
        assert all(r.passed and r.resisted for r in runs)
        assert all(r.defenses == [spec] for r in runs)


def test_demo_before_and_after_a_defense(tmp_path, capsys):
    out = tmp_path / "demo"
    assert cli.main(["demo", "--out", str(out), "--category", "injection"]) == 0
    assert cli.main(["demo", "--out", str(out), "--category", "injection", "--defense", "all"]) == 0
    assert 'agent-eval compare scripted:gullible "scripted:gullible +hardened_prompt+tag_untrusted"' in capsys.readouterr().out
    assert cli.main(["compare", "scripted:gullible", "scripted:gullible +hardened_prompt+tag_untrusted",
                     "--results", str(out)]) == 0
    text = capsys.readouterr().out
    assert "injection resistance" in text and "+100 pts" in text
    assert "no longer hijacked (10)" in text and "broken (0)" in text
