import json

import pytest

from agent_eval import cli
from agent_eval.agent import AgentResult
from agent_eval.models import ModelReply, ScriptedModel
from agent_eval.pricing import estimate_cost, load_prices, price_for
from agent_eval.runner import TaskRun, summarize


def test_exact_and_prefix_lookup():
    assert price_for("openai:gpt-4o-mini") == (0.15, 0.60)
    # dated model ids fall back to the longest matching prefix
    assert price_for("anthropic:claude-sonnet-4-5-20250929") == (3.00, 15.00)
    assert price_for("ollama:llama3.1") == (0.0, 0.0)
    assert price_for("mystery:model") is None


def test_estimate_cost():
    # 1M input at $3 + 100k output at $15 = 3 + 1.5
    assert estimate_cost("anthropic:claude-sonnet-4-5", 1_000_000, 100_000) == pytest.approx(4.5)
    assert estimate_cost("mystery:model", 10, 10) is None


def test_custom_price_file(tmp_path):
    f = tmp_path / "prices.json"
    f.write_text(json.dumps({"mystery:model": [1, 2]}))
    table = load_prices(f)
    assert estimate_cost("mystery:model", 1_000_000, 1_000_000, table) == pytest.approx(3.0)
    f.write_text(json.dumps({"bad": 5}))
    with pytest.raises(ValueError):
        load_prices(f)


def _run(model, tokens_in, tokens_out, passed=True):
    agent = AgentResult("x", stop_reason="final", input_tokens=tokens_in, output_tokens=tokens_out)
    return TaskRun("t", "files", model, passed, [], agent)


def test_summary_includes_cost():
    runs = [_run("openai:gpt-4o-mini", 1_000_000, 0), _run("openai:gpt-4o-mini", 0, 1_000_000)]
    s = summarize(runs)
    assert s["total_cost_usd"] == pytest.approx(0.75)
    assert s["avg_cost_per_run_usd"] == pytest.approx(0.375)
    assert s["per_task"]["t"]["avg_cost_usd"] == pytest.approx(0.375)
    assert summarize([_run("mystery:model", 5, 5)])["total_cost_usd"] is None


class CountingModel(ScriptedModel):
    """Scripted model that also reports token usage, so cost shows up end to end."""

    def complete(self, system, messages):
        reply = super().complete(system, messages)
        return ModelReply(reply.text, 1000, 100)


def test_cli_run_with_repeats_and_prices(tmp_path, monkeypatch, capsys):
    task = {"id": "t", "prompt": "p", "checks": [{"type": "answer_equals", "value": "1"}]}
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "t.json").write_text(json.dumps(task))
    (tmp_path / "prices.json").write_text(json.dumps({"scripted": [3, 15]}))
    model = CountingModel([json.dumps({"final": "1"})] * 2 + [json.dumps({"final": "2"})])
    monkeypatch.setattr(cli, "load_model", lambda spec: model)

    code = cli.main([
        "run", "--model", "scripted", "--tasks", str(tmp_path / "tasks"), "--repeats", "3",
        "--prices", str(tmp_path / "prices.json"), "--out", str(tmp_path / "results"),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "2/3 runs passed" in out and "pass@3 100%" in out and "pass^3 0%" in out
    summary = json.loads(next((tmp_path / "results").glob("*/summary.json")).read_text())
    # 3 runs x (1000 in * $3 + 100 out * $15) / 1M = 3 x 0.0045
    assert summary["total_cost_usd"] == pytest.approx(0.0135)
