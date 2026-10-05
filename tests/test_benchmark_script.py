"""scripts/run_benchmark.sh, in dry-run mode (no model calls)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_benchmark.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def dry_run(*models, **env):
    full_env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable, **env}
    out = subprocess.run(["bash", str(SCRIPT), *models], env=full_env, capture_output=True, text=True, check=True)
    return out.stdout


def commands(text):
    return [line[2:] for line in text.splitlines() if line.startswith("+ ")]


def test_dry_run_plans_baseline_defended_report_and_compare():
    text = dry_run("openai:gpt-4o-mini", REPEATS="2", OUT="results/x")
    assert "40 tasks x 2 trial(s)" in text and "160 agent runs per model" in text
    cmds = commands(text)
    assert cmds[0].endswith("-m agent_eval check --model openai:gpt-4o-mini")
    assert cmds[1].endswith("run --model openai:gpt-4o-mini --repeats 2 --out results/x")
    assert cmds[2].endswith("run --model openai:gpt-4o-mini --repeats 2 --defense all --out results/x")
    assert "report results/x" in cmds[3] and "--markdown results/x/leaderboard.md" in cmds[3]
    assert "compare openai:gpt-4o-mini openai:gpt-4o-mini\\ +hardened_prompt+tag_untrusted" in cmds[4]
    assert cmds[4].endswith("--markdown results/x/compare_openai_gpt-4o-mini.md")
    assert len(cmds) == 5


def test_dry_run_without_defenses_skips_the_second_run():
    cmds = commands(dry_run("ollama:llama3.1", "openai:gpt-4o", DEFENSE="none"))
    assert sum(" run --model " in c for c in cmds) == 2
    assert not any("--defense" in c or " compare " in c for c in cmds)
    assert "--model ollama:llama3.1 --model openai:gpt-4o" in cmds[0]


def test_default_models():
    cmds = commands(dry_run())
    assert "--model anthropic:claude-haiku-4-5 --model anthropic:claude-sonnet-4-5" in cmds[0]
