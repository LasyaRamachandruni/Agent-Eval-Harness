"""scripts/run_real_models.sh in dry-run mode: the plan it would execute, no model calls."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_real_models.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def dry_run(**env):
    full_env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable, **env}
    out = subprocess.run(["bash", str(SCRIPT)], env=full_env, capture_output=True, text=True, check=True)
    return [line[2:] for line in out.stdout.splitlines() if line.startswith("+ ")]


def test_default_plan_covers_every_model_benchmark_and_strategy():
    cmds = dry_run()
    runs = [c for c in cmds if " run --model " in c]
    assert len(runs) == 4 * 3 * 4  # models x benchmarks x strategies
    assert all("--repeats 5" in c and "--temperature 0.7" in c for c in runs)
    assert any("ollama:llama3.1" in c and "--concurrency 1" in c for c in runs)
    assert cmds[0].endswith("download hotpotqa humaneval")
    assert any(" plan " in c for c in cmds)
    reports = [c for c in cmds if " report " in c]
    assert [r.split(" report ")[1].split()[0] for r in reports] == [
        f"results/real/{b}" for b in ("builtin", "hotpotqa", "humaneval")
    ]


def test_settings_narrow_the_plan():
    cmds = dry_run(MODELS="openai:gpt-4o-mini", BENCHMARKS="builtin", STRATEGIES="react", REPEATS="2", BUDGET="3")
    runs = [c for c in cmds if " run --model " in c]
    assert len(runs) == 1
    assert "--repeats 2" in runs[0] and "--budget 3" in runs[0]
    assert "--run-dir results/real/builtin/openai_gpt-4o-mini_react_json" in runs[0]
    assert not any("download" in c for c in cmds)
