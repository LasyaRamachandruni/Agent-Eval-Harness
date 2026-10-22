import re
from pathlib import Path

import pytest

from agent_eval import __version__
from agent_eval.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"agent-eval {__version__}"


def test_package_version_matches_pyproject():
    # read with a regex: tomllib only exists from Python 3.11, and CI also runs 3.10
    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == __version__
