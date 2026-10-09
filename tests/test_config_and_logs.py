import json
import logging

import pytest

from agent_eval.config import ConfigError, Settings
from agent_eval.logs import JsonFormatter, TextFormatter, configure_logging, get_logger


def test_defaults_when_env_is_empty():
    s = Settings.from_env({})
    assert s.concurrency == 4 and s.max_retries == 6
    assert str(s.data_dir) == "data" and s.log_format == "json"


def test_reads_and_normalises_values():
    s = Settings.from_env(
        {
            "AGENT_EVAL_CONCURRENCY": "16",
            "AGENT_EVAL_LOG_LEVEL": "debug",
            "AGENT_EVAL_LOG_FORMAT": "TEXT",
            "AGENT_EVAL_DATA_DIR": "/srv/data",
            "OLLAMA_HOST": "http://gpu:11434/",
        }
    )
    assert s.concurrency == 16 and s.log_level == "DEBUG" and s.log_format == "text"
    assert str(s.data_dir) == "/srv/data" and s.ollama_host == "http://gpu:11434"


@pytest.mark.parametrize(
    "env, message",
    [
        ({"AGENT_EVAL_CONCURRENCY": "lots"}, "must be an integer"),
        ({"AGENT_EVAL_CONCURRENCY": "0"}, "at least 1"),
        ({"AGENT_EVAL_LOG_FORMAT": "xml"}, "must be one of"),
    ],
)
def test_bad_values_raise_config_error(env, message):
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)


def _record(**fields):
    record = logging.LogRecord("agent_eval.x", logging.WARNING, __file__, 1, "retrying", None, None)
    record.fields = fields
    return record


def test_json_formatter_merges_fields():
    data = json.loads(JsonFormatter().format(_record(attempt=2, wait_s=1.5)))
    assert data["event"] == "retrying" and data["level"] == "warning"
    assert data["attempt"] == 2 and data["wait_s"] == 1.5 and "ts" in data


def test_text_formatter():
    assert TextFormatter().format(_record(attempt=2)) == "warning agent_eval.x retrying attempt=2"


def test_configure_logging_writes_json_to_stderr(capsys):
    configure_logging("INFO", "json")
    get_logger("tests").info("hello", extra={"fields": {"k": 1}})
    line = capsys.readouterr().err.strip()
    assert json.loads(line)["k"] == 1
    configure_logging("WARNING", "json")
