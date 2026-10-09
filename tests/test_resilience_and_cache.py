import random
import urllib.error
from types import SimpleNamespace as NS

import pytest

from agent_eval.cache import CachingModel, ResponseCache, cache_key, sample_scope
from agent_eval.models import GenerationParams, ModelReply, ToolSpec
from agent_eval.resilience import RetryingModel, is_retryable, retry_after
from tests.fakes import FunctionModel


class RateLimitError(Exception):
    def __init__(self, headers=None):
        super().__init__("slow down")
        self.status_code = 429
        self.response = NS(headers=headers or {})


class BadRequestError(Exception):
    status_code = 400


def flaky(failures):
    """A model that raises the given exceptions in order, then answers."""
    pending = list(failures)

    def fn(system, messages, tools):
        if pending:
            raise pending.pop(0)
        return "ok"

    return FunctionModel(fn)


def test_retryable_classification():
    assert is_retryable(RateLimitError())
    assert is_retryable(type("OverloadedError", (Exception,), {})())
    assert is_retryable(type("APIStatusError", (Exception,), {"status_code": 529})())
    assert is_retryable(urllib.error.URLError("refused"))
    assert is_retryable(TimeoutError())
    assert not is_retryable(BadRequestError())
    assert not is_retryable(ValueError("bad"))


def test_retry_after_header():
    assert retry_after(RateLimitError({"retry-after": "7"})) == 7.0
    assert retry_after(RateLimitError({"retry-after": "soon"})) is None
    assert retry_after(ValueError()) is None


def test_retries_rate_limits_with_backoff_then_succeeds():
    waits = []
    model = RetryingModel(flaky([RateLimitError(), RateLimitError()]), sleep=waits.append, rng=random.Random(0))
    assert model.complete("", []).text == "ok"
    assert model.retries == 2 and len(waits) == 2
    assert 0 <= waits[0] <= 1.0 and 0 <= waits[1] <= 2.0  # full jitter on 1s, 2s


def test_honours_retry_after_capped_at_max_delay():
    waits = []
    model = RetryingModel(flaky([RateLimitError({"retry-after": "120"})]), sleep=waits.append, max_delay=30)
    model.complete("", [])
    assert waits == [30]


def test_gives_up_after_max_retries_and_never_retries_bad_requests():
    model = RetryingModel(flaky([RateLimitError()] * 5), max_retries=2, sleep=lambda s: None)
    with pytest.raises(RateLimitError):
        model.complete("", [])
    model = RetryingModel(flaky([BadRequestError()]), sleep=lambda s: None)
    with pytest.raises(BadRequestError):
        model.complete("", [])
    assert model.retries == 0


def test_retrying_model_forwards_native_calls():
    model = RetryingModel(flaky([RateLimitError()]), sleep=lambda s: None)
    assert model.complete_with_tools("", [], []).text == "ok"


def test_cache_key_depends_on_every_input():
    base = ("m", {"temperature": 0.7}, "sys", [{"role": "user", "content": "hi"}], None, 0)
    key = cache_key(*base)
    assert key == cache_key(*base)
    tools = [ToolSpec("t", "d", {})]
    variants = [
        ("m2", *base[1:]),
        (base[0], {"temperature": 0.0}, *base[2:]),
        (*base[:2], "other", *base[3:]),
        (*base[:3], [{"role": "user", "content": "bye"}], *base[4:]),
        (*base[:4], tools, 0),
        (*base[:5], 1),
    ]
    assert len({cache_key(*v) for v in variants} | {key}) == 7


def test_caching_model_hits_misses_and_persists(tmp_path):
    counter = {"n": 0}

    def fn(system, messages, tools):
        counter["n"] += 1
        return ModelReply(f"reply {counter['n']}", 100, 10)

    inner = FunctionModel(fn)
    inner.params = GenerationParams(temperature=0.7)
    model = CachingModel(inner, ResponseCache(tmp_path / "c.sqlite"))
    msgs = [{"role": "user", "content": "q"}]
    first = model.complete("s", msgs)
    again = model.complete("s", msgs)
    assert (first.text, first.cached) == ("reply 1", False)
    assert (again.text, again.cached, again.input_tokens) == ("reply 1", True, 100)
    with sample_scope(1):
        assert model.complete("s", msgs).text == "reply 2"  # a new trial is a new sample
    assert (model.hits, model.misses) == (1, 2)

    reopened = CachingModel(inner, ResponseCache(tmp_path / "c.sqlite"))
    assert reopened.complete("s", msgs).cached
    assert len(reopened.cache) == 2
    model.cache.close()
    reopened.cache.close()


def test_cache_round_trips_tool_calls(tmp_path):
    from tests.fakes import native_call

    model = CachingModel(
        FunctionModel(lambda s, m, t: native_call("read_file", path="a")), ResponseCache(tmp_path / "c")
    )
    model.complete_with_tools("s", [], [])
    hit = model.complete_with_tools("s", [], [])
    assert hit.cached and hit.tool_calls[0].args == {"path": "a"}
    model.cache.close()
