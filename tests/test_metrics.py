import pytest

from agent_eval.metrics import (
    pass_at_k,
    pass_hat_k,
    success_variance,
    suite_pass_at_k,
    suite_pass_hat_k,
)


def test_k1_is_plain_success_rate():
    assert pass_at_k(4, 3, 1) == pytest.approx(0.75)
    assert pass_hat_k(4, 3, 1) == pytest.approx(0.75)


def test_known_values():
    # 5 trials, 2 passed, pick 2: C(3,2)/C(5,2) = 3/10 fail both -> pass@2 = 0.7
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)
    # both succeed: C(2,2)/C(5,2) = 1/10
    assert pass_hat_k(5, 2, 2) == pytest.approx(0.1)


def test_extremes():
    assert pass_at_k(3, 0, 3) == 0.0
    assert pass_hat_k(3, 3, 3) == 1.0
    assert pass_at_k(3, 1, 3) == 1.0  # any 3 of 3 includes the success
    assert pass_hat_k(3, 2, 3) == 0.0


def test_pass_at_k_rises_and_pass_hat_k_falls():
    n, c = 8, 5
    at = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    hat = [pass_hat_k(n, c, k) for k in range(1, n + 1)]
    assert at == sorted(at)
    assert hat == sorted(hat, reverse=True)


@pytest.mark.parametrize("n,c,k", [(3, 4, 1), (3, 1, 0), (3, 1, 4), (0, 0, 1)])
def test_bad_arguments(n, c, k):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_variance():
    assert success_variance(4, 4) == 0.0
    assert success_variance(4, 0) == 0.0
    assert success_variance(4, 2) == 0.25


def test_suite_averages():
    outcomes = {"a": [True, True], "b": [True, False]}
    assert suite_pass_at_k(outcomes, 2) == pytest.approx(1.0)
    assert suite_pass_hat_k(outcomes, 2) == pytest.approx(0.5)
    assert suite_pass_at_k({}, 1) == 0.0
