"""Reliability metrics for repeated runs.

When every task is run n times we can ask two different questions:

- pass@k: if I sample k attempts, how likely is it that AT LEAST ONE succeeds?
  (useful when you can retry or pick the best attempt)
- pass^k: if I sample k attempts, how likely is it that ALL of them succeed?
  (useful when the agent must be dependable every single time)

Both use the unbiased estimators over n trials with c successes:

    pass@k = 1 - C(n - c, k) / C(n, k)
    pass^k = C(c, k) / C(n, k)

With k = 1 both reduce to the plain success rate c / n. As k grows, pass@k
rises and pass^k falls; the gap between them shows how inconsistent an agent is.
"""

from __future__ import annotations

from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k attempts (drawn from n trials, c passed) succeeds."""
    _check(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Probability that all k attempts (drawn from n trials, c passed) succeed."""
    _check(n, c, k)
    return comb(c, k) / comb(n, k)


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n) or not (1 <= k <= n):
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n (got n={n}, c={c}, k={k})")


def success_variance(n: int, c: int) -> float:
    """Variance of a task's pass/fail outcome across trials: p * (1 - p).

    0.0 means the task always passes or always fails (consistent); the maximum,
    0.25, means it is a coin flip.
    """
    if n == 0:
        return 0.0
    p = c / n
    return p * (1 - p)


def suite_pass_at_k(outcomes: dict[str, list[bool]], k: int) -> float:
    """Average pass@k over tasks. `outcomes` maps task id -> list of pass/fail per trial."""
    return _mean([pass_at_k(len(o), sum(o), k) for o in outcomes.values()])


def suite_pass_hat_k(outcomes: dict[str, list[bool]], k: int) -> float:
    """Average pass^k over tasks."""
    return _mean([pass_hat_k(len(o), sum(o), k) for o in outcomes.values()])


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
