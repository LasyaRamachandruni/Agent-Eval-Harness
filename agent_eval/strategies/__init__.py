"""Agent strategies, selectable with --strategy.

    direct      one model call, no tools (baseline)
    react       tool-using loop: think, act, observe (port of Basic_-React_Agent_Langchain)
    reflection  react, then critique-and-revise rounds (port of Langgraph_Reflection_Agent)
    reflexion   fresh retries with verbal lessons, grounding check and attempt score
                (port of Neural-Reflexion-Agent)

Every strategy except `direct` works with both protocols: "json" (one JSON
action per turn, any chat model) and "native" (the provider's tool calling).
"""

from __future__ import annotations

from typing import Any

from .base import PROTOCOLS, AgentLoop, Protocol, Strategy
from .direct import DirectStrategy
from .react import ReactStrategy
from .reflection import ReflectionStrategy
from .reflexion import ReflexionStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    "direct": DirectStrategy,
    "react": ReactStrategy,
    "reflection": ReflectionStrategy,
    "reflexion": ReflexionStrategy,
}


def get_strategy(name: str, protocol: Protocol = "json", **options: Any) -> Strategy:
    """Build a strategy by name; `options` go to its constructor (e.g. trials=3)."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r} (choose from: {', '.join(STRATEGIES)})")
    return STRATEGIES[name](protocol=protocol, **options)


__all__ = [
    "PROTOCOLS",
    "STRATEGIES",
    "AgentLoop",
    "DirectStrategy",
    "Protocol",
    "ReactStrategy",
    "ReflectionStrategy",
    "ReflexionStrategy",
    "Strategy",
    "get_strategy",
]
