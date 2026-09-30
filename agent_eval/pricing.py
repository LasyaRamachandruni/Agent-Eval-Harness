"""Rough cost estimates from token counts.

Prices are USD per 1 million tokens (input, output), taken from the providers'
public list prices. They change over time, so treat the numbers as estimates:
check the provider's pricing page and pass your own table with
`agent-eval run --prices prices.json` when you need exact figures.

Model names are matched exactly first, then by the longest matching prefix, so
"anthropic:claude-sonnet-4-5-20250929" uses the "anthropic:claude-sonnet-4-5" row.
"""

from __future__ import annotations

import json
from pathlib import Path

# model spec -> (input $/1M tokens, output $/1M tokens)
PRICES: dict[str, tuple[float, float]] = {
    "anthropic:claude-opus-4-1": (15.00, 75.00),
    "anthropic:claude-sonnet-4-5": (3.00, 15.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "ollama:": (0.0, 0.0),  # local models are free to run
    "scripted": (0.0, 0.0),
}


def load_prices(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read a JSON file like {"openai:gpt-4o": [2.5, 10]} and merge it over the defaults."""
    custom = json.loads(Path(path).read_text())
    table = dict(PRICES)
    for model, pair in custom.items():
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"price for {model} must be [input_per_1m, output_per_1m]")
        table[model] = (float(pair[0]), float(pair[1]))
    return table


def price_for(model: str, table: dict[str, tuple[float, float]] | None = None) -> tuple[float, float] | None:
    """Look up (input, output) prices for a model, or None if it is not in the table."""
    table = PRICES if table is None else table
    if model in table:
        return table[model]
    matches = [key for key in table if model.startswith(key)]
    return table[max(matches, key=len)] if matches else None


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    table: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """Estimated USD cost of the given token usage, or None if the model has no known price."""
    price = price_for(model, table)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
