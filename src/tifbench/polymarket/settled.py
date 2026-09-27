"""Shared market settled Yes/No check for sample Stage 1."""

from __future__ import annotations

from typing import Any

from tifbench.polymarket.fetch import _parse_jsonish, _parse_outcome


def is_settled_yes_no(market: dict[str, Any]) -> bool:
    outcomes = _parse_jsonish(market.get("outcomes"))
    if not (
        isinstance(outcomes, list)
        and len(outcomes) == 2
        and {str(x).strip().lower() for x in outcomes} == {"yes", "no"}
    ):
        return False
    return _parse_outcome(market) in (0, 1)
