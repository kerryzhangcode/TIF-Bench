"""Shared Stage-1 gate: post-deadline observation markets fail closed."""

from __future__ import annotations

import re
from typing import Any

POST_DEADLINE_OBS_RE = re.compile(
    r"\b(continuous|uninterrupted|consecutive)\s+\d+[-\s]?day\b|"
    r"\b\d+[-\s]?day\s+(period|stretch|window)\b.{0,80}\b(after|following)\b|"
    r"\bremain open\b.{0,80}\b(after|past)\b.{0,40}\b(deadline|end date|specified date)\b",
    re.I | re.DOTALL,
)

POST_DEADLINE_OBSERVATION = "POST_DEADLINE_OBSERVATION"


def post_deadline_observation_reason(
    market_or_description: dict[str, Any] | str | None,
) -> str | None:
    """Return POST_DEADLINE_OBSERVATION when rules require observing after the deadline."""
    if isinstance(market_or_description, dict):
        text = str(market_or_description.get("description") or "")
    else:
        text = str(market_or_description or "")
    if POST_DEADLINE_OBS_RE.search(text):
        return POST_DEADLINE_OBSERVATION
    return None
