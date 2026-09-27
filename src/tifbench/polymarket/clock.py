"""Formal benchmark clock: Gamma market.closedTime only (UTC)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Paper clock: prediction before market close — not first public confirmation,
# not on-chain settlement, and never endDate / open / updatedAt substitutes.
CLOCK_DEFINITION = "gamma_market.closedTime_utc"


def parse_gamma_timestamp(raw: Any) -> datetime | None:
    """Parse a Gamma timestamp string/number to UTC, or None if missing/invalid."""
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, (int, float)):
            ts = datetime.fromtimestamp(raw, tz=timezone.utc)
        else:
            text = str(raw).strip().replace("Z", "+00:00")
            # Gamma often emits "YYYY-MM-DD HH:MM:SS+00" without the T separator.
            if "T" not in text and " " in text:
                text = text.replace(" ", "T", 1)
            ts = datetime.fromisoformat(text)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def parse_market_closed_time(market: dict[str, Any]) -> datetime | None:
    """Return UTC closedTime, or None if missing/invalid. No fallback fields."""
    return parse_gamma_timestamp(market.get("closedTime"))


def parse_market_active_start(market: dict[str, Any]) -> datetime | None:
    """Market active-interval start: Gamma startDate, else createdAt. Not a news search bound."""
    for key in ("startDate", "createdAt", "created_at"):
        ts = parse_gamma_timestamp(market.get(key))
        if ts is not None:
            return ts
    return None
