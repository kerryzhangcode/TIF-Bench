"""Forecast evidence helpers — must never contain future/outcome leakage."""

from __future__ import annotations

from datetime import datetime, timezone

from tifbench.schemas import ForecastEvidencePacket, ReasonCode


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def check_forecast_leakage(
    packet: ForecastEvidencePacket,
    forecast_at: datetime,
    *,
    inclusive_cutoff: bool = False,
) -> list[ReasonCode]:
    """Return reason codes if any item leaks past forecast_at or is terminal-tagged."""
    forecast_at = _ensure_utc(forecast_at)
    codes: list[ReasonCode] = []
    for item in packet.items:
        published = _ensure_utc(item.published_at)
        after_cutoff = (
            published > forecast_at if inclusive_cutoff else published >= forecast_at
        )
        if after_cutoff:
            codes.append(ReasonCode.FORECAST_EVIDENCE_LEAKAGE)
            break
        if item.is_terminal_or_resolution:
            codes.append(ReasonCode.FORECAST_EVIDENCE_LEAKAGE)
            break
    return codes


def format_forecast_state(
    question: str,
    resolution_criteria: str,
    forecast_at: datetime,
    packet: ForecastEvidencePacket,
) -> str:
    """Serialize forecast inputs into a single state string for model APIs."""
    lines = [
        f"Question: {question}",
        f"Resolution criteria: {resolution_criteria}",
        f"Forecast time (UTC): {forecast_at.isoformat()}",
        "",
        "Public evidence available at forecast time:",
    ]
    if not packet.items:
        lines.append("(no evidence items)")
    if packet.brief:
        lines.append(packet.brief)
        return "\n".join(lines)
    for i, item in enumerate(packet.items, 1):
        lines.append(
            f"[{i}] id={item.source_id} published_at={item.published_at.isoformat()} "
            f"title={item.title}\n{item.text}"
        )
    return "\n".join(lines)
