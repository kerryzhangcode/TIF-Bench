"""Shared LLM forecast prompt, schema, and fail-closed p_yes parsing."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from tifbench.evidence.forecast import format_forecast_state
from tifbench.openai_client import extract_output_text
from tifbench.schemas import ForecastEvidencePacket

FORECAST_INSTRUCTIONS = """Based only on the evidence below, estimate the probability that the market question resolves YES under the stated resolution criteria.
Return JSON with a single field p_yes in [0, 1].
Do not use outside knowledge beyond the provided evidence.
Do not search, browse, or retrieve additional sources.
Do not use market prices, outcomes, or any information after the forecast time.

{state}
"""

FORECAST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "p_yes": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": ["p_yes"],
}

SCHEMA_NAME = "forecast_p_yes"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def format_forecast_input(
    question: str,
    resolution_criteria: str,
    forecast_at: datetime,
    packet: ForecastEvidencePacket,
) -> str:
    state = format_forecast_state(question, resolution_criteria, forecast_at, packet)
    return FORECAST_INSTRUCTIONS.format(state=state)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _content_from_raw(raw: dict[str, Any] | str) -> str:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("forecast response must be a string or object")
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message["content"])
    try:
        return extract_output_text(raw)
    except ValueError:
        pass
    if "p_yes" in raw:
        return json.dumps({"p_yes": raw["p_yes"]})
    raise ValueError("Could not extract forecast text from response")


def parse_p_yes(raw: dict[str, Any] | str) -> float:
    """Fail-closed: require a JSON object with only p_yes in [0, 1]."""
    text = _strip_fences(_content_from_raw(raw))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("forecast response is not valid JSON") from exc
    if not isinstance(payload, dict) or "p_yes" not in payload:
        raise ValueError("forecast response missing p_yes")
    if len(payload) != 1:
        raise ValueError("forecast response must contain only p_yes")
    try:
        probability = float(payload["p_yes"])
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast p_yes is not a number") from exc
    if not 0.0 <= probability <= 1.0:
        raise ValueError("forecast p_yes is outside [0, 1]")
    return probability


def resolved_model_from_response(raw: dict[str, Any] | str) -> str | None:
    if not isinstance(raw, dict):
        return None
    resolved = raw.get("model")
    if resolved is None and isinstance(raw.get("response"), dict):
        resolved = raw["response"].get("model")
    return str(resolved) if resolved else None
