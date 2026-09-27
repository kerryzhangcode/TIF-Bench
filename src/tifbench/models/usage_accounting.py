"""Extract forecast API usage and resolve USD cost (fail-soft)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _usage_block(raw: dict[str, Any]) -> dict[str, Any] | None:
    usage = raw.get("usage")
    if isinstance(usage, dict):
        return usage
    response = raw.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    return None


def extract_usage(raw: dict[str, Any] | str) -> dict[str, Any] | None:
    """Normalize token/cost fields from OpenAI Responses, OpenRouter chat, or Decisions.

    Returns None when nothing usable is present. Never raises on shape mismatch.
    """
    if isinstance(raw, str) or not isinstance(raw, dict):
        return None
    usage = _usage_block(raw)
    if not usage:
        return None

    out: dict[str, Any] = {}

    input_tokens = _as_int(
        usage.get("input_tokens", usage.get("prompt_tokens"))
    )
    output_tokens = _as_int(
        usage.get("output_tokens", usage.get("completion_tokens"))
    )
    if input_tokens is not None:
        out["input_tokens"] = input_tokens
    if output_tokens is not None:
        out["output_tokens"] = output_tokens

    details_out = usage.get("output_tokens_details") or usage.get(
        "completion_tokens_details"
    )
    if isinstance(details_out, dict):
        reasoning = _as_int(details_out.get("reasoning_tokens"))
        if reasoning is not None:
            out["reasoning_tokens"] = reasoning

    details_in = usage.get("input_tokens_details") or usage.get(
        "prompt_tokens_details"
    )
    if isinstance(details_in, dict):
        cached = _as_int(details_in.get("cached_tokens"))
        if cached is not None:
            out["cached_tokens"] = cached

    provider_cost = _as_float(usage.get("cost"))
    if provider_cost is None:
        provider_cost = _as_float(usage.get("total_cost"))
    if provider_cost is not None:
        out["provider_cost_usd"] = provider_cost

    return out or None


def pricing_digest(pricing_cfg: dict[str, Any] | None) -> str:
    payload = pricing_cfg if isinstance(pricing_cfg, dict) else {}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def resolve_cost_usd(
    *,
    usage: dict[str, Any] | None,
    model_key: str,
    forecast_mode: Literal["batch", "live"],
    pricing_cfg: dict[str, Any] | None,
) -> tuple[float | None, str]:
    """Return (cost_usd, cost_source). Fail-soft: never raises for missing data."""
    if not usage:
        return None, "missing_usage"

    provider_cost = _as_float(usage.get("provider_cost_usd"))
    if provider_cost is not None:
        return provider_cost, "openrouter_usage"

    rates = None
    if isinstance(pricing_cfg, dict):
        candidate = pricing_cfg.get(model_key)
        if isinstance(candidate, dict):
            rates = candidate
    if not rates:
        return None, "missing_pricing"

    input_rate = _as_float(rates.get("input_per_mtok"))
    output_rate = _as_float(rates.get("output_per_mtok"))
    if input_rate is None or output_rate is None:
        return None, "missing_pricing"

    input_tokens = _as_int(usage.get("input_tokens")) or 0
    output_tokens = _as_int(usage.get("output_tokens")) or 0
    cached_tokens = _as_int(usage.get("cached_tokens")) or 0
    cached_rate = _as_float(rates.get("cached_input_per_mtok"))

    if cached_rate is not None and cached_tokens > 0:
        billable_input = max(0, input_tokens - cached_tokens)
        cost = (
            billable_input * input_rate
            + cached_tokens * cached_rate
            + output_tokens * output_rate
        ) / 1_000_000.0
    else:
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0

    if forecast_mode == "batch":
        multiplier = _as_float(rates.get("batch_multiplier"))
        if multiplier is None:
            multiplier = 0.5
        cost *= multiplier

    return cost, "openai_token_pricing"


def cost_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate usage/cost by model for cost_summary.json."""
    by_model: dict[str, dict[str, Any]] = {}
    for row in records:
        model = str(row.get("model") or "")
        bucket = by_model.setdefault(
            model,
            {
                "model": model,
                "n_calls": 0,
                "n_with_cost": 0,
                "n_missing_cost": 0,
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_reasoning_tokens": 0,
                "cost_sources": {},
            },
        )
        bucket["n_calls"] += 1
        cost = _as_float(row.get("cost_usd"))
        if cost is None:
            bucket["n_missing_cost"] += 1
        else:
            bucket["n_with_cost"] += 1
            bucket["total_cost_usd"] += cost
        source = row.get("cost_source")
        if source:
            sources = bucket["cost_sources"]
            sources[str(source)] = int(sources.get(str(source), 0)) + 1
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        for key, total_key in (
            ("input_tokens", "total_input_tokens"),
            ("output_tokens", "total_output_tokens"),
            ("reasoning_tokens", "total_reasoning_tokens"),
        ):
            n = _as_int(usage.get(key))
            if n is not None:
                bucket[total_key] += n

    rows = []
    for model in sorted(by_model):
        bucket = by_model[model]
        if bucket["n_with_cost"] == 0:
            bucket["total_cost_usd"] = None
        rows.append(bucket)
    return rows
