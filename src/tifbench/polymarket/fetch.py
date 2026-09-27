"""Polymarket Gamma/CLOB fetch helpers for sample + formal paths."""

from __future__ import annotations

from typing import Any

import httpx


def _gamma_get(
    client: httpx.Client,
    base: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    r = client.get(f"{base.rstrip('/')}{path}", params=params or {}, timeout=60.0)
    r.raise_for_status()
    return r.json()


def fetch_market_by_id(client: httpx.Client, gamma_base: str, market_id: str) -> dict[str, Any]:
    """GET /markets/{id} — reliable for many closed markets where ?slug= returns empty."""
    data = _gamma_get(client, gamma_base, f"/markets/{market_id}")
    if not isinstance(data, dict) or not data.get("id"):
        raise LookupError(f"No market for market_id={market_id}")
    return data


def prices_history_url(clob_base: str) -> str:
    return f"{clob_base.rstrip('/')}/prices-history"


def fetch_price_history(
    client: httpx.Client,
    clob_base: str,
    token_id: str,
    *,
    interval: str | None = "max",
    fidelity: int = 60,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict[str, Any]]:
    """CLOB prices-history: returns list of {t, p} in unix seconds / probability.

    Prefer startTs/endTs for short formal cutoff windows. interval=max alone often
    returns only recent history and is insufficient for multi-week horizons.
    """
    params: dict[str, Any] = {"market": token_id, "fidelity": fidelity}
    if start_ts is not None:
        params["startTs"] = int(start_ts)
    if end_ts is not None:
        params["endTs"] = int(end_ts)
    if interval is not None and start_ts is None:
        params["interval"] = interval
    r = client.get(prices_history_url(clob_base), params=params, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "history" in data:
        return list(data["history"] or [])
    if isinstance(data, list):
        return data
    return []


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _yes_token_id(market: dict[str, Any]) -> str | None:
    tokens = _parse_jsonish(market.get("clobTokenIds") or market.get("clob_token_ids"))
    if isinstance(tokens, list) and tokens:
        return str(tokens[0])
    return market.get("token_id")


def _normalize_yes_no_label(label: Any) -> int | None:
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in ("yes", "y", "true"):
        return 1
    if s in ("no", "n", "false"):
        return 0
    return None


def _parse_outcome(market: dict[str, Any]) -> int | None:
    """Infer binary YES (1) / NO (0) only when labels map explicitly to Yes/No."""
    for key in ("outcome", "resolvedOutcome", "winningOutcome"):
        mapped = _normalize_yes_no_label(market.get(key))
        if mapped is not None:
            return mapped
        o = market.get(key)
        if o in (1, "1") or o in (0, "0"):
            outcomes = _parse_jsonish(market.get("outcomes"))
            if (
                isinstance(outcomes, list)
                and len(outcomes) == 2
                and {_normalize_yes_no_label(x) for x in outcomes} == {0, 1}
            ):
                return int(o)
            return None

    outcomes = _parse_jsonish(market.get("outcomes"))
    prices = _parse_jsonish(market.get("outcomePrices"))
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return None
    label_map = [_normalize_yes_no_label(x) for x in outcomes]
    if set(label_map) != {0, 1}:
        return None

    if isinstance(prices, list) and len(prices) == 2:
        try:
            nums = [float(p) for p in prices]
        except (TypeError, ValueError):
            return None
        if max(nums) >= 0.99 and min(nums) <= 0.01:
            winner = max(range(len(nums)), key=lambda i: nums[i])
            return label_map[winner]

    return None
