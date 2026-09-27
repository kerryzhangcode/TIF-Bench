"""Price history: latest observation at or before forecast_at, with max lag."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from tifbench.io import read_json, write_json_atomic
from tifbench.polymarket.fetch import (
    _yes_token_id,
    fetch_price_history,
    prices_history_url,
)
from tifbench.schemas import ReasonCode


@dataclass
class PriceAtForecast:
    probability: float
    asof: datetime
    lag_hours: float


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, (int, float)):
        # Polymarket CLOB often uses unix seconds
        ts = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)}")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _extract_points(price_history: list[dict[str, Any]]) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    for row in price_history:
        # Common shapes: {t, p}, {timestamp, price}, {t, price}
        t_raw = row.get("t", row.get("timestamp", row.get("time")))
        p_raw = row.get("p", row.get("price", row.get("prob")))
        if t_raw is None or p_raw is None:
            continue
        points.append((_parse_ts(t_raw), float(p_raw)))
    points.sort(key=lambda x: x[0])
    return points


def latest_price_at_or_before(
    price_history: list[dict[str, Any]],
    forecast_at: datetime,
    max_lag_hours: float,
) -> tuple[PriceAtForecast | None, ReasonCode | None]:
    """
    p_M(t) = latest price with t_obs <= forecast_at.
    Never uses a future observation. Enforces max_lag_hours.
    """
    forecast_at = _ensure_utc(forecast_at)

    points = _extract_points(price_history)
    eligible = [(t, p) for t, p in points if t <= forecast_at]
    if not eligible:
        return None, ReasonCode.NO_PRICE_AT_OR_BEFORE_FORECAST

    t_obs, price = eligible[-1]
    lag_hours = (forecast_at - t_obs).total_seconds() / 3600.0
    if lag_hours > max_lag_hours:
        return None, ReasonCode.PRICE_LAG_EXCEEDED

    return PriceAtForecast(probability=price, asof=t_obs, lag_hours=lag_hours), None


def _freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return
    write_json_atomic(path, value)


def formal_price_window_bounds(
    cutoff: datetime,
    *,
    max_lag_hours: float,
    lookahead_hours: float,
) -> tuple[datetime, datetime]:
    """Fetch window around cutoff. Selection still uses only t_obs <= cutoff."""
    cutoff = _ensure_utc(cutoff)
    start = cutoff - timedelta(hours=float(max_lag_hours))
    end = cutoff + timedelta(hours=float(lookahead_hours))
    return start, end


def freeze_formal_price_at_cutoff(
    client: httpx.Client,
    cfg: dict[str, Any],
    *,
    market: dict[str, Any],
    cutoff: datetime,
    folder: Path,
    freeze_name: str,
) -> tuple[float | None, datetime | None, float | None, ReasonCode | None, Path]:
    """Fetch a short CLOB window, freeze request/response/selection, pick <=cutoff.

    Never uses a future point or the market's final/settlement odds. Formal selection
    must come from this frozen artifact, not from interval=max price_history.json.
    """
    cutoff = _ensure_utc(cutoff)
    max_lag = float(cfg["polymarket"]["max_price_lag_hours"])
    lookahead = float(cfg["polymarket"].get("price_fetch_lookahead_hours", 1))
    fidelity = int(cfg["polymarket"].get("price_fidelity", 60))
    clob = str(cfg["polymarket"]["clob_base_url"])
    token_id = _yes_token_id(market)
    freeze_path = folder / freeze_name

    frozen: dict[str, Any] | None = None
    if freeze_path.exists():
        try:
            frozen = read_json(freeze_path)
        except json.JSONDecodeError:
            freeze_path.unlink()
            frozen = None

    if frozen is not None:
        history = frozen.get("response_history") or []
        hit, err = latest_price_at_or_before(history, cutoff, max_lag)
        selected = frozen.get("selected")
        if err:
            if selected is not None:
                raise RuntimeError(f"Frozen price selection inconsistent: {freeze_path}")
            return None, None, None, err, freeze_path
        assert hit is not None and selected is not None
        if (
            abs(float(selected["probability"]) - hit.probability) > 1e-12
            or selected.get("asof") != hit.asof.isoformat()
        ):
            raise RuntimeError(f"Frozen price selection inconsistent: {freeze_path}")
        return hit.probability, hit.asof, hit.lag_hours, None, freeze_path

    if not token_id:
        artifact = {
            "request": None,
            "response_history": [],
            "cutoff": cutoff.isoformat(),
            "max_lag_hours": max_lag,
            "selected": None,
            "selection_error": ReasonCode.NO_PRICE_AT_OR_BEFORE_FORECAST.value,
        }
        _freeze_json(freeze_path, artifact)
        return None, None, None, ReasonCode.NO_PRICE_AT_OR_BEFORE_FORECAST, freeze_path

    window_start, window_end = formal_price_window_bounds(
        cutoff, max_lag_hours=max_lag, lookahead_hours=lookahead
    )
    params = {
        "market": token_id,
        "startTs": int(window_start.timestamp()),
        "endTs": int(window_end.timestamp()),
        "fidelity": fidelity,
    }
    url = prices_history_url(clob)
    try:
        history = fetch_price_history(
            client,
            clob,
            token_id,
            interval=None,
            fidelity=fidelity,
            start_ts=params["startTs"],
            end_ts=params["endTs"],
        )
        fetch_error = None
    except httpx.HTTPError as exc:
        history = []
        fetch_error = f"{type(exc).__name__}: {exc}"

    hit, err = latest_price_at_or_before(history, cutoff, max_lag)
    selected = None
    if hit is not None:
        selected = {
            "probability": hit.probability,
            "asof": hit.asof.isoformat(),
            "lag_hours": hit.lag_hours,
            "t": hit.asof.timestamp(),
            "p": hit.probability,
        }
    artifact = {
        "request": {"url": url, "params": params},
        "response_history": history,
        "cutoff": cutoff.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "max_lag_hours": max_lag,
        "selected": selected,
        "selection_error": err.value if err else None,
        "fetch_error": fetch_error,
    }
    _freeze_json(freeze_path, artifact)
    if err:
        return None, None, None, err, freeze_path
    assert hit is not None
    return hit.probability, hit.asof, hit.lag_hours, None, freeze_path
