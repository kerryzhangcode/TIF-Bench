"""OpenRouter Decisions forecast clients (Jev / Kev)."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from tifbench.config import require_api_key
from tifbench.evidence.forecast import format_forecast_state
from tifbench.io import read_json, write_json
from tifbench.models.openai_responses_forecast import ForecastWorkItem
from tifbench.models.usage_accounting import extract_usage
from tifbench.schemas import ForecastEvidencePacket, ForecastResult

FORECAST_QUESTION_KEY = "forecast"

# Single allowlist shared with registry.get_model.
DECISIONS_MODEL_NAMES = frozenset({"jev", "kev"})


class DecisionsForecastModel:
    """Parameterized OpenRouter Decisions (`/alpha/decisions`) forecast client."""

    def __init__(self, cfg: dict[str, Any], *, name: str):
        if name not in DECISIONS_MODEL_NAMES:
            raise KeyError(f"Unknown Decisions model: {name}")
        if name not in cfg["models"]:
            raise KeyError(f"Missing models.{name} config")
        self.name = name
        self.cfg = cfg["models"][name]
        self.raw_root = Path(cfg["paths"]["raw"]) / "markets"
        self.api_key = require_api_key(self.cfg["api_key_env"])
        self.requested_model = self.cfg["model"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.prompt_version = f"forecast_{name}"

    def _cache_path(self, event_id: str, payload: dict[str, Any]) -> Path:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return self.raw_root / event_id / f"{self.name}_cache_{digest}.json"

    def _decisions_url(self) -> str:
        return f"{self.base_url}/alpha/decisions"

    def _payload(
        self,
        *,
        question: str,
        resolution_criteria: str,
        forecast_at: datetime,
        packet: ForecastEvidencePacket,
    ) -> dict[str, Any]:
        state = format_forecast_state(question, resolution_criteria, forecast_at, packet)
        return {
            "model": self.requested_model,
            "state": state,
            "questions": {
                FORECAST_QUESTION_KEY: {
                    "type": "noul",
                    "instructions": (
                        "Based only on the evidence in state, what is the probability "
                        "that the market question resolves YES?"
                    ),
                    "criteria": {
                        "true": "The event resolves YES under the resolution criteria.",
                        "false": "The event resolves NO under the resolution criteria.",
                    },
                }
            },
        }

    def _live_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/tifbench",
            "X-Title": "tifbench",
        }
        # Retry transient OpenRouter / local-proxy failures (429/5xx, ProxyError).
        delays = (2.0, 5.0, 15.0, 30.0, 60.0)
        retry_status = frozenset({429, 502, 503, 504})
        with httpx.Client(timeout=120.0) as client:
            last_exc: BaseException | None = None
            for attempt, delay in enumerate((*delays, None)):
                try:
                    r = client.post(
                        self._decisions_url(), json=payload, headers=headers
                    )
                    if r.status_code in retry_status:
                        if delay is None:
                            r.raise_for_status()
                        print(
                            f"Decisions {r.status_code} for {self.name}; "
                            f"retry {attempt + 1}/{len(delays)} in {delay:.0f}s"
                        )
                        time.sleep(delay)
                        continue
                    r.raise_for_status()
                    return r.json()
                except (
                    httpx.ProxyError,
                    httpx.ConnectError,
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.RemoteProtocolError,
                ) as exc:
                    last_exc = exc
                    if delay is None:
                        raise
                    print(
                        f"Decisions transport error for {self.name}: {exc}; "
                        f"retry {attempt + 1}/{len(delays)} in {delay:.0f}s"
                    )
                    time.sleep(delay)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"Decisions live create failed for {self.name}")

    def forecast(
        self,
        *,
        event_id: str,
        question: str,
        resolution_criteria: str,
        forecast_at: datetime,
        packet: ForecastEvidencePacket,
        run_id: str,
    ) -> ForecastResult:
        item = ForecastWorkItem(
            instance_id=event_id,
            question=question,
            resolution_criteria=resolution_criteria,
            forecast_at=forecast_at,
            packet=packet,
        )
        return self.forecast_many(
            [item],
            run_id=run_id,
            mode="live",
            live_max_concurrent=1,
        )[0]

    def forecast_many(
        self,
        items: list[ForecastWorkItem],
        *,
        run_id: str,
        mode: Literal["batch", "live"] = "live",
        batch_dir: Path | None = None,
        batch_chunk_size: int = 200,
        batch_poll_interval_sec: int = 30,
        live_max_concurrent: int = 4,
        on_results: Callable[[list[ForecastResult]], None] | None = None,
    ) -> list[ForecastResult]:
        """Live concurrent Decisions calls.

        ``mode=batch`` is accepted for formal runner compatibility but still
        uses live concurrency (Decisions has no OpenRouter Batch surface).
        """
        _ = (mode, batch_dir, batch_chunk_size, batch_poll_interval_sec)
        results: list[ForecastResult] = []

        def _emit(chunk: list[ForecastResult]) -> None:
            if not chunk:
                return
            results.extend(chunk)
            if on_results is not None:
                on_results(chunk)

        def _one(item: ForecastWorkItem) -> ForecastResult:
            payload = self._payload(
                question=item.question,
                resolution_criteria=item.resolution_criteria,
                forecast_at=item.forecast_at,
                packet=item.packet,
            )
            cache = self._cache_path(item.instance_id, payload)
            if cache.exists():
                raw = read_json(cache)
            else:
                raw = self._live_create(payload)
                write_json(cache, raw)
            probability, resolved = parse_decisions_response(raw)
            return ForecastResult(
                event_id=item.instance_id,
                model=self.name,
                requested_model=self.requested_model,
                resolved_model=resolved,
                prompt_version=self.prompt_version,
                probability_yes=probability,
                forecast_at=item.forecast_at,
                run_id=run_id,
                raw_path=str(cache),
                usage=extract_usage(raw),
            )

        workers = max(1, int(live_max_concurrent))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, item) for item in items]
            for fut in as_completed(futures):
                _emit([fut.result()])
        return results


def parse_decisions_response(raw: dict[str, Any]) -> tuple[float, str | None]:
    """Parse OpenRouter Decisions response: answers.forecast.noul + model snapshot."""
    if "model" not in raw:
        raise ValueError("Decisions response missing 'model' (resolved snapshot)")
    resolved = raw["model"]
    answers = raw.get("answers")
    if not isinstance(answers, dict) or FORECAST_QUESTION_KEY not in answers:
        raise ValueError("Decisions response missing answers.forecast")
    ans = answers[FORECAST_QUESTION_KEY]
    if not isinstance(ans, dict) or "noul" not in ans:
        raise ValueError("Decisions response missing answers.forecast.noul")
    probability = float(ans["noul"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Decisions forecast probability is outside [0, 1]")
    return probability, str(resolved)


# Backward-compatible name used by older tests/imports.
