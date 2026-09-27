"""OpenRouter chat-completions forecast baselines."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from tifbench.config import require_api_key
from tifbench.io import ensure_dir, read_json, write_json_atomic
from tifbench.models.forecast_batch_util import (
    chunk_dir_for_custom_ids,
    require_batch_mapping_keys,
)
from tifbench.models.forecast_llm import (
    FORECAST_SCHEMA,
    SCHEMA_NAME,
    digest,
    format_forecast_input,
    parse_p_yes,
    resolved_model_from_response,
)
from tifbench.models.openai_responses_forecast import ForecastWorkItem
from tifbench.models.usage_accounting import extract_usage
from tifbench.openrouter_client.batch import (
    load_openrouter_batch_if_present,
    submit_and_poll_openrouter_batch,
)
from tifbench.schemas import ForecastEvidencePacket, ForecastResult

PROMPT_VERSIONS = {
    "claude_opus_5_5": "forecast_claude_opus_5_5",
    "gemma_3_12b": "forecast_gemma_3_12b",
    "qwen3_5_9b": "forecast_qwen3_5_9b",
}

_RESPONSE_FORMATS = frozenset({"json_schema", "json_object"})


class OpenRouterChatModel:
    """Parameterized OpenRouter chat forecast client with Batch or live transport."""

    def __init__(self, cfg: dict[str, Any], *, name: str):
        if name not in cfg["models"]:
            raise KeyError(f"Missing models.{name} config")
        self.name = name
        self.cfg = cfg["models"][name]
        self.raw_root = Path(cfg["paths"]["raw"]) / "markets"
        self.api_key = require_api_key(self.cfg["api_key_env"])
        self.requested_model = self.cfg["model"]
        self.base_url = str(self.cfg["base_url"]).rstrip("/")
        reasoning = self.cfg.get("reasoning")
        self.reasoning_effort: str | None = None
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            self.reasoning_effort = str(reasoning["effort"])
        fmt = str(self.cfg.get("response_format") or "json_schema").strip().lower()
        if fmt not in _RESPONSE_FORMATS:
            raise ValueError(
                f"models.{name}.response_format must be json_schema|json_object, "
                f"got {fmt!r}"
            )
        self.response_format = fmt
        self.prompt_version = PROMPT_VERSIONS.get(name, f"forecast_{name}")
        configured = str(self.cfg.get("transport", "openrouter_batch")).strip().lower()
        if configured not in ("openrouter_batch", "openrouter_live"):
            raise ValueError(
                f"models.{name}.transport must be openrouter_batch|openrouter_live, "
                f"got {configured!r}"
            )
        self.forecast_transport = configured

    def _cache_dir(self, event_id: str) -> Path:
        return ensure_dir(self.raw_root / event_id / f"{self.name}_cache")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/tifbench",
            "X-Title": "tifbench",
        }

    def build_chat_body(self, item: ForecastWorkItem) -> dict[str, Any]:
        input_text = format_forecast_input(
            item.question, item.resolution_criteria, item.forecast_at, item.packet
        )
        body: dict[str, Any] = {
            "model": self.requested_model,
            "messages": [{"role": "user", "content": input_text}],
        }
        if self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": SCHEMA_NAME,
                    "strict": True,
                    "schema": FORECAST_SCHEMA,
                },
            }
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def build_request_record(self, item: ForecastWorkItem) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "body": self.build_chat_body(item),
        }

    def request_paths(
        self, event_id: str, request_record: dict[str, Any]
    ) -> tuple[str, Path, Path]:
        digest_hex = digest(request_record)
        cache_dir = self._cache_dir(event_id)
        return (
            digest_hex,
            cache_dir / f"{digest_hex}.request.json",
            cache_dir / f"{digest_hex}.response.json",
        )

    def materialize_request(self, item: ForecastWorkItem) -> tuple[dict[str, Any], Path]:
        request_record = self.build_request_record(item)
        _, request_path, response_path = self.request_paths(item.instance_id, request_record)
        write_json_atomic(request_path, request_record)
        return request_record, response_path

    def _result_from_raw(
        self,
        *,
        item: ForecastWorkItem,
        run_id: str,
        raw: dict[str, Any],
        response_path: Path,
    ) -> ForecastResult:
        return ForecastResult(
            event_id=item.instance_id,
            model=self.name,
            requested_model=self.requested_model,
            resolved_model=resolved_model_from_response(raw),
            prompt_version=self.prompt_version,
            probability_yes=parse_p_yes(raw),
            forecast_at=item.forecast_at,
            run_id=run_id,
            raw_path=str(response_path),
            usage=extract_usage(raw),
        )

    def _live_create(self, body: dict[str, Any]) -> dict[str, Any]:
        delays = (2.0, 5.0, 15.0, 30.0, 60.0)
        retry_status = frozenset({429, 502, 503, 504})
        with httpx.Client(timeout=180.0) as client:
            last: httpx.Response | None = None
            last_exc: BaseException | None = None
            for attempt, delay in enumerate((*delays, None)):
                try:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        json=body,
                        headers=self._headers(),
                    )
                    last = response
                    if response.status_code in retry_status:
                        if delay is None:
                            break
                        print(
                            f"OpenRouter {response.status_code} for {self.name}; "
                            f"retry {attempt + 1}/{len(delays)} in {delay:.0f}s"
                        )
                        time.sleep(delay)
                        continue
                    response.raise_for_status()
                    return response.json()
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
                        f"OpenRouter transport error for {self.name}: {exc}; "
                        f"retry {attempt + 1}/{len(delays)} in {delay:.0f}s"
                    )
                    time.sleep(delay)
            if last is not None:
                last.raise_for_status()
                return last.json()
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"OpenRouter live create failed for {self.name}")

    def _result_from_cached_or_live(
        self,
        *,
        item: ForecastWorkItem,
        run_id: str,
        request_record: dict[str, Any],
        response_path: Path,
        live_attempts: int = 3,
    ) -> ForecastResult:
        """Load a parseable cache, or live-create; never leave an unparseable cache."""
        if response_path.exists():
            try:
                return self._result_from_raw(
                    item=item,
                    run_id=run_id,
                    raw=read_json(response_path),
                    response_path=response_path,
                )
            except ValueError as exc:
                print(
                    f"Discarding unparseable {self.name} cache "
                    f"{response_path.name}: {exc}"
                )
                response_path.unlink(missing_ok=True)

        last_exc: BaseException | None = None
        for attempt in range(max(1, live_attempts)):
            raw = self._live_create(request_record["body"])
            try:
                result = self._result_from_raw(
                    item=item, run_id=run_id, raw=raw, response_path=response_path
                )
            except ValueError as exc:
                last_exc = exc
                print(
                    f"Unparseable {self.name} live response "
                    f"(attempt {attempt + 1}/{live_attempts}): {exc}"
                )
                continue
            write_json_atomic(response_path, raw)
            return result
        assert last_exc is not None
        raise last_exc

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
        request_record, response_path = self.materialize_request(item)
        return self._result_from_cached_or_live(
            item=item,
            run_id=run_id,
            request_record=request_record,
            response_path=response_path,
        )

    def forecast_many(
        self,
        items: list[ForecastWorkItem],
        *,
        run_id: str,
        mode: Literal["batch", "live"],
        batch_dir: Path | None = None,
        batch_chunk_size: int = 200,
        batch_poll_interval_sec: int = 30,
        live_max_concurrent: int = 4,
        on_results: Callable[[list[ForecastResult]], None] | None = None,
    ) -> list[ForecastResult]:
        results: list[ForecastResult] = []
        pending: list[tuple[ForecastWorkItem, dict[str, Any], Path]] = []

        def _emit(chunk: list[ForecastResult]) -> None:
            if not chunk:
                return
            results.extend(chunk)
            if on_results is not None:
                on_results(chunk)

        for item in items:
            request_record, response_path = self.materialize_request(item)
            if response_path.exists():
                try:
                    _emit([
                        self._result_from_raw(
                            item=item,
                            run_id=run_id,
                            raw=read_json(response_path),
                            response_path=response_path,
                        )
                    ])
                    continue
                except ValueError as exc:
                    print(
                        f"Discarding unparseable {self.name} cache "
                        f"{response_path.name}: {exc}"
                    )
                    response_path.unlink(missing_ok=True)
            pending.append((item, request_record, response_path))
        if not pending:
            return results

        use_batch = mode == "batch" and self.forecast_transport == "openrouter_batch"
        if not use_batch:
            workers = max(1, int(live_max_concurrent))
            # Process in small waves so completed caches are emitted even if a
            # later wave hangs on the network (avoids orphan response files).
            wave = max(workers * 4, 16)

            def _one(
                triple: tuple[ForecastWorkItem, dict[str, Any], Path],
            ) -> ForecastResult:
                item, request_record, response_path = triple
                return self._result_from_cached_or_live(
                    item=item,
                    run_id=run_id,
                    request_record=request_record,
                    response_path=response_path,
                )

            for offset in range(0, len(pending), wave):
                slice_jobs = pending[offset : offset + wave]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_one, triple) for triple in slice_jobs]
                    for fut in as_completed(futures):
                        _emit([fut.result()])
            return results

        if batch_dir is None:
            raise ValueError("batch_dir is required when mode='batch'")
        ensure_dir(batch_dir)
        chunk = max(1, int(batch_chunk_size))
        for offset in range(0, len(pending), chunk):
            slice_jobs = pending[offset : offset + chunk]
            custom_ids = [item.instance_id for item, _, _ in slice_jobs]
            chunk_dir = ensure_dir(chunk_dir_for_custom_ids(batch_dir, custom_ids))
            prior = load_openrouter_batch_if_present(chunk_dir)
            if prior is None:
                prior = submit_and_poll_openrouter_batch(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.requested_model,
                    requests=[
                        (item.instance_id, request_record["body"])
                        for item, request_record, _ in slice_jobs
                    ],
                    out_dir=chunk_dir,
                    poll_interval_sec=int(batch_poll_interval_sec),
                )
            require_batch_mapping_keys(prior, custom_ids, chunk_dir=chunk_dir)
            for item, _request_record, response_path in slice_jobs:
                raw = prior[item.instance_id]
                if not response_path.exists():
                    write_json_atomic(response_path, raw)
            for item, _request_record, response_path in slice_jobs:
                _emit([
                    self._result_from_raw(
                        item=item,
                        run_id=run_id,
                        raw=read_json(response_path),
                        response_path=response_path,
                    )
                ])
        return results
