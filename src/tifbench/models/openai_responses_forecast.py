"""OpenAI Responses API forecast baselines (Sol / Astra) with live + Batch support."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

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
from tifbench.models.usage_accounting import extract_usage
from tifbench.openai_client import (
    build_responses_body,
    load_batch_output_if_present,
    make_client,
    responses_create,
    submit_and_download_batch,
    write_batch_jsonl,
)
from tifbench.schemas import ForecastEvidencePacket, ForecastResult


@dataclass(frozen=True)
class ForecastWorkItem:
    instance_id: str
    question: str
    resolution_criteria: str
    forecast_at: datetime
    packet: ForecastEvidencePacket


class OpenAIResponsesForecastModel:
    """Shared live + OpenAI Batch path for GPT Responses forecast baselines."""

    name: str
    prompt_version: str
    forecast_transport: str = "openai_batch"

    def __init__(self, cfg: dict[str, Any], *, config_key: str, prompt_version: str):
        self.cfg = cfg["models"][config_key]
        self.raw_root = Path(cfg["paths"]["raw"]) / "markets"
        self.api_key = require_api_key(self.cfg["api_key_env"])
        self.requested_model = self.cfg["model"]
        self.reasoning_effort = self.cfg.get("reasoning", {}).get("effort", "medium")
        self.prompt_version = prompt_version
        self._client = None

    def _client_or_create(self):
        if self._client is None:
            self._client = make_client(self.api_key)
        return self._client

    def _cache_dir(self, event_id: str) -> Path:
        return ensure_dir(self.raw_root / event_id / f"{self.name}_cache")

    def build_request_record(self, item: ForecastWorkItem) -> dict[str, Any]:
        input_text = format_forecast_input(
            item.question, item.resolution_criteria, item.forecast_at, item.packet
        )
        body = build_responses_body(
            model=self.requested_model,
            input_text=input_text,
            reasoning_effort=self.reasoning_effort,
            json_schema=FORECAST_SCHEMA,
            schema_name=SCHEMA_NAME,
        )
        return {"prompt_version": self.prompt_version, "body": body}

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
        if response_path.exists():
            raw = read_json(response_path)
        else:
            response = responses_create(self._client_or_create(), request_record["body"])
            raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            write_json_atomic(response_path, raw)
        return self._result_from_raw(
            item=item, run_id=run_id, raw=raw, response_path=response_path
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
                # Emit one-by-one so a later bad cache does not block earlier goods.
                _emit([
                    self._result_from_raw(
                        item=item,
                        run_id=run_id,
                        raw=read_json(response_path),
                        response_path=response_path,
                    )
                ])
            else:
                pending.append((item, request_record, response_path))
        if not pending:
            return results

        if mode == "live":
            workers = max(1, int(live_max_concurrent))

            def _one(
                triple: tuple[ForecastWorkItem, dict[str, Any], Path],
            ) -> ForecastResult:
                item, request_record, response_path = triple
                if response_path.exists():
                    raw = read_json(response_path)
                else:
                    response = responses_create(
                        self._client_or_create(), request_record["body"]
                    )
                    raw = (
                        response.model_dump()
                        if hasattr(response, "model_dump")
                        else dict(response)
                    )
                    write_json_atomic(response_path, raw)
                return self._result_from_raw(
                    item=item, run_id=run_id, raw=raw, response_path=response_path
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_one, triple) for triple in pending]
                for fut in as_completed(futures):
                    _emit([fut.result()])
            return results

        if batch_dir is None:
            raise ValueError("batch_dir is required when mode='batch'")
        ensure_dir(batch_dir)
        client = self._client_or_create()
        chunk = max(1, int(batch_chunk_size))
        for offset in range(0, len(pending), chunk):
            slice_jobs = pending[offset : offset + chunk]
            custom_ids = [item.instance_id for item, _, _ in slice_jobs]
            chunk_dir = ensure_dir(chunk_dir_for_custom_ids(batch_dir, custom_ids))
            prior = load_batch_output_if_present(chunk_dir)
            if prior is None:
                jsonl = chunk_dir / "input.jsonl"
                write_batch_jsonl(
                    jsonl,
                    [(item.instance_id, request_record["body"])
                     for item, request_record, _ in slice_jobs],
                )
                submit_and_download_batch(
                    client,
                    jsonl,
                    chunk_dir,
                    poll_interval_sec=int(batch_poll_interval_sec),
                )
                prior = load_batch_output_if_present(chunk_dir)
            if prior is None:
                raise RuntimeError(f"Batch output missing under {chunk_dir}")
            require_batch_mapping_keys(prior, custom_ids, chunk_dir=chunk_dir)
            # Freeze all raws first, then emit per item so good rows persist
            # before a later parse failure.
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
