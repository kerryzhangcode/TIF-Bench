"""Run Responses API jobs via OpenAI Batch or bounded live concurrency.

Freeze layout matches ``evidence.formal._call_cached``:
``{root}/{name}/{digest}.request.json`` + ``{digest}.response.json``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tifbench.io import ensure_dir, read_json, write_json
from tifbench.openai_client.responses import (
    extract_output_text,
    load_batch_output_if_present,
    responses_create,
    submit_and_download_batch,
    write_batch_jsonl,
)


def _digest(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def _freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Frozen artifact differs: {path}")
        return
    write_json(path, value)


@dataclass(frozen=True)
class ResponseJob:
    """One Responses call with a stable freeze directory."""

    custom_id: str
    body: dict[str, Any]
    freeze_root: Path  # directory that will contain ``{name}/``
    name: str  # e.g. relevance / summary / queries


def job_paths(job: ResponseJob) -> tuple[str, Path, Path]:
    digest = _digest(job.body)
    folder = ensure_dir(job.freeze_root / job.name)
    request_path = folder / f"{digest}.request.json"
    response_path = folder / f"{digest}.response.json"
    return digest, request_path, response_path


def materialize_job_request(job: ResponseJob) -> tuple[str, Path]:
    digest, request_path, response_path = job_paths(job)
    _freeze_json(request_path, job.body)
    return digest, response_path


def read_job_output(job: ResponseJob) -> tuple[dict[str, Any], str] | None:
    digest, _, response_path = job_paths(job)
    if not response_path.exists():
        return None
    raw = read_json(response_path)
    return json.loads(extract_output_text(raw)), digest


def execute_response_jobs(
    client: Any,
    jobs: list[ResponseJob],
    *,
    mode: Literal["batch", "live"],
    batch_dir: Path | None = None,
    batch_chunk_size: int = 200,
    batch_poll_interval_sec: int = 30,
    live_max_concurrent: int = 4,
) -> dict[str, tuple[dict[str, Any], str]]:
    """Execute jobs; return ``custom_id -> (parsed_json, request_digest)``.

    Already-frozen responses are reused and not re-submitted.
    """
    pending: list[ResponseJob] = []
    results: dict[str, tuple[dict[str, Any], str]] = {}
    for job in jobs:
        materialize_job_request(job)
        existing = read_job_output(job)
        if existing is not None:
            results[job.custom_id] = existing
        else:
            pending.append(job)

    if not pending:
        return results

    if mode == "live":
        workers = max(1, int(live_max_concurrent))

        def _one(job: ResponseJob) -> tuple[str, dict[str, Any], str]:
            digest, _, response_path = job_paths(job)
            if response_path.exists():
                raw = read_json(response_path)
            else:
                response = responses_create(client, job.body)
                raw = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else dict(response)
                )
                _freeze_json(response_path, raw)
            parsed = json.loads(extract_output_text(raw))
            return job.custom_id, parsed, digest

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, job) for job in pending]
            for fut in as_completed(futures):
                custom_id, parsed, digest = fut.result()
                results[custom_id] = (parsed, digest)
        return results

    if batch_dir is None:
        raise ValueError("batch_dir is required when mode='batch'")
    ensure_dir(batch_dir)
    chunk = max(1, int(batch_chunk_size))
    for offset in range(0, len(pending), chunk):
        slice_jobs = pending[offset : offset + chunk]
        chunk_dir = ensure_dir(batch_dir / f"chunk_{offset // chunk:04d}")
        prior = load_batch_output_if_present(chunk_dir)
        if prior is None:
            jsonl = chunk_dir / "input.jsonl"
            write_batch_jsonl(
                jsonl,
                [(job.custom_id, job.body) for job in slice_jobs],
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
        for job in slice_jobs:
            if job.custom_id not in prior:
                raise RuntimeError(
                    f"Batch output missing custom_id={job.custom_id} in {chunk_dir}"
                )
            raw = prior[job.custom_id]
            digest, _, response_path = job_paths(job)
            if not response_path.exists():
                _freeze_json(response_path, raw)
            parsed = json.loads(extract_output_text(raw))
            results[job.custom_id] = (parsed, digest)
    return results
