"""OpenAI Responses API helpers: sync calls and Batch JSONL submit/poll."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from tifbench.io import ensure_dir, read_json, write_json, write_jsonl


def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def build_responses_body(
    *,
    model: str,
    input_text: str,
    reasoning_effort: str,
    json_schema: dict[str, Any],
    schema_name: str = "curation_result",
) -> dict[str, Any]:
    """Body for POST /v1/responses (also used inside Batch JSONL)."""
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": input_text}],
            }
        ],
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            }
        },
    }


def extract_output_text(response: Any) -> str:
    """Pull concatenated output_text from a Responses API result."""
    if isinstance(response, dict):
        if "output_text" in response and response["output_text"]:
            return response["output_text"]
        chunks: list[str] = []
        for item in response.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") in ("output_text", "text") and "text" in content:
                    chunks.append(content["text"])
        if chunks:
            return "".join(chunks)
        body = response.get("response") or response.get("body")
        if isinstance(body, dict):
            return extract_output_text(body)
        raise ValueError("Could not extract output_text from response dict")

    text = getattr(response, "output_text", None)
    if text:
        return text
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            t = getattr(content, "text", None)
            if t:
                chunks.append(t)
    if chunks:
        return "".join(chunks)
    raise ValueError("Could not extract output_text from Responses result")


def responses_create(client: OpenAI, body: dict[str, Any]) -> Any:
    return client.responses.create(**body)


def write_batch_jsonl(
    path: Path,
    items: list[tuple[str, dict[str, Any]]],
) -> Path:
    rows = [
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        for custom_id, body in items
    ]
    write_jsonl(path, rows)
    return path


def submit_and_download_batch(
    client: OpenAI,
    jsonl_path: Path,
    out_dir: Path,
    *,
    completion_window: str = "24h",
    poll_interval_sec: int = 30,
    endpoint: str = "/v1/responses",
) -> Path:
    ensure_dir(out_dir)
    out_path = out_dir / "output.jsonl"
    if out_path.exists():
        return out_path

    meta_path = out_dir / "batch_meta.json"
    batch_id = None
    if meta_path.exists():
        prior = read_json(meta_path)
        if isinstance(prior.get("id"), str) and prior["id"]:
            batch_id = prior["id"]

    if batch_id is None:
        with jsonl_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint=endpoint,
            completion_window=completion_window,
        )
        write_json(meta_path, batch.model_dump())
        batch_id = batch.id
        print(f"Submitted batch {batch_id}; polling…")
    else:
        print(f"Resuming batch {batch_id}; polling…")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"  status={status}")
        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval_sec)

    write_json(out_dir / "batch_final.json", batch.model_dump())
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} finished with status={batch.status}")

    output_file_id = batch.output_file_id
    if not output_file_id:
        raise RuntimeError(f"Batch {batch.id} completed but has no output_file_id")

    content = client.files.content(output_file_id)
    out_path.write_bytes(content.read())
    if batch.error_file_id:
        err = client.files.content(batch.error_file_id)
        (out_dir / "errors.jsonl").write_bytes(err.read())
    return out_path


def parse_batch_output(output_jsonl: Path) -> dict[str, dict[str, Any]]:
    from tifbench.io import read_jsonl

    mapping: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(output_jsonl):
        custom_id = row["custom_id"]
        if row.get("error"):
            raise RuntimeError(
                f"Batch row error custom_id={custom_id}: {row.get('error')}"
            )
        resp = row.get("response") or {}
        if isinstance(resp, dict) and resp.get("status_code") not in (None, 200):
            raise RuntimeError(
                f"Batch row HTTP {resp.get('status_code')} custom_id={custom_id}: "
                f"{resp.get('body')}"
            )
        body = resp.get("body") if isinstance(resp, dict) else None
        mapping[custom_id] = body if isinstance(body, dict) else row
    return mapping


def load_batch_output_if_present(out_dir: Path) -> dict[str, dict[str, Any]] | None:
    path = out_dir / "output.jsonl"
    if path.exists():
        return parse_batch_output(path)
    return None
