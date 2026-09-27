"""OpenRouter chat-completions Batch submit/poll for formal forecasts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from tifbench.io import ensure_dir, read_json, write_json


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/tifbench",
        "X-Title": "tifbench",
    }


def _mapping_from_batch(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = batch.get("results")
    if not isinstance(results, list):
        raise RuntimeError(
            f"OpenRouter batch {batch.get('id')} completed without results"
        )
    mapping: dict[str, dict[str, Any]] = {}
    for row in results:
        custom_id = row.get("custom_id")
        if not custom_id:
            raise RuntimeError(f"OpenRouter batch row missing custom_id: {row}")
        if row.get("error"):
            raise RuntimeError(
                f"OpenRouter batch row error custom_id={custom_id}: {row.get('error')}"
            )
        resp = row.get("response") or {}
        status_code = resp.get("status_code")
        if status_code not in (None, 200):
            raise RuntimeError(
                f"OpenRouter batch row HTTP {status_code} custom_id={custom_id}: "
                f"{resp.get('body')}"
            )
        body = resp.get("body")
        if not isinstance(body, dict):
            raise RuntimeError(
                f"OpenRouter batch row missing body custom_id={custom_id}: {row}"
            )
        mapping[str(custom_id)] = body
    return mapping


def _poll_openrouter_batch(
    *,
    client: httpx.Client,
    root: str,
    headers: dict[str, str],
    batch_id: str,
    out_dir: Path,
    poll_interval_sec: int,
    not_found_grace_sec: float = 120.0,
) -> dict[str, Any]:
    """Poll until terminal status.

    OpenRouter can briefly 404 right after create (eventual consistency);
    tolerate that for ``not_found_grace_sec`` before failing.
    """
    print(f"Polling OpenRouter batch {batch_id}…")
    first_seen_at: float | None = None
    not_found_started_at: float | None = None
    while True:
        polled = client.get(f"{root}/batches/{batch_id}", headers=headers)
        if polled.status_code == 404:
            now = time.monotonic()
            if first_seen_at is not None:
                # Already resolved once; a later 404 is a real failure.
                polled.raise_for_status()
            if not_found_started_at is None:
                not_found_started_at = now
                print(f"  status=not_found (retrying up to {not_found_grace_sec:.0f}s)…")
            elif now - not_found_started_at >= not_found_grace_sec:
                polled.raise_for_status()
            time.sleep(min(poll_interval_sec, 5))
            continue
        polled.raise_for_status()
        if first_seen_at is None:
            first_seen_at = time.monotonic()
        batch = polled.json()
        status = batch.get("status")
        print(f"  status={status}")
        write_json(out_dir / "batch_final.json", batch)
        if status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_interval_sec)


def load_openrouter_batch_if_present(out_dir: Path) -> dict[str, dict[str, Any]] | None:
    """Reuse a completed batch_final.json if results are already on disk."""
    path = out_dir / "batch_final.json"
    if not path.exists():
        return None
    batch = read_json(path)
    if batch.get("status") != "completed":
        return None
    try:
        mapping = _mapping_from_batch(batch)
    except RuntimeError:
        return None
    return mapping or None


def submit_and_poll_openrouter_batch(
    *,
    api_key: str,
    base_url: str,
    model: str,
    requests: list[tuple[str, dict[str, Any]]],
    out_dir: Path,
    poll_interval_sec: int = 30,
    timeout_sec: float = 120.0,
) -> dict[str, dict[str, Any]]:
    """Submit one OpenRouter batch and return custom_id -> response body.

    If ``batch_meta.json`` already has a batch id (previous crash mid-poll),
    resume polling that id instead of creating a duplicate billed batch.
    """
    ensure_dir(out_dir)
    root = base_url.rstrip("/")
    headers = _headers(api_key)
    custom_ids = [custom_id for custom_id, _ in requests]

    existing = load_openrouter_batch_if_present(out_dir)
    if existing is not None:
        missing = [cid for cid in custom_ids if cid not in existing]
        if not missing:
            return existing

    meta_path = out_dir / "batch_meta.json"
    submit_path = out_dir / "batch_submit.json"
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": model,
        "completion_window": "24h",
        "requests": [
            {"custom_id": custom_id, "body": body}
            for custom_id, body in requests
        ],
    }

    with httpx.Client(timeout=timeout_sec) as client:
        batch_id: str | None = None
        if meta_path.exists():
            prior_meta = read_json(meta_path)
            prior_id = prior_meta.get("id")
            if isinstance(prior_id, str) and prior_id:
                # Only resume if the frozen submit matches this chunk's ids.
                if submit_path.exists():
                    prior_submit = read_json(submit_path)
                    prior_ids = [
                        row.get("custom_id")
                        for row in (prior_submit.get("requests") or [])
                        if isinstance(row, dict)
                    ]
                    if sorted(str(x) for x in prior_ids if x) == sorted(custom_ids):
                        batch_id = prior_id
                else:
                    batch_id = prior_id

        if batch_id is None:
            write_json(submit_path, payload)
            created = client.post(f"{root}/batches", json=payload, headers=headers)
            created.raise_for_status()
            batch = created.json()
            write_json(meta_path, batch)
            batch_id = batch.get("id")
            if not batch_id:
                raise RuntimeError(f"OpenRouter batch create missing id: {batch}")
            print(f"Submitted OpenRouter batch {batch_id}; polling…")
        else:
            print(f"Resuming OpenRouter batch {batch_id}; polling…")

        batch = _poll_openrouter_batch(
            client=client,
            root=root,
            headers=headers,
            batch_id=str(batch_id),
            out_dir=out_dir,
            poll_interval_sec=poll_interval_sec,
        )

    if batch.get("status") != "completed":
        raise RuntimeError(
            f"OpenRouter batch {batch_id} finished with status={batch.get('status')}: "
            f"{batch.get('error')}"
        )
    mapping = _mapping_from_batch(batch)
    write_json(out_dir / "batch_results_index.json", {"custom_ids": sorted(mapping)})
    return mapping
