"""Bridge sample-benchmark selected markets into construct-formal."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from tifbench.io import ensure_dir, read_json, write_json, write_json_atomic
from tifbench.polymarket.fetch import _parse_outcome, fetch_market_by_id
from tifbench.polymarket.sample_benchmark import sample_target_n
from tifbench.schemas import CuratedEvent


def cohort_identity_payload(
    sample_dir: Path,
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Canonical cohort identity fields (sorted ids for stable hashing)."""
    return {
        "sample_dir": str(sample_dir.resolve()),
        "collect_run_id": manifest.get("collect_run_id"),
        "curate_run_id": manifest.get("run_id"),
        "market_ids": sorted(str(r["market_id"]) for r in selected),
        "event_ids": sorted(str(r["event_id"]) for r in selected),
    }


def cohort_identity_sha256(
    sample_dir: Path,
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
) -> str:
    payload = cohort_identity_payload(sample_dir, manifest, selected)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_same_cohort_or_empty(
    out_dir: Path,
    *,
    expected_sha256: str,
    event_id_filter: str | None,
    limit: int | None = None,
    months: list[str] | None = None,
) -> None:
    """Refuse partial merges that would mix instances from another sample cohort."""
    partial = (
        event_id_filter is not None
        or limit is not None
        or months is not None
    )
    if not partial:
        return
    prior_path = out_dir / "cohort_from_sample.json"
    instances_path = out_dir / "instances.json"
    if not prior_path.exists() and not instances_path.exists():
        return
    if not prior_path.exists():
        raise RuntimeError(
            "partial build found existing instances.json without "
            "cohort_from_sample.json; refuse cross-cohort mix"
        )
    prior = read_json(prior_path)
    prior_sha = str(prior.get("cohort_identity_sha256") or "")
    if not prior_sha:
        raise RuntimeError(
            "partial build found cohort_from_sample.json without "
            "cohort_identity_sha256; refuse cross-cohort mix"
        )
    if prior_sha != expected_sha256:
        raise RuntimeError(
            "partial build cohort_identity_sha256 mismatch: "
            f"existing={prior_sha} incoming={expected_sha256}; "
            "refuse mixing another sample cohort's instances/report "
            "(--event-id / --limit / --months)"
        )


def load_sample_selection(
    sample_dir: Path, cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate a quota-complete sample curate directory."""
    selected_path = sample_dir / "selected.json"
    manifest_path = sample_dir / "curate_manifest.json"
    if not selected_path.exists() or not manifest_path.exists():
        raise RuntimeError(
            f"--from-sample requires {selected_path.name} and {manifest_path.name}"
        )
    selected = read_json(selected_path)
    manifest = read_json(manifest_path)
    if not isinstance(selected, list):
        raise RuntimeError("selected.json must be a list")
    fill = manifest.get("fill") or {}
    if fill.get("quota_complete") is not True:
        raise RuntimeError("sample cohort incomplete: need fill.quota_complete=true")
    if manifest.get("cross_bucket_dedupe_ok") is not True:
        raise RuntimeError("sample cohort incomplete: need cross_bucket_dedupe_ok=true")
    target = sample_target_n(cfg)
    fill_target = fill.get("target_n")
    if fill_target is not None and int(fill_target) != target:
        raise RuntimeError(
            f"fill.target_n={fill_target} != config target_n={target}"
        )
    if len(selected) != target:
        raise RuntimeError(
            f"selected n={len(selected)} != config target_n={target}"
        )
    event_ids = [str(r.get("event_id") or "") for r in selected]
    market_ids = [str(r.get("market_id") or "") for r in selected]
    if not all(event_ids) or not all(market_ids):
        raise RuntimeError("selected rows require event_id and market_id")
    if len(set(event_ids)) != len(event_ids):
        raise RuntimeError("selected.json has duplicate event_id")
    if len(set(market_ids)) != len(market_ids):
        raise RuntimeError("selected.json has duplicate market_id")
    manifest_selected = manifest.get("selected")
    if isinstance(manifest_selected, list):
        manifest_ids = {
            str(r.get("market_id") or "")
            for r in manifest_selected
            if isinstance(r, dict)
        }
        manifest_ids.discard("")
        if set(market_ids) != manifest_ids:
            raise RuntimeError(
                "selected.json market_ids != curate_manifest.selected market_ids"
            )
    return selected, manifest


def materialize_sample_markets(
    cfg: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Freeze Gamma market.json under data/raw/markets/{event_id}/."""
    gamma = str(cfg["polymarket"]["gamma_base_url"])
    raw_markets = ensure_dir(Path(cfg["paths"]["raw"]) / "markets")
    paths: list[Path] = []
    own_client = client is None
    http = client or httpx.Client(timeout=60.0)
    try:
        for row in selected:
            event_id = str(row["event_id"])
            market_id = str(row["market_id"])
            out = ensure_dir(raw_markets / event_id)
            market_path = out / "market.json"
            if market_path.exists():
                existing = read_json(market_path)
                if str(existing.get("id")) != market_id:
                    raise RuntimeError(
                        f"{event_id}: frozen market id {existing.get('id')} "
                        f"!= selected market_id {market_id}"
                    )
                paths.append(market_path)
                continue
            market = fetch_market_by_id(http, gamma, market_id)
            write_json(market_path, market)
            paths.append(market_path)
    finally:
        if own_client:
            http.close()
    return paths


def selected_to_events(
    cfg: dict[str, Any], selected: list[dict[str, Any]]
) -> list[CuratedEvent]:
    """Map sample rows to CuratedEvent; question/criteria/outcome from frozen market.json."""
    events: list[CuratedEvent] = []
    raw_markets = Path(cfg["paths"]["raw"]) / "markets"
    for row in selected:
        event_id = str(row["event_id"])
        market_path = raw_markets / event_id / "market.json"
        if not market_path.exists():
            raise RuntimeError(f"{event_id}: missing frozen market.json")
        market = read_json(market_path)
        market_outcome = _parse_outcome(market)
        if market_outcome not in (0, 1):
            raise RuntimeError(f"{event_id}: frozen market has no binary outcome")
        row_outcome = row.get("outcome")
        if row_outcome is not None and int(row_outcome) != market_outcome:
            raise RuntimeError(
                f"{event_id}: selected outcome {row_outcome} != market {market_outcome}"
            )
        question = str(market.get("question") or market.get("title") or "").strip()
        criteria = str(market.get("description") or "").strip()
        if not question or not criteria:
            raise RuntimeError(f"{event_id}: frozen market missing question or description")
        terminal = row.get("actual_terminal_time")
        if isinstance(terminal, str) and terminal:
            terminal = datetime.fromisoformat(terminal.replace("Z", "+00:00"))
        elif not terminal:
            terminal = None
        events.append(
            CuratedEvent(
                event_id=event_id,
                underlying_event_id=str(
                    row.get("underlying_event_key") or event_id
                ),
                question=question,
                resolution_criteria=criteria,
                category=str(row.get("domain") or ""),
                outcome=int(market_outcome),
                actual_terminal_time=terminal,
                exclusion_reason="",
            )
        )
    return events


def select_construct_work(
    selected: list[dict[str, Any]],
    *,
    event_id: str | None = None,
    limit: int | None = None,
    months: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Choose the subset of selected markets to build this run.

    ``limit`` takes the first N rows in ``selected.json`` order (stable).
    ``months`` keeps rows whose ``sample_month`` is in the list (config order
    already applied by the caller). ``event_id``, ``limit``, and ``months`` are
    mutually exclusive.
    """
    filters = [
        name
        for name, val in (
            ("event_id", event_id),
            ("limit", limit),
            ("months", months),
        )
        if val is not None
    ]
    if len(filters) > 1:
        raise ValueError(
            "Use only one of event_id, limit, and months "
            f"(got {', '.join(filters)})"
        )
    if event_id is not None:
        work = [r for r in selected if str(r.get("event_id")) == event_id]
        if not work:
            raise RuntimeError(f"No sample event matching event_id={event_id}")
        return work
    if limit is not None:
        n = int(limit)
        if n < 1:
            raise ValueError("limit must be >= 1")
        return list(selected)[:n]
    if months is not None:
        wanted = set(months)
        if not wanted:
            raise ValueError("months must be non-empty")
        work = [
            r for r in selected if str(r.get("sample_month") or "") in wanted
        ]
        if not work:
            raise RuntimeError(
                f"No selected markets with sample_month in {sorted(wanted)}"
            )
        return work
    return list(selected)


def write_cohort_audit(
    out_dir: Path,
    *,
    sample_dir: Path,
    manifest: dict[str, Any],
    selected: list[dict[str, Any]],
    built: list[dict[str, Any]],
    target_n: int,
    event_id: str | None = None,
    limit: int | None = None,
    months: list[str] | None = None,
) -> Path:
    path = out_dir / "cohort_from_sample.json"
    identity = cohort_identity_payload(sample_dir, manifest, selected)
    partial = len(built) < len(selected)
    write_json_atomic(
        path,
        {
            **identity,
            "sample_dir": str(sample_dir),
            "cohort_identity_sha256": cohort_identity_sha256(
                sample_dir, manifest, selected
            ),
            "target_n": target_n,
            "n_selected": len(selected),
            "built_event_ids": [str(r["event_id"]) for r in built],
            "event_id_filter": event_id,
            "limit": limit,
            "months_filter": list(months) if months is not None else None,
            "partial_build": partial,
            "note": (
                "selected market quota is not news READY / publish readiness; "
                "partial --event-id / --limit / --months builds must not publish "
                "benchmark.jsonl"
            ),
        },
    )
    return path
