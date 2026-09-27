"""Forecast a frozen formal cohort with identical per-article top-K evidence.

Official model inputs come only from the published benchmark.jsonl (question,
resolution criteria, cutoff, and ranked evidence text). Outcome and Polymarket
prices are used solely for scoring.

Default transport is Batch-first (OpenAI Batch for Astra, OpenRouter Batch for
Claude). Grok and Jev stay live; ``forecast_mode=live`` forces all
batch-capable models onto live concurrent calls.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from tifbench.benchmark import (
    BENCHMARK_NAME,
    default_benchmark_dir,
    file_sha256,
    load_benchmark_rows,
    outcome_int,
    parse_utc_iso,
    row_content_sha256,
)
from tifbench.evidence.forecast import check_forecast_leakage
from tifbench.eval.brier import brier_score, mean_brier
from tifbench.evidence.formal import DERIVED_SUBDIR
from tifbench.io import ensure_dir, read_json, write_json_atomic
from tifbench.models.openai_responses_forecast import ForecastWorkItem
from tifbench.models.registry import get_model
from tifbench.models.usage_accounting import (
    cost_summary_rows,
    pricing_digest,
    resolve_cost_usd,
)
from tifbench.schemas import EvidenceItem, FormalForecastInstance, ForecastEvidencePacket, ForecastResult

ALLOWED_TOP_K = (3, 5, 7, 10)


def _row_horizon_hours(row: dict[str, Any]) -> int:
    return int(round(float(row["forecast"]["horizon_days"]) * 24))


def packet_for_instance(instance: FormalForecastInstance, top_k: int) -> ForecastEvidencePacket:
    """Audit helper for construction-era FormalForecastInstance objects."""
    if top_k < 1 or len(instance.news_items) < top_k:
        raise ValueError(f"{instance.instance_id}: fewer than {top_k} frozen articles")
    items = [EvidenceItem(source_id=news.source_id, title=news.title,
                          url=news.article_url, published_at=news.published_at,
                          text=news.summary)
             for news in instance.news_items[:top_k]]
    packet = ForecastEvidencePacket(event_id=instance.instance_id,
                                    forecast_at=instance.forecast_at, items=items)
    if check_forecast_leakage(packet, instance.forecast_at, inclusive_cutoff=True):
        raise ValueError(f"{instance.instance_id}: evidence after forecast cutoff")
    if any(news.modified_at > instance.forecast_at for news in instance.news_items[:top_k]):
        raise ValueError(f"{instance.instance_id}: article updated after cutoff")
    return packet


def packet_for_benchmark(row: dict[str, Any], top_k: int) -> ForecastEvidencePacket:
    """Build the model evidence packet from a publishable benchmark row."""
    instance_id = row["instance_id"]
    evidence = row["evidence"]
    if top_k < 1 or len(evidence) < top_k:
        raise ValueError(f"{instance_id}: fewer than {top_k} frozen articles")
    cutoff = parse_utc_iso(row["forecast"]["cutoff_utc"])
    start = parse_utc_iso(row["market"]["start_utc"])
    close = parse_utc_iso(row["market"]["close_utc"])
    if not (start <= cutoff <= close):
        raise ValueError(f"{instance_id}: start/cutoff/close time ordering violated")
    ranked = sorted(evidence, key=lambda item: item["rank"])
    if [item["rank"] for item in ranked] != list(range(1, len(ranked) + 1)):
        raise ValueError(f"{instance_id}: evidence ranks must be contiguous 1..n")
    selected = ranked[:top_k]
    items = [
        EvidenceItem(
            source_id=item["article_id"],
            title=item["title"],
            url=item["url"],
            published_at=parse_utc_iso(item["published_at_utc"]),
            text=item["text"],
        )
        for item in selected
    ]
    packet = ForecastEvidencePacket(event_id=instance_id, forecast_at=cutoff, items=items)
    if check_forecast_leakage(packet, cutoff, inclusive_cutoff=True):
        raise ValueError(f"{instance_id}: evidence after forecast cutoff")
    return packet


def _cohort_meta(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    model_names: list[str],
    benchmark_path: Path,
    horizon_hours: int | None = None,
    min_evidence: int | None = None,
) -> dict:
    """Fingerprint the exact forecast cohort so resume cannot silently mix runs.

    Includes the dataset file hash and per-row content hashes so identical IDs /
    evidence IDs with changed text cannot resume into a prior run.

    ``horizon_hours`` / ``min_evidence`` are recorded on the meta for audits but
    are intentionally omitted from the sha256 payload: the eligible instance set
    already encodes those filters, and omitting them keeps default-path resumes
    compatible with older run_meta files that lacked these keys.
    """
    effective_min_evidence = top_k if min_evidence is None else int(min_evidence)
    instance_rows = []
    for row in rows:
        packet = packet_for_benchmark(row, top_k)
        evidence_prefix = sorted(row["evidence"], key=lambda item: item["rank"])[:top_k]
        instance_rows.append({
            "instance_id": row["instance_id"],
            "market_id": row["market"]["market_id"],
            "horizon_days": row["forecast"]["horizon_days"],
            "evidence_ids": [item.source_id for item in packet.items],
            "evidence_text_sha256": [
                hashlib.sha256(item["text"].encode()).hexdigest() for item in evidence_prefix
            ],
            "row_sha256": row_content_sha256(row),
            "outcome": row["outcome"],
        })
    payload = {
        "top_k": top_k,
        "models": list(model_names),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": file_sha256(benchmark_path),
        "instances": instance_rows,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return {
        "top_k": top_k,
        "models": list(model_names),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": payload["benchmark_sha256"],
        "instance_ids": [r["instance_id"] for r in instance_rows],
        "n_instances": len(instance_rows),
        "cohort_sha256": digest,
        "horizon_hours": horizon_hours,
        "min_evidence": effective_min_evidence,
    }


def _summary_rows(records: list[dict], top_k: int) -> list[dict]:
    scores: dict[tuple[str, float], list[float]] = {}
    for row in records:
        scores.setdefault((row["model"], row["horizon_days"]), []).append(row["brier_score"])
    return [
        {
            "model": model,
            "horizon_days": horizon,
            "top_k": top_k,
            "independent_markets": len({
                r["market_id"] for r in records
                if r["model"] == model and r["horizon_days"] == horizon
            }),
            "n": len(values),
            "mean_brier": mean_brier(values),
        }
        for (model, horizon), values in sorted(scores.items())
    ]


def _persist_run(out_dir: Path, *, meta: dict, records: list[dict], top_k: int) -> None:
    write_json_atomic(out_dir / "run_meta.json", meta)
    write_json_atomic(out_dir / "forecasts.json", records)
    write_json_atomic(out_dir / "summary.json", _summary_rows(records, top_k))
    write_json_atomic(out_dir / "cost_summary.json", cost_summary_rows(records))


def _forecast_mode(cfg: dict, override: str | None) -> Literal["batch", "live"]:
    mode = (override or cfg.get("experiment", {}).get("forecast_mode") or "batch")
    mode = str(mode).strip().lower()
    if mode not in ("batch", "live"):
        raise ValueError(f"forecast_mode must be batch|live, got {mode!r}")
    return mode  # type: ignore[return-value]


def _record_from_result(
    *,
    row: dict[str, Any],
    packet: ForecastEvidencePacket,
    result: ForecastResult,
    top_k: int,
    run_id: str,
    forecast_mode: Literal["batch", "live"],
    pricing_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cutoff = parse_utc_iso(row["forecast"]["cutoff_utc"])
    outcome = outcome_int(row["outcome"])
    usage = result.usage
    cost_usd, cost_source = resolve_cost_usd(
        usage=usage,
        model_key=result.model,
        forecast_mode=forecast_mode,
        pricing_cfg=pricing_cfg,
    )
    return {
        "instance_id": row["instance_id"],
        "market_id": row["market"]["market_id"],
        "horizon_days": row["forecast"]["horizon_days"],
        "horizon_hours": int(round(float(row["forecast"]["horizon_days"]) * 24)),
        "top_k": top_k,
        "evidence_ids": [item.source_id for item in packet.items],
        "forecast_at": cutoff.isoformat(),
        "model": result.model,
        "requested_model": result.requested_model,
        "resolved_model": result.resolved_model,
        "prompt_version": result.prompt_version,
        "raw_path": result.raw_path,
        "probability_yes": result.probability_yes,
        "outcome": outcome,
        "brier_score": brier_score(result.probability_yes, outcome),
        "market_probability": row["polymarket"]["yes_probability"],
        "run_id": run_id,
        "usage": usage,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
    }


def run_formal_forecasts(
    cfg: dict,
    *,
    top_k: int,
    resume_run_id: str | None = None,
    forecast_mode: str | None = None,
    horizon_hours: int | None = None,
    min_evidence: int | None = None,
) -> Path:
    if top_k not in ALLOWED_TOP_K:
        raise ValueError(f"top_k must be one of {ALLOWED_TOP_K}")
    if horizon_hours is not None and int(horizon_hours) <= 0:
        raise ValueError("horizon_hours must be a positive int when set")
    effective_min_evidence = top_k if min_evidence is None else int(min_evidence)
    if effective_min_evidence < top_k:
        raise ValueError(
            f"min_evidence ({effective_min_evidence}) must be >= top_k ({top_k})"
        )
    mode = _forecast_mode(cfg, forecast_mode)
    experiment = cfg.get("experiment") or {}
    pricing_cfg = experiment.get("forecast_pricing")
    if pricing_cfg is not None and not isinstance(pricing_cfg, dict):
        raise ValueError("experiment.forecast_pricing must be a mapping when set")
    live_max_concurrent = int(experiment.get("live_max_concurrent", 4))
    batch_chunk_size = int(experiment.get("batch_chunk_size", 200))
    batch_poll_interval_sec = int(experiment.get("batch_poll_interval_sec", 30))

    benchmark_dir = default_benchmark_dir(cfg)
    benchmark_path = benchmark_dir / BENCHMARK_NAME
    if not benchmark_path.exists():
        raise RuntimeError(
            f"Missing published benchmark at {benchmark_path}; "
            "run export-benchmark (or construct-formal) first"
        )
    rows = load_benchmark_rows(benchmark_path)
    # Each K has its own explicitly reported cohort; never discard a valid
    # top-3 snapshot merely because it cannot supply top-10 evidence.
    # Optional filters: fixed horizon and/or a shared min-evidence floor so
    # several top-k values can share one market set (e.g. 3d ∩ ≥10).
    eligible = [
        row for row in rows
        if len(row["evidence"]) >= effective_min_evidence
        and (
            horizon_hours is None
            or _row_horizon_hours(row) == int(horizon_hours)
        )
    ]
    if not eligible:
        raise RuntimeError(
            "No benchmark instances match filters "
            f"(min_evidence>={effective_min_evidence}"
            + (
                f", horizon_hours={int(horizon_hours)}"
                if horizon_hours is not None
                else ""
            )
            + ")"
        )

    model_names = list(cfg["experiment"]["models"])
    cohort = _cohort_meta(
        eligible,
        top_k=top_k,
        model_names=model_names,
        benchmark_path=benchmark_path,
        horizon_hours=int(horizon_hours) if horizon_hours is not None else None,
        min_evidence=effective_min_evidence,
    )
    derived_root = ensure_dir(Path(cfg["paths"]["derived"]) / DERIVED_SUBDIR)

    records: list[dict] = []
    done: set[tuple[str, str]] = set()
    if resume_run_id:
        out_dir = derived_root / resume_run_id
        meta_path = out_dir / "run_meta.json"
        forecasts_path = out_dir / "forecasts.json"
        if not meta_path.exists():
            raise RuntimeError(f"Cannot resume: missing {meta_path}")
        prior_meta = read_json(meta_path)
        for key in ("top_k", "models", "instance_ids", "cohort_sha256", "benchmark_sha256"):
            if prior_meta.get(key) != cohort.get(key):
                raise RuntimeError(
                    f"Cannot resume {resume_run_id}: cohort/{key} mismatch "
                    "(refusing to mix runs or cohorts)"
                )
        prior_horizon = prior_meta.get("horizon_hours")
        prior_min_evidence = prior_meta.get("min_evidence", prior_meta.get("top_k"))
        if prior_horizon != cohort.get("horizon_hours") or prior_min_evidence != cohort.get(
            "min_evidence"
        ):
            raise RuntimeError(
                f"Cannot resume {resume_run_id}: horizon_hours/min_evidence mismatch "
                "(refusing to mix filter cohorts)"
            )
        if forecasts_path.exists():
            records = list(read_json(forecasts_path))
            for row in records:
                if row.get("run_id") != resume_run_id:
                    raise RuntimeError(
                        f"Cannot resume {resume_run_id}: forecasts.json mixes run_id values"
                    )
                done.add((row["instance_id"], row["model"]))
        run_id = resume_run_id
        meta = prior_meta
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        out_dir = ensure_dir(derived_root / run_id)
        meta = {
            **cohort,
            "run_id": run_id,
            "forecast_mode": mode,
            "forecast_pricing_sha256": pricing_digest(
                pricing_cfg if isinstance(pricing_cfg, dict) else {}
            ),
            "status": "in_progress",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _persist_run(out_dir, meta=meta, records=records, top_k=top_k)

    meta = {
        **meta,
        "forecast_mode": mode,
        "forecast_pricing_sha256": pricing_digest(
            pricing_cfg if isinstance(pricing_cfg, dict) else {}
        ),
    }
    models = [get_model(name, cfg) for name in model_names]
    packets = {row["instance_id"]: packet_for_benchmark(row, top_k) for row in eligible}
    rows_by_id = {row["instance_id"]: row for row in eligible}

    for model in models:
        pending_rows = [
            row for row in eligible
            if (row["instance_id"], model.name) not in done
        ]
        if not pending_rows:
            continue

        if hasattr(model, "forecast_many"):
            items = [
                ForecastWorkItem(
                    instance_id=row["instance_id"],
                    question=row["market"]["question"],
                    resolution_criteria=row["market"]["resolution_criteria"],
                    forecast_at=parse_utc_iso(row["forecast"]["cutoff_utc"]),
                    packet=packets[row["instance_id"]],
                )
                for row in pending_rows
            ]
            batch_dir = ensure_dir(out_dir / "batches" / model.name)

            def _on_results(
                chunk_results: list[ForecastResult],
                *,
                _model_name: str = model.name,
            ) -> None:
                for result in chunk_results:
                    key = (result.event_id, _model_name)
                    if key in done:
                        continue
                    row = rows_by_id[result.event_id]
                    records.append(
                        _record_from_result(
                            row=row,
                            packet=packets[result.event_id],
                            result=result,
                            top_k=top_k,
                            run_id=run_id,
                            forecast_mode=mode,
                            pricing_cfg=pricing_cfg if isinstance(pricing_cfg, dict) else None,
                        )
                    )
                    done.add(key)
                _persist_run(out_dir, meta=meta, records=records, top_k=top_k)

            model.forecast_many(
                items,
                run_id=run_id,
                mode=mode,
                batch_dir=batch_dir,
                batch_chunk_size=batch_chunk_size,
                batch_poll_interval_sec=batch_poll_interval_sec,
                live_max_concurrent=live_max_concurrent,
                on_results=_on_results,
            )
            continue

        # Live path (Jev and test stubs without forecast_many).
        for row in pending_rows:
            packet = packets[row["instance_id"]]
            cutoff = parse_utc_iso(row["forecast"]["cutoff_utc"])
            result = model.forecast(
                event_id=row["instance_id"],
                question=row["market"]["question"],
                resolution_criteria=row["market"]["resolution_criteria"],
                forecast_at=cutoff,
                packet=packet,
                run_id=run_id,
            )
            records.append(
                _record_from_result(
                    row=row,
                    packet=packet,
                    result=result,
                    top_k=top_k,
                    run_id=run_id,
                    forecast_mode=mode,
                    pricing_cfg=pricing_cfg if isinstance(pricing_cfg, dict) else None,
                )
            )
            done.add((row["instance_id"], model.name))
            # Survive API errors on later markets: persist after each prediction.
            _persist_run(out_dir, meta=meta, records=records, top_k=top_k)

    expected = len(eligible) * len(models)
    if len(records) != expected:
        raise RuntimeError(
            f"Formal run incomplete: {len(records)}/{expected} predictions at {out_dir}"
        )
    meta = {
        **meta,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_predictions": len(records),
    }
    _persist_run(out_dir, meta=meta, records=records, top_k=top_k)
    print(f"Formal forecasts: {len(records)} predictions, output at {out_dir}")
    return out_dir
