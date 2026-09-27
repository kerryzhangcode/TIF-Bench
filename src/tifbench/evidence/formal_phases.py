"""Phased construct-formal orchestration (I/O-compatible with formal.py).

Phases: queries → retrieve → relevance (all horizons) → summary (all horizons)
→ assemble. GPT via Batch or bounded live; digests match ``_call_cached``.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from google_news_api import GoogleNewsClient
from google_news_api.exceptions import HTTPError

from tifbench.config import require_api_key
from tifbench.evidence.article import (
    ArticleFailure,
    extract_article,
)
from tifbench.evidence import formal as F
from tifbench.evidence.google_decode import (
    decode_publisher_urls_batch,
    is_google_rate_limit,
)
from tifbench.io import ensure_dir, read_json, write_json, write_json_atomic
from tifbench.openai_client import (
    ResponseJob,
    build_responses_body,
    execute_response_jobs,
    make_client,
)
from tifbench.polymarket.clock import (
    CLOCK_DEFINITION,
    parse_market_active_start,
    parse_market_closed_time,
)
from tifbench.polymarket.prices import freeze_formal_price_at_cutoff
from tifbench.polymarket.sample_benchmark import sample_target_n
from tifbench.polymarket.sample_bridge import (
    assert_same_cohort_or_empty,
    cohort_identity_sha256,
    load_sample_selection,
    materialize_sample_markets,
    select_construct_work,
    selected_to_events,
    write_cohort_audit,
)
from tifbench.schemas import (
    CuratedEvent,
    EvidenceItem,
    FormalForecastInstance,
    FormalNewsItem,
    ReasonCode,
)

PHASE_ORDER = ["queries", "retrieve", "relevance", "summary", "assemble"]


def expand_phases(phase: str) -> list[str]:
    if phase == "all":
        # Model stages: all relevance waves then all summary waves (inside construct_phased).
        return ["queries", "retrieve", "model", "assemble"]
    if phase in PHASE_ORDER:
        return [phase]
    raise ValueError(
        f"Unknown phase {phase!r}; expected one of {PHASE_ORDER + ['all']}"
    )


def _batch_custom_id(kind: str, *parts: str) -> str:
    """OpenAI Batch custom_id: typically [a-zA-Z0-9_-]{1,64} (no colons)."""
    import hashlib

    digest = hashlib.sha256("|".join((kind, *parts)).encode()).hexdigest()[:48]
    out = f"{kind}_{digest}"
    return out[:64]


def _news(cfg: dict) -> dict:
    return cfg["formal_news"]


def _gpt_mode(cfg: dict, override: str | None) -> Literal["batch", "live"]:
    mode = (override or _news(cfg).get("gpt_mode") or "batch").strip().lower()
    if mode not in ("batch", "live"):
        raise ValueError(f"gpt_mode must be batch|live, got {mode!r}")
    return mode  # type: ignore[return-value]


def _orch(cfg: dict) -> dict[str, int | float]:
    n = _news(cfg)
    return {
        "retrieve_max_workers": int(n.get("retrieve_max_workers", 1)),
        "retrieve_article_max_workers": int(n.get("retrieve_article_max_workers", 8)),
        "live_max_concurrent": int(n.get("live_max_concurrent", 4)),
        "batch_chunk_size": int(n.get("batch_chunk_size", 200)),
        "batch_poll_interval_sec": int(n.get("batch_poll_interval_sec", 30)),
        "google_decode_requests_per_minute": int(
            n.get("google_decode_requests_per_minute", 20)
        ),
        "google_decode_get_min_interval_sec": float(
            n.get("google_decode_get_min_interval_sec", 0.75)
        ),
        "google_decode_get_concurrency": int(
            n.get("google_decode_get_concurrency", 4)
        ),
        "google_decode_post_min_interval_sec": float(
            n.get("google_decode_post_min_interval_sec", 1.0)
        ),
        "google_decode_batch_size": int(n.get("google_decode_batch_size", 50)),
        "google_decode_max_retries": int(n.get("google_decode_max_retries", 5)),
    }


def _google_news_client(cfg: dict) -> GoogleNewsClient:
    o = _orch(cfg)
    # Google News RSS retrieval uses the client; article URL resolution uses
    # the batched decoder with its own 429 retry pacing.
    return GoogleNewsClient(
        language="en",
        country="US",
        timeout=20.0,
        requests_per_minute=int(o["google_decode_requests_per_minute"]),
        max_retries=0,
    )


def _run_jobs(
    client: Any,
    jobs: list[ResponseJob],
    *,
    cfg: dict,
    mode: Literal["batch", "live"],
    batch_subdir: Path,
) -> dict[str, tuple[dict[str, Any], str]]:
    if not jobs:
        return {}
    o = _orch(cfg)
    return execute_response_jobs(
        client,
        jobs,
        mode=mode,
        batch_dir=batch_subdir if mode == "batch" else None,
        batch_chunk_size=o["batch_chunk_size"],
        batch_poll_interval_sec=o["batch_poll_interval_sec"],
        live_max_concurrent=o["live_max_concurrent"],
    )


def _queries_list_path(event_root: Path) -> Path:
    return event_root / "queries" / "queries_list.json"


def _retrieve_manifest_path(folder: Path) -> Path:
    return folder / "retrieve_manifest.json"


def _relevance_queue_path(folder: Path) -> Path:
    return folder / "relevance_queue.json"


def _kept_path(folder: Path) -> Path:
    return folder / "kept_news.json"


def _seed_article_from_event(event_root: Path, folder: Path, source_id: str) -> None:
    dest = folder / "articles" / source_id
    html_path = dest / "publisher_v1.html"
    meta_path = dest / "publisher_v1.json"
    html_tmp = dest / "publisher_v1.html.tmp"
    resolution_path = dest / "resolution_v1.json"
    if html_path.exists() and meta_path.exists():
        return
    # Half-written dest must not be mixed with sibling files (HTML/meta/resolution).
    had_partial = html_path.exists() or meta_path.exists() or html_tmp.exists()
    for partial in (html_path, meta_path, html_tmp):
        if partial.exists():
            partial.unlink()
    if had_partial and resolution_path.exists():
        resolution_path.unlink()
    for sibling in sorted(event_root.glob("*_tminus*/articles/" + source_id)):
        if sibling.resolve() == dest.resolve():
            continue
        if not (sibling / "publisher_v1.html").exists():
            continue
        if not (sibling / "publisher_v1.json").exists():
            continue
        ensure_dir(dest)
        for name in ("publisher_v1.html", "publisher_v1.json", "resolution_v1.json"):
            src = sibling / name
            if not src.exists():
                continue
            target = dest / name
            if target.exists():
                continue
            try:
                target.hardlink_to(src)
            except OSError:
                shutil.copy2(src, target)
        return


def _load_queries(event_root: Path) -> list[str]:
    path = _queries_list_path(event_root)
    if not path.exists():
        raise RuntimeError(f"Missing queries freeze: {path} (run --phase queries)")
    queries = list(read_json(path).get("queries") or [])
    if not 2 <= len(queries) <= 3:
        raise ValueError(f"Invalid frozen queries at {path}")
    return queries


def _load_kept(folder: Path) -> list[FormalNewsItem]:
    path = _kept_path(folder)
    if not path.exists():
        return []
    return [FormalNewsItem.model_validate(x) for x in read_json(path)]


def phase_queries(
    cfg: dict,
    *,
    events: list[CuratedEvent],
    raw_root: Path,
    api_client: Any,
    mode: Literal["batch", "live"],
    batch_root: Path,
) -> None:
    news = _news(cfg)
    model, effort = news["model"], news["reasoning_effort"]
    jobs: list[ResponseJob] = []
    meta: list[tuple[str, Path]] = []
    for ev in events:
        event_root = ensure_dir(raw_root / ev.event_id)
        body = build_responses_body(
            model=model,
            input_text=F.QUERY_PROMPT.format(
                question=ev.question, criteria=ev.resolution_criteria
            ),
            reasoning_effort=effort,
            json_schema=F.QUERY_SCHEMA,
            schema_name="queries",
        )
        custom_id = _batch_custom_id("queries", ev.event_id)
        jobs.append(
            ResponseJob(
                custom_id=custom_id,
                body=body,
                # Same layout as generate_queries → _call_cached(root=event_root, name="queries")
                freeze_root=event_root,
                name="queries",
            )
        )
        meta.append((custom_id, event_root))
    results = _run_jobs(
        api_client, jobs, cfg=cfg, mode=mode, batch_subdir=batch_root / "queries"
    )
    for custom_id, event_root in meta:
        parsed, _ = results[custom_id]
        queries = list(
            dict.fromkeys(str(q).strip() for q in parsed.get("queries", []))
        )
        if not 2 <= len(queries) <= 3 or any(not q or len(q) > 180 for q in queries):
            raise ValueError(
                f"query generator did not return 2-3 valid queries for {event_root.name}"
            )
        F._freeze_json(
            _queries_list_path(event_root),
            {"queries": queries, "event_id": event_root.name},
        )


@dataclass
class _HorizonRetrieveWork:
    """RSS+decode output ready for publisher HTML fetch."""

    event_id: str
    instance_id: str
    horizon_hours: int
    folder: Path
    cutoff: datetime
    evidence_start: datetime
    market_close: datetime
    probability: float
    asof: datetime
    price_path: Path
    decoded: list[tuple[Any, dict[str, Any], str]]
    failures: list[dict[str, Any]]
    min_body_chars: int
    max_body_chars: int
    article_workers: int


_RETRIEVE_QUEUE_DEPTH = 2


def _prepare_horizon(
    cfg: dict,
    *,
    ev: CuratedEvent,
    event_root: Path,
    queries: list[str],
    market: dict[str, Any],
    market_close: datetime,
    evidence_start: datetime,
    horizon: int,
    http_client: httpx.Client,
) -> _HorizonRetrieveWork | None:
    """RSS + decode for one horizon. None means skip (done/excluded)."""
    news = _news(cfg)
    cutoff = market_close - timedelta(hours=int(horizon))
    instance_id = f"{ev.event_id}_tminus{int(horizon)}h"
    folder = ensure_dir(event_root / instance_id)
    excl = folder / "retrieve_exclude.json"
    if _retrieve_manifest_path(folder).exists():
        return None
    if evidence_start >= cutoff:
        write_json(excl, {"reason": "EMPTY_EVIDENCE_WINDOW"})
        return None
    probability, asof, _, price_err, price_path = freeze_formal_price_at_cutoff(
        http_client,
        cfg,
        market=market,
        cutoff=cutoff,
        folder=folder,
        freeze_name=F.PRICE_FREEZE_NAME,
    )
    if price_err or probability is None or asof is None:
        write_json(
            excl,
            {
                "reason": price_err.value if price_err else "NO_PRICE",
                "price_freeze": str(price_path),
            },
        )
        return None
    if excl.exists():
        excl.unlink()

    candidates = F._rss_candidates(
        http_client,
        folder,
        queries,
        evidence_start,
        cutoff,
        int(news["max_rss_candidates"]),
        slice_days=int(news.get("rss_slice_days", F.DEFAULT_RSS_SLICE_DAYS)),
    )
    decode_retries = int(news.get("google_decode_max_retries", 5))
    get_min = float(news.get("google_decode_get_min_interval_sec", 0.75))
    get_conc = int(news.get("google_decode_get_concurrency", 4))
    post_min = float(news.get("google_decode_post_min_interval_sec", 1.0))
    decode_batch = int(news.get("google_decode_batch_size", 50))
    min_body = int(news["min_body_chars"])
    max_body = int(news["max_body_chars"])
    article_workers = max(1, int(news.get("retrieve_article_max_workers", 8)))

    decoded: list[tuple[Any, dict[str, Any], str]] = []
    failures: list[dict[str, Any]] = []
    pending: list[tuple[Any, dict[str, Any], Path]] = []
    for item in candidates:
        _seed_article_from_event(event_root, folder, item.source_id)
        record: dict[str, Any] = {
            "source_id": item.source_id,
            "rss_url": item.url,
            "published_at": item.published_at.isoformat(),
            "title": item.title,
            "source_text": item.text,
        }
        article_dir = ensure_dir(folder / "articles" / item.source_id)
        resolution_path = article_dir / "resolution_v1.json"
        if resolution_path.exists():
            try:
                resolution = read_json(resolution_path)
            except json.JSONDecodeError:
                resolution_path.unlink()
            else:
                if (
                    resolution.get("rss_url") == item.url
                    and resolution.get("decoder") == F.URL_DECODER
                    and resolution.get("publisher_url")
                ):
                    decoded.append((item, record, str(resolution["publisher_url"])))
                    continue
                resolution_path.unlink()
        pending.append((item, record, article_dir))

    if pending:
        batch_out = decode_publisher_urls_batch(
            [item.url for item, _rec, _dir in pending],
            country="US",
            language="en",
            get_min_interval_sec=get_min,
            get_concurrency=get_conc,
            post_min_interval_sec=post_min,
            batch_size=decode_batch,
            max_retries=decode_retries,
        )
        for item, record, article_dir in pending:
            got = batch_out.get(item.url)
            if isinstance(got, BaseException):
                unresolved = (
                    is_google_rate_limit(got)
                    or isinstance(got, (ValueError, HTTPError))
                    or "ARTICLE_URL_UNRESOLVED" in str(got)
                )
                record["status"] = (
                    "ARTICLE_URL_UNRESOLVED"
                    if unresolved
                    else "ARTICLE_FETCH_OR_PARSE_FAILED"
                )
                detail = str(got)
                if unresolved and not detail.startswith("ARTICLE_URL_UNRESOLVED"):
                    detail = f"ARTICLE_URL_UNRESOLVED: {type(got).__name__}: {got}"
                record["detail"] = detail[:300]
                failures.append(record)
                continue
            if not isinstance(got, str):
                record["status"] = "ARTICLE_URL_UNRESOLVED"
                record["detail"] = "ARTICLE_URL_UNRESOLVED: missing batch result"
                failures.append(record)
                continue
            try:
                publisher_url = F._store_publisher_url(article_dir, item.url, got)
                decoded.append((item, record, publisher_url))
            except (ValueError, RuntimeError) as exc:
                record["status"] = "ARTICLE_URL_UNRESOLVED"
                record["detail"] = str(exc)[:300]
                failures.append(record)

    return _HorizonRetrieveWork(
        event_id=ev.event_id,
        instance_id=instance_id,
        horizon_hours=int(horizon),
        folder=folder,
        cutoff=cutoff,
        evidence_start=evidence_start,
        market_close=market_close,
        probability=float(probability),
        asof=asof,
        price_path=Path(price_path),
        decoded=decoded,
        failures=failures,
        min_body_chars=min_body,
        max_body_chars=max_body,
        article_workers=article_workers,
    )


def _fetch_horizon(work: _HorizonRetrieveWork) -> None:
    """Publisher HTML fetch + retrieve_manifest for one prepared horizon."""
    folder = work.folder
    cutoff = work.cutoff
    min_body = work.min_body_chars
    max_body = work.max_body_chars
    article_workers = work.article_workers
    decoded = work.decoded
    failures = list(work.failures)
    ready: list[dict[str, Any]] = []

    def _fetch_one(
        item: Any, record: dict[str, Any], publisher_url: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        try:
            with httpx.Client(timeout=30.0) as client:
                article, body_path = F._fetch_article_body(
                    client,
                    folder,
                    item,
                    publisher_url,
                    cutoff=cutoff,
                    min_body_chars=min_body,
                )
            body = str(article["body"])
            if F.MARKET_LEAKAGE.search(f"{item.title}\n{body}"):
                fail = {**record, "status": "PREDICTION_MARKET_LEAKAGE"}
                return fail, None
            ok = {
                **record,
                "status": "READY_FOR_RELEVANCE",
                "body_path": body_path,
                "body_for_model": body[:max_body],
                "article_url": str(article["article_url"]),
                "body_sha256": str(article["body_sha256"]),
                "published_at_article": article["published_at"].isoformat(),
                "modified_at": article["modified_at"].isoformat(),
            }
            return record, ok
        except ArticleFailure as exc:
            fail = {**record, "status": exc.reason, "detail": str(exc)[:300]}
            return fail, None
        except (httpx.HTTPError, ValueError, OSError, RuntimeError) as exc:
            fail = {
                **record,
                "status": "ARTICLE_FETCH_OR_PARSE_FAILED",
                "detail": str(exc)[:300],
            }
            return fail, None

    by_source: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {}
    if article_workers == 1 or len(decoded) <= 1:
        for item, record, url in decoded:
            by_source[item.source_id] = _fetch_one(item, record, url)
    else:
        with ThreadPoolExecutor(max_workers=article_workers) as pool:
            futs = {
                pool.submit(_fetch_one, item, record, url): item.source_id
                for item, record, url in decoded
            }
            for fut in as_completed(futs):
                by_source[futs[fut]] = fut.result()

    for item, record, _url in decoded:
        fail_or_base, ok = by_source[item.source_id]
        if ok is None:
            failures.append(fail_or_base)
        else:
            ready.append(ok)

    write_json_atomic(
        _retrieve_manifest_path(folder),
        {
            "instance_id": work.instance_id,
            "event_id": work.event_id,
            "horizon_hours": work.horizon_hours,
            "forecast_at": cutoff.isoformat(),
            "evidence_start": work.evidence_start.isoformat(),
            "market_close_time": work.market_close.isoformat(),
            "market_probability": work.probability,
            "market_probability_asof": work.asof.isoformat(),
            "price_freeze": str(work.price_path),
            "ready": ready,
            "failures": failures,
        },
    )


def _retrieve_one_event(
    cfg: dict,
    *,
    ev: CuratedEvent,
    raw_root: Path,
    http_client: httpx.Client,
    resolver: GoogleNewsClient,
) -> None:
    """Legacy sequential path (used when retrieve_max_workers > 1)."""
    del resolver  # decode uses batchexecute helper, not the RSS client
    news = _news(cfg)
    event_root = ensure_dir(raw_root / ev.event_id)
    queries = _load_queries(event_root)
    market_path = Path(cfg["paths"]["raw"]) / "markets" / ev.event_id / "market.json"
    market = read_json(market_path) if market_path.exists() else None
    market_close = parse_market_closed_time(market) if market else None
    exclude_path = event_root / "event_exclude.json"
    if ev.outcome not in (0, 1):
        write_json(exclude_path, {"reason": "MISSING_BINARY_OUTCOME"})
        return
    if market is None or market_close is None:
        write_json(
            exclude_path, {"reason": ReasonCode.MISSING_MARKET_CLOSE_TIME.value}
        )
        return
    if exclude_path.exists():
        exclude_path.unlink()

    window_days = int(news["window_days"])
    evidence_start = market_close - timedelta(days=window_days)
    for horizon in cfg["experiment"]["horizon_hours"]:
        work = _prepare_horizon(
            cfg,
            ev=ev,
            event_root=event_root,
            queries=queries,
            market=market,
            market_close=market_close,
            evidence_start=evidence_start,
            horizon=int(horizon),
            http_client=http_client,
        )
        if work is not None:
            _fetch_horizon(work)


def _iter_prepare_horizon_jobs(
    cfg: dict,
    *,
    events: list[CuratedEvent],
    raw_root: Path,
    http_client: httpx.Client,
):
    """Yield prepared horizon work in market×horizon order (RSS+decode serial)."""
    news = _news(cfg)
    window_days = int(news["window_days"])
    for ev in events:
        event_root = ensure_dir(raw_root / ev.event_id)
        exclude_path = event_root / "event_exclude.json"
        if ev.outcome not in (0, 1):
            write_json(exclude_path, {"reason": "MISSING_BINARY_OUTCOME"})
            continue
        market_path = Path(cfg["paths"]["raw"]) / "markets" / ev.event_id / "market.json"
        market = read_json(market_path) if market_path.exists() else None
        market_close = parse_market_closed_time(market) if market else None
        if market is None or market_close is None:
            write_json(
                exclude_path, {"reason": ReasonCode.MISSING_MARKET_CLOSE_TIME.value}
            )
            continue
        if exclude_path.exists():
            exclude_path.unlink()
        queries = _load_queries(event_root)
        evidence_start = market_close - timedelta(days=window_days)
        for horizon in cfg["experiment"]["horizon_hours"]:
            work = _prepare_horizon(
                cfg,
                ev=ev,
                event_root=event_root,
                queries=queries,
                market=market,
                market_close=market_close,
                evidence_start=evidence_start,
                horizon=int(horizon),
                http_client=http_client,
            )
            if work is not None:
                yield work


def phase_retrieve(
    cfg: dict, *, events: list[CuratedEvent], raw_root: Path
) -> None:
    workers = max(1, int(_orch(cfg)["retrieve_max_workers"]))
    # Default path: RSS+decode producer overlapped with HTML consumer.
    # With multiple workers, each event is retrieved independently.

    with _google_news_client(cfg) as resolver:
        if workers == 1:
            work_q: queue.Queue[_HorizonRetrieveWork | None] = queue.Queue(
                maxsize=_RETRIEVE_QUEUE_DEPTH
            )
            stop = threading.Event()
            errors: list[BaseException] = []

            def _consumer() -> None:
                while True:
                    item = work_q.get()
                    try:
                        if item is None:
                            return
                        if stop.is_set():
                            # Drain so producer put()/sentinel never blocks forever.
                            continue
                        _fetch_horizon(item)
                    except Exception as exc:
                        errors.append(exc)
                        stop.set()
                        # Keep draining until sentinel; do not return early.

            consumer = threading.Thread(
                target=_consumer, name="retrieve-html-consumer", daemon=True
            )
            consumer.start()
            try:
                with httpx.Client(timeout=30.0) as http_client:
                    for work in _iter_prepare_horizon_jobs(
                        cfg,
                        events=events,
                        raw_root=raw_root,
                        http_client=http_client,
                    ):
                        if stop.is_set() or errors:
                            break
                        while True:
                            if stop.is_set() or errors:
                                break
                            try:
                                work_q.put(work, timeout=1.0)
                                break
                            except queue.Full:
                                continue
            finally:
                # On error: skip remaining queued horizons (drain only).
                # On success: leave stop clear so the consumer finishes in-flight
                # queue items before the sentinel — otherwise the last
                # ~_RETRIEVE_QUEUE_DEPTH horizons never get manifests.
                if errors:
                    stop.set()
                while True:
                    try:
                        work_q.put(None, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                consumer.join()
            if errors:
                raise errors[0]
        else:

            def _task(ev: CuratedEvent) -> None:
                with httpx.Client(timeout=30.0) as http_client:
                    _retrieve_one_event(
                        cfg,
                        ev=ev,
                        raw_root=raw_root,
                        http_client=http_client,
                        resolver=resolver,
                    )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_task, ev) for ev in events]
                for fut in as_completed(futs):
                    fut.result()

    status_counts: Counter[str] = Counter()
    n_manifests = 0
    n_ready_articles = 0
    for ev in events:
        event_root = raw_root / ev.event_id
        for horizon in cfg["experiment"]["horizon_hours"]:
            folder = event_root / f"{ev.event_id}_tminus{int(horizon)}h"
            path = _retrieve_manifest_path(folder)
            if not path.exists():
                continue
            n_manifests += 1
            manifest = read_json(path)
            ready = manifest.get("ready") or []
            n_ready_articles += len(ready)
            for row in ready:
                status_counts[str(row.get("status") or "READY_FOR_RELEVANCE")] += 1
            for row in manifest.get("failures") or []:
                status_counts[str(row.get("status") or "UNKNOWN")] += 1
    top = ", ".join(f"{k}={v}" for k, v in status_counts.most_common(8))
    print(
        f"phase_retrieve done: manifests={n_manifests} "
        f"ready_articles={n_ready_articles} statuses=[{top}]"
    )


def _prior_kept(
    cfg: dict, event_root: Path, event_id: str, horizon_hours: int
) -> list[FormalNewsItem]:
    horizons = [int(h) for h in cfg["experiment"]["horizon_hours"]]
    idx = horizons.index(int(horizon_hours))
    if idx == 0:
        return []
    longer = horizons[idx - 1]
    return _load_kept(event_root / f"{event_id}_tminus{longer}h")


def _relevance_row_is_keeper(row: dict[str, Any]) -> bool:
    """Same keep gates as summary uses on scored relevance rows."""
    relevance = row.get("relevance") or {}
    score = relevance.get("score")
    quote = str(relevance.get("supporting_quote", "")).strip()
    body = str(row.get("body_for_model") or "")
    return (
        bool(relevance.get("relevant"))
        and isinstance(score, int)
        and score in F.RELEVANCE_KEEP_SCORES
        and bool(quote)
        and quote.casefold() in body.casefold()
    )


def _prior_relevance_passers(
    cfg: dict, event_root: Path, event_id: str, horizon_hours: int
) -> list[dict[str, Any]]:
    """Longer-horizon relevance keepers for cross-horizon carry (no summary needed)."""
    horizons = [int(h) for h in cfg["experiment"]["horizon_hours"]]
    idx = horizons.index(int(horizon_hours))
    if idx == 0:
        return []
    longer = horizons[idx - 1]
    path = _relevance_queue_path(event_root / f"{event_id}_tminus{longer}h")
    if not path.exists():
        return []
    queue = read_json(path)
    return [row for row in (queue.get("scored") or []) if _relevance_row_is_keeper(row)]


def _passer_published_at(row: dict[str, Any]) -> datetime:
    raw = row.get("published_at_article") or row.get("published_at")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(raw))


def _can_carry_relevance(prior: dict[str, Any], row: dict[str, Any]) -> bool:
    """Carry only when the article body is the same bytes the prior score judged.

    Same ``source_id`` can resolve to different HTML across horizons; reusing a
    prior quote against a new body fails summary keep gates.
    """
    prior_sha = str(prior.get("body_sha256") or "")
    row_sha = str(row.get("body_sha256") or "")
    if not prior_sha or prior_sha != row_sha:
        return False
    body = row.get("body_for_model") or prior.get("body_for_model")
    return _relevance_row_is_keeper({**prior, "body_for_model": body})


def phase_relevance_wave(
    cfg: dict,
    *,
    events: list[CuratedEvent],
    raw_root: Path,
    horizon_hours: int,
    api_client: Any,
    mode: Literal["batch", "live"],
    batch_root: Path,
) -> None:
    news = _news(cfg)
    model, effort = news["model"], news["reasoning_effort"]
    jobs: list[ResponseJob] = []
    folder_queues: dict[Path, dict[str, Any]] = {}

    for ev in events:
        event_root = raw_root / ev.event_id
        if (event_root / "event_exclude.json").exists():
            continue
        folder = event_root / f"{ev.event_id}_tminus{int(horizon_hours)}h"
        if (folder / "retrieve_exclude.json").exists():
            continue
        manifest_path = _retrieve_manifest_path(folder)
        if not manifest_path.exists():
            raise RuntimeError(f"Missing retrieve manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        cutoff = datetime.fromisoformat(manifest["forecast_at"])
        start = datetime.fromisoformat(manifest["evidence_start"])
        passer_by_id: dict[str, dict[str, Any]] = {}
        for prior in _prior_relevance_passers(
            cfg, event_root, ev.event_id, horizon_hours
        ):
            try:
                published = _passer_published_at(prior)
            except (TypeError, ValueError):
                continue
            if start <= published <= cutoff:
                passer_by_id[str(prior["source_id"])] = prior
        market_path = Path(cfg["paths"]["raw"]) / "markets" / ev.event_id / "market.json"
        market = read_json(market_path) if market_path.exists() else None
        market_start = parse_market_active_start(market) if market else None

        gpt_queue: list[dict[str, Any]] = []
        carried_stubs: list[dict[str, Any]] = []
        carried_ids: list[str] = []
        for row in manifest.get("ready") or []:
            source_id = str(row["source_id"])
            prior = passer_by_id.get(source_id)
            if prior is not None and _can_carry_relevance(prior, row):
                stub = {
                    **prior,
                    **row,
                    "relevance": prior["relevance"],
                    "relevance_hash": prior.get("relevance_hash")
                    or prior.get("relevance_request_sha256")
                    or "",
                    "carried_from_relevance": True,
                }
                stub.pop("custom_id", None)
                carried_stubs.append(stub)
                carried_ids.append(source_id)
                continue
            prompt = F.format_relevance_prompt(
                question=ev.question,
                criteria=ev.resolution_criteria,
                body=row["body_for_model"],
                market_start=market_start,
                forecast_cutoff=cutoff,
            )
            body = build_responses_body(
                model=model,
                input_text=prompt,
                reasoning_effort=effort,
                json_schema=F.RELEVANCE_SCHEMA,
                schema_name="relevance",
            )
            custom_id = _batch_custom_id(
                "rel", ev.event_id, str(horizon_hours), row["source_id"]
            )
            jobs.append(
                ResponseJob(
                    custom_id=custom_id,
                    body=body,
                    freeze_root=folder / "model_calls" / row["source_id"],
                    name="relevance",
                )
            )
            gpt_queue.append({**row, "custom_id": custom_id})
        folder_queues[folder] = {
            "horizon_hours": int(horizon_hours),
            "event_id": ev.event_id,
            "carried_source_ids": sorted(carried_ids),
            "queue": gpt_queue,
            "carried_stubs": carried_stubs,
        }

    results = _run_jobs(
        api_client,
        jobs,
        cfg=cfg,
        mode=mode,
        batch_subdir=batch_root / f"relevance_tminus{horizon_hours}h",
    )
    for folder, payload in folder_queues.items():
        scored: list[dict[str, Any]] = list(payload.pop("carried_stubs", []))
        for row in payload["queue"]:
            parsed, digest = results[row["custom_id"]]
            scored.append({**row, "relevance": parsed, "relevance_hash": digest})
        payload["scored"] = scored
        write_json(_relevance_queue_path(folder), payload)
    print(
        f"phase_relevance tminus{horizon_hours}h: "
        f"folders={len(folder_queues)} gpt_jobs={len(jobs)}"
    )


def phase_summary_wave(
    cfg: dict,
    *,
    events: list[CuratedEvent],
    raw_root: Path,
    horizon_hours: int,
    api_client: Any,
    mode: Literal["batch", "live"],
    batch_root: Path,
) -> None:
    news = _news(cfg)
    model, effort = news["model"], news["reasoning_effort"]
    max_summary_words = int(news.get("max_summary_words", 1000))
    max_summary_words_hard = int(news.get("max_summary_words_hard", 1200))
    jobs: list[ResponseJob] = []
    work: list[dict[str, Any]] = []

    for ev in events:
        event_root = raw_root / ev.event_id
        if (event_root / "event_exclude.json").exists():
            continue
        folder = event_root / f"{ev.event_id}_tminus{int(horizon_hours)}h"
        if (folder / "retrieve_exclude.json").exists():
            continue
        manifest = read_json(_retrieve_manifest_path(folder))
        queue = read_json(_relevance_queue_path(folder))
        cutoff = datetime.fromisoformat(manifest["forecast_at"])
        start = datetime.fromisoformat(manifest["evidence_start"])
        carry = [
            n
            for n in _prior_kept(cfg, event_root, ev.event_id, horizon_hours)
            if start <= n.published_at <= cutoff
        ]
        carried_ids = {n.source_id for n in carry}
        decisions: list[dict[str, Any]] = []
        for news_item in carry:
            decisions.append(
                {
                    "source_id": news_item.source_id,
                    "rss_url": news_item.rss_url,
                    "published_at": news_item.published_at.isoformat(),
                    "status": "KEPT_CARRIED_FORWARD",
                    **F._score_audit(
                        news_item.relevance_score, news_item.published_at, cutoff
                    ),
                }
            )
        for row in manifest.get("failures") or []:
            decisions.append(dict(row))

        relevant: list[dict[str, Any]] = []
        for row in queue.get("scored") or []:
            if str(row["source_id"]) in carried_ids:
                continue
            relevance = row["relevance"]
            score = relevance.get("score")
            quote = str(relevance.get("supporting_quote", "")).strip()
            body_for_model = row["body_for_model"]
            record: dict[str, Any] = {
                "source_id": row["source_id"],
                "rss_url": row["rss_url"],
                "published_at": row["published_at"],
            }
            if (
                not relevance.get("relevant")
                or not isinstance(score, int)
                or score not in F.RELEVANCE_KEEP_SCORES
                or not quote
                or quote.casefold() not in body_for_model.casefold()
            ):
                record["status"] = "SEMANTICALLY_IRRELEVANT"
                record["relevance"] = relevance
                decisions.append(record)
                continue
            published_at = datetime.fromisoformat(row["published_at_article"])
            record.update(
                {
                    "status": "RELEVANT_PENDING_DEDUP",
                    "relevance": relevance,
                    "relevance_request_sha256": row["relevance_hash"],
                    **F._score_audit(score, published_at, cutoff),
                }
            )
            decisions.append(record)
            relevant.append(
                {
                    "item": EvidenceItem(
                        source_id=row["source_id"],
                        title=row["title"],
                        url=row["rss_url"],
                        published_at=datetime.fromisoformat(row["published_at"]),
                        text=row.get("source_text") or "",
                    ),
                    "article": {
                        "article_url": row["article_url"],
                        "body_sha256": row["body_sha256"],
                        "published_at": published_at,
                        "modified_at": datetime.fromisoformat(row["modified_at"]),
                    },
                    "body_path": row["body_path"],
                    "body": body_for_model,
                    "score": score,
                    "quote": quote,
                    "relevance_hash": row["relevance_hash"],
                    "record": record,
                }
            )

        relevant.sort(
            key=lambda x: F._rank_key(
                x["score"], x["article"]["published_at"], x["item"].source_id, cutoff
            )
        )
        representatives: list[dict[str, Any]] = []
        for news_item in carry:
            prior_html = Path(news_item.body_path).read_bytes().decode("utf-8")
            prior_body = str(
                extract_article(prior_html, news_item.article_url)["body"]
            )
            representatives.append(
                {
                    "item": EvidenceItem(
                        source_id=news_item.source_id,
                        title=news_item.title,
                        url=news_item.rss_url,
                        published_at=news_item.published_at,
                        text="",
                    ),
                    "article": {
                        "article_url": news_item.article_url,
                        "body_sha256": news_item.body_sha256,
                    },
                    "body": prior_body[: int(news["max_body_chars"])],
                    "quote": news_item.relevance_quote,
                }
            )
        for candidate in relevant:
            duplicate = F._duplicate_of(candidate, representatives)
            if duplicate:
                candidate["record"].update(
                    {"status": "DUPLICATE_REPORT", "duplicate_of": duplicate}
                )
            else:
                representatives.append(candidate)

        summary_targets = [c for c in representatives if "record" in c]
        for candidate in summary_targets:
            item = candidate["item"]
            body = build_responses_body(
                model=model,
                input_text=F.format_summary_prompt(
                    question=ev.question,
                    quote=candidate["quote"],
                    title=item.title,
                    body=candidate["body"],
                    max_summary_words=max_summary_words,
                ),
                reasoning_effort=effort,
                json_schema=F.SUMMARY_SCHEMA,
                schema_name="summary",
            )
            custom_id = _batch_custom_id(
                "sum", ev.event_id, str(horizon_hours), item.source_id
            )
            jobs.append(
                ResponseJob(
                    custom_id=custom_id,
                    body=body,
                    freeze_root=folder / "model_calls" / item.source_id,
                    name="summary",
                )
            )
            candidate["summary_custom_id"] = custom_id

        work.append(
            {
                "folder": folder,
                "ev": ev,
                "cutoff": cutoff,
                "carry": carry,
                "decisions": decisions,
                "summary_targets": summary_targets,
                "manifest": manifest,
                "max_summary_words_hard": max_summary_words_hard,
            }
        )

    results = _run_jobs(
        api_client,
        jobs,
        cfg=cfg,
        mode=mode,
        batch_subdir=batch_root / f"summary_tminus{horizon_hours}h",
    )

    for item in work:
        folder: Path = item["folder"]
        cutoff: datetime = item["cutoff"]
        kept: list[FormalNewsItem] = list(item["carry"])
        decisions: list[dict] = item["decisions"]
        for candidate in item["summary_targets"]:
            record = candidate["record"]
            custom_id = candidate["summary_custom_id"]
            try:
                summary_result, summary_hash = results[custom_id]
                summary = str(summary_result.get("summary", "")).strip()
                hard_cap = int(item.get("max_summary_words_hard", max_summary_words_hard))
                if not summary or len(summary.split()) > hard_cap:
                    record["status"] = "INVALID_SUMMARY"
                    continue
                if F.MARKET_LEAKAGE.search(summary):
                    record["status"] = "PREDICTION_MARKET_LEAKAGE"
                    continue
                art = candidate["article"]
                src = candidate["item"]
                published_at = art["published_at"]
                kept.append(
                    FormalNewsItem(
                        source_id=src.source_id,
                        title=src.title,
                        rss_url=src.url,
                        article_url=str(art["article_url"]),
                        source=src.text.split(":", 1)[0],
                        published_at=published_at,
                        modified_at=art["modified_at"],
                        body_sha256=str(art["body_sha256"]),
                        body_path=candidate["body_path"],
                        relevance_score=candidate["score"],
                        relevance_quote=candidate["quote"],
                        summary=summary,
                        model=model,
                        relevance_request_sha256=candidate["relevance_hash"],
                        summary_request_sha256=summary_hash,
                    )
                )
                record.update(
                    {
                        "status": "KEPT",
                        "summary_request_sha256": summary_hash,
                        **F._score_audit(candidate["score"], published_at, cutoff),
                    }
                )
            except (KeyError, ValueError, TypeError) as exc:
                record["status"] = "SUMMARY_FAILED"
                record["detail"] = str(exc)[:300]

        kept.sort(
            key=lambda x: F._rank_key(
                x.relevance_score, x.published_at, x.source_id, cutoff
            )
        )
        write_json(folder / "source_decisions.json", decisions)
        write_json(
            _kept_path(folder),
            [x.model_dump(mode="json") for x in kept],
        )
    kept_total = sum(len(read_json(_kept_path(item["folder"]))) for item in work)
    print(
        f"phase_summary tminus{horizon_hours}h: "
        f"folders={len(work)} gpt_jobs={len(jobs)} kept_articles={kept_total}"
    )


def _horizon_has_scored_relevance(folder: Path) -> bool:
    path = _relevance_queue_path(folder)
    if not path.exists():
        return False
    return "scored" in read_json(path)


def _horizon_has_kept(folder: Path) -> bool:
    return _kept_path(folder).exists()


def _pending_horizon_folders(
    events: list[CuratedEvent], raw_root: Path, horizon_hours: int
) -> list[Path]:
    folders: list[Path] = []
    for ev in events:
        event_root = raw_root / ev.event_id
        if (event_root / "event_exclude.json").exists():
            continue
        folder = event_root / f"{ev.event_id}_tminus{int(horizon_hours)}h"
        if (folder / "retrieve_exclude.json").exists():
            continue
        if not _retrieve_manifest_path(folder).exists():
            continue
        folders.append(folder)
    return folders


def phase_model(
    cfg: dict,
    *,
    events: list[CuratedEvent],
    raw_root: Path,
    api_client: Any,
    mode: Literal["batch", "live"],
    batch_root: Path,
    only: Literal["relevance", "summary", "both"] = "both",
) -> None:
    """Run model stages.

    ``relevance``: all horizons long→short (relevance-pass carry; no kept_news gate).
    ``summary``: all horizons long→short (kept carry when longer kept exists).
    ``both`` (``--phase all``): all relevance waves, then all summary waves.
    """
    horizons = [int(h) for h in cfg["experiment"]["horizon_hours"]]
    ran_relevance = 0
    ran_summary = 0

    if only in ("relevance", "both"):
        for h in horizons:
            pending = _pending_horizon_folders(events, raw_root, h)
            if not pending:
                continue
            if all(_horizon_has_scored_relevance(f) for f in pending):
                continue
            phase_relevance_wave(
                cfg,
                events=events,
                raw_root=raw_root,
                horizon_hours=h,
                api_client=api_client,
                mode=mode,
                batch_root=batch_root,
            )
            ran_relevance += 1
            print(f"relevance wave complete: tminus{h}h")

    if only in ("summary", "both"):
        for h in horizons:
            pending = _pending_horizon_folders(events, raw_root, h)
            if not pending:
                continue
            if all(_horizon_has_kept(f) for f in pending):
                continue
            if not all(_horizon_has_scored_relevance(f) for f in pending):
                raise RuntimeError(
                    f"Cannot run summary for tminus{h}h: run --phase relevance first"
                )
            phase_summary_wave(
                cfg,
                events=events,
                raw_root=raw_root,
                horizon_hours=h,
                api_client=api_client,
                mode=mode,
                batch_root=batch_root,
            )
            ran_summary += 1
            print(f"summary wave complete: tminus{h}h")

    if ran_relevance == 0 and ran_summary == 0:
        print(f"phase_model only={only}: nothing pending")


def phase_assemble(
    cfg: dict,
    *,
    events: list[CuratedEvent],
    work_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    sample_dir: Path,
    sample_manifest: dict[str, Any],
    identity_sha: str,
    target_n: int,
    out_dir: Path,
    raw_root: Path,
    event_id: str | None,
    limit: int | None = None,
    months: list[str] | None = None,
) -> Path:
    instances: list[FormalForecastInstance] = []
    report: list[dict[str, Any]] = []
    required_news = int(_news(cfg)["min_articles"])

    for ev in sorted(events, key=lambda x: x.event_id):
        event_root = raw_root / ev.event_id
        market_path = Path(cfg["paths"]["raw"]) / "markets" / ev.event_id / "market.json"
        market = read_json(market_path) if market_path.exists() else None
        exclude = event_root / "event_exclude.json"
        if exclude.exists():
            report.append(
                {
                    "event_id": ev.event_id,
                    "status": "EXCLUDE",
                    "reason": read_json(exclude)["reason"],
                    "clock_definition": CLOCK_DEFINITION,
                    "market_close_time": None,
                }
            )
            continue
        market_close = parse_market_closed_time(market) if market else None
        assert market_close is not None
        for horizon in cfg["experiment"]["horizon_hours"]:
            instance_id = f"{ev.event_id}_tminus{int(horizon)}h"
            folder = event_root / instance_id
            retr_excl = folder / "retrieve_exclude.json"
            if retr_excl.exists():
                payload = read_json(retr_excl)
                report.append(
                    {
                        "instance_id": instance_id,
                        "status": "EXCLUDE",
                        "reason": payload["reason"],
                        "clock_definition": CLOCK_DEFINITION,
                        "market_close_time": market_close.isoformat(),
                        "forecast_at": None,
                        "price_freeze": payload.get("price_freeze"),
                    }
                )
                continue
            manifest = read_json(_retrieve_manifest_path(folder))
            kept = _load_kept(folder)
            decisions = (
                read_json(folder / "source_decisions.json")
                if (folder / "source_decisions.json").exists()
                else []
            )
            cutoff = datetime.fromisoformat(manifest["forecast_at"])
            evidence_start = datetime.fromisoformat(manifest["evidence_start"])
            probability = float(manifest["market_probability"])
            asof = datetime.fromisoformat(manifest["market_probability_asof"])
            price_path = Path(manifest["price_freeze"])
            report_fields = {
                "rss_candidates_found": F._rss_found(folder),
                "rss_candidates_checked": len(decisions)
                - sum(row.get("status") == "KEPT_CARRIED_FORWARD" for row in decisions),
                "kept": len(kept),
                "required_news": required_news,
                "max_k_supported": F._supported_k(len(kept)),
                "evidence_start": evidence_start.isoformat(),
                "forecast_at": cutoff.isoformat(),
                "market_close_time": market_close.isoformat(),
                "clock_definition": CLOCK_DEFINITION,
                "source_status_counts": F._construction_counts(decisions),
            }
            if len(kept) < required_news:
                report.append(
                    {
                        "instance_id": instance_id,
                        "status": "EXCLUDE",
                        "reason": ReasonCode.INSUFFICIENT_TIME_SAFE_NEWS.value,
                        **report_fields,
                    }
                )
                continue
            instance = FormalForecastInstance(
                instance_id=instance_id,
                event_id=ev.event_id,
                underlying_event_id=ev.underlying_event_id or ev.event_id,
                horizon_hours=int(horizon),
                forecast_at=cutoff,
                market_close_time=market_close,
                clock_definition=CLOCK_DEFINITION,
                actual_terminal_time=ev.actual_terminal_time,
                evidence_start=evidence_start,
                outcome=ev.outcome,
                market_probability=probability,
                market_probability_asof=asof,
                question=ev.question,
                resolution_criteria=ev.resolution_criteria,
                news_items=kept,
                construction_version=F.PROTOCOL,
                provenance={
                    "market": str(market_path),
                    "market_sha256": F._digest(market),
                    "sample_dir": str(sample_dir),
                    "cohort_from_sample": str(out_dir / "cohort_from_sample.json"),
                    "queries": str(event_root / "queries"),
                    "rss_query_manifest": str(folder / "queries.json"),
                    "rss_audit": str(folder / "rss_audit.json"),
                    "source_decisions": str(folder / "source_decisions.json"),
                    "instance_record": str(folder / "instance.json"),
                    "price_at_cutoff": str(price_path),
                    "url_decoder": F.URL_DECODER,
                    "construction_config_sha256": F._digest(_news(cfg)),
                    "clock_definition": CLOCK_DEFINITION,
                },
            )
            F._freeze_json(folder / "instance.json", instance.model_dump(mode="json"))
            instances.append(instance)
            report.append(
                {
                    "instance_id": instance_id,
                    "status": "READY",
                    "market_probability": probability,
                    "market_probability_asof": asof.isoformat(),
                    "price_freeze": str(price_path),
                    **report_fields,
                }
            )
            print(
                f"{instance_id}: {len(kept)} time-safe relevant articles "
                f"(clock={CLOCK_DEFINITION})"
            )

    output = out_dir / "instances.json"
    if event_id and output.exists():
        existing = [
            FormalForecastInstance.model_validate(x) for x in read_json(output)
        ]
        instances = [x for x in existing if x.event_id != event_id] + instances
        prior_report = (
            read_json(out_dir / "construction_report.json")
            if (out_dir / "construction_report.json").exists()
            else []
        )
        report = [
            row
            for row in prior_report
            if row.get("event_id") != event_id
            and not str(row.get("instance_id", "")).startswith(event_id + "_tminus")
        ] + report
    write_json(output, [x.model_dump(mode="json") for x in instances])
    write_json(out_dir / "construction_report.json", report)

    partial = len(work_rows) < len(selected)
    if partial:
        print(
            f"Partial build (event_id={event_id!r} limit={limit!r} "
            f"months={months!r}): "
            f"wrote {len(instances)} instances for {len(work_rows)}/{len(selected)} "
            "selected; run without --limit/--event-id/--months to publish benchmark.jsonl"
        )
    else:
        from tifbench.benchmark import default_benchmark_dir, export_benchmark_from_instances

        benchmark_path = export_benchmark_from_instances(
            output,
            default_benchmark_dir(cfg),
            required_horizons=tuple(int(h) for h in cfg["experiment"]["horizon_hours"]),
            min_articles=int(cfg["formal_news"]["min_articles"]),
        )
        print(f"Published complete-horizon benchmark at {benchmark_path}")

    snapshot_rows = [row for row in report if "instance_id" in row]
    source_totals = Counter()
    for row in snapshot_rows:
        source_totals.update(row.get("source_status_counts", {}))
    horizons = [int(h) for h in cfg["experiment"]["horizon_hours"]]
    selected_market_n = len(selected)
    ready_event_ids = sorted(
        {
            str(row.get("event_id") or str(row["instance_id"]).split("_tminus")[0])
            for row in snapshot_rows
            if row.get("status") == "READY"
        }
    )
    ready_by_horizon: dict[str, int] = {}
    shortfall_by_horizon: dict[str, int] = {}
    for h in horizons:
        key = f"tminus{h}h"
        n_ready = sum(
            1
            for row in snapshot_rows
            if row.get("status") == "READY"
            and str(row.get("instance_id", "")).endswith(f"_tminus{h}h")
        )
        ready_by_horizon[key] = n_ready
        shortfall_by_horizon[key] = max(0, selected_market_n - n_ready)
    ready_markets_all_horizons = 0
    for eid in {str(r.get("event_id")) for r in selected}:
        ok = all(
            any(
                row.get("status") == "READY"
                and str(row.get("instance_id", "")) == f"{eid}_tminus{h}h"
                for row in snapshot_rows
            )
            for h in horizons
        )
        if ok:
            ready_markets_all_horizons += 1
    write_json(
        out_dir / "construction_summary.json",
        {
            "construction_version": F.PROTOCOL,
            "clock_definition": CLOCK_DEFINITION,
            "sample_dir": str(sample_dir),
            "cohort_identity_sha256": identity_sha,
            "partial_event_id_build": bool(event_id) and partial,
            "partial_limit_build": limit is not None and partial,
            "partial_months_build": months is not None and partial,
            "limit_n": limit,
            "months_filter": list(months) if months is not None else None,
            "built_markets_n": len(work_rows),
            "selected_markets_quota_target_n": target_n,
            "selected_markets_n": selected_market_n,
            "selected_markets_note": (
                "selected_markets_n is the sample-benchmark quota cohort size "
                f"(target_n={target_n}); it is NOT automatic publish readiness."
            ),
            "ready_markets": len(ready_event_ids),
            "ready_markets_all_horizons": ready_markets_all_horizons,
            "ready_markets_shortfall_vs_selected": max(
                0, selected_market_n - len(ready_event_ids)
            ),
            "ready_markets_all_horizons_shortfall_vs_selected": max(
                0, selected_market_n - ready_markets_all_horizons
            ),
            "ready_by_horizon": ready_by_horizon,
            "ready_shortfall_by_horizon_vs_selected": shortfall_by_horizon,
            "no_automatic_backfill": True,
            "target_n": target_n,
            "candidate_events": len(
                {
                    row.get("event_id") or row["instance_id"].split("_tminus")[0]
                    for row in report
                    if "instance_id" in row or "event_id" in row
                }
            ),
            "ready_snapshots": sum(row["status"] == "READY" for row in snapshot_rows),
            "excluded_snapshots": sum(
                row["status"] == "EXCLUDE" for row in snapshot_rows
            ),
            "ready_news_counts": [
                row["kept"] for row in snapshot_rows if row["status"] == "READY"
            ],
            "source_status_totals": dict(sorted(source_totals.items())),
            "exclusion_reasons": dict(
                sorted(
                    Counter(
                        row["reason"]
                        for row in report
                        if row["status"] == "EXCLUDE"
                    ).items()
                )
            ),
            "post_benchmark_random_audit": "NOT_YET_PERFORMED",
        },
    )
    print(
        f"Formal construction ({F.PROTOCOL}, {CLOCK_DEFINITION}): "
        f"{len(instances)} instances; ready_markets={len(ready_event_ids)}/"
        f"{selected_market_n} selected; "
        f"report at {out_dir / 'construction_report.json'}"
    )
    return output


def construct_formal_dataset_phased(
    cfg: dict,
    *,
    from_sample: Path,
    event_id: str | None = None,
    limit: int | None = None,
    months: list[str] | None = None,
    phase: str = "all",
    gpt_mode: str | None = None,
) -> Path:
    """Construct the benchmark through the staged pipeline."""
    sample_dir = Path(from_sample)
    selected, sample_manifest = load_sample_selection(sample_dir, cfg)
    work = select_construct_work(
        selected, event_id=event_id, limit=limit, months=months
    )
    target_n = sample_target_n(cfg)
    identity_sha = cohort_identity_sha256(sample_dir, sample_manifest, selected)
    raw_root = ensure_dir(Path(cfg["paths"]["raw"]) / F.RAW_SUBDIR)
    out_dir = ensure_dir(Path(cfg["paths"]["curated"]) / F.CURATED_SUBDIR)
    assert_same_cohort_or_empty(
        out_dir,
        expected_sha256=identity_sha,
        event_id_filter=event_id,
        limit=limit,
        months=months,
    )
    materialize_sample_markets(cfg, work)
    events = selected_to_events(cfg, work)
    write_cohort_audit(
        out_dir,
        sample_dir=sample_dir,
        manifest=sample_manifest,
        selected=selected,
        built=work,
        target_n=target_n,
        event_id=event_id,
        limit=limit,
        months=months,
    )
    print(
        f"construct-formal work={len(work)}/{len(selected)} selected "
        f"(event_id={event_id!r} limit={limit!r} months={months!r} phase={phase} "
        f"gpt_mode={_gpt_mode(cfg, gpt_mode)})"
    )

    mode = _gpt_mode(cfg, gpt_mode)
    # Namespace Batch dirs by the *work set* so --limit 12 then --limit 50 do not
    # reuse a stale chunk output.jsonl that lacks the new custom_ids. Per-article
    # freezes under each event still reuse across limit increases.
    import hashlib

    work_key = hashlib.sha256(
        ",".join(sorted(str(r["event_id"]) for r in work)).encode()
    ).hexdigest()[:12]
    batch_root = ensure_dir(
        raw_root / "_batches" / identity_sha[:12] / f"work_{work_key}"
    )
    phases = expand_phases(phase)
    needs_api = phase in ("all", "queries", "relevance", "summary")
    api_client = (
        make_client(require_api_key(_news(cfg)["api_key_env"])) if needs_api else None
    )

    if "queries" in phases:
        assert api_client is not None
        phase_queries(
            cfg,
            events=events,
            raw_root=raw_root,
            api_client=api_client,
            mode=mode,
            batch_root=batch_root,
        )
    if "retrieve" in phases:
        phase_retrieve(cfg, events=events, raw_root=raw_root)
    if phase == "relevance":
        assert api_client is not None
        phase_model(
            cfg,
            events=events,
            raw_root=raw_root,
            api_client=api_client,
            mode=mode,
            batch_root=batch_root,
            only="relevance",
        )
    elif phase == "summary":
        assert api_client is not None
        phase_model(
            cfg,
            events=events,
            raw_root=raw_root,
            api_client=api_client,
            mode=mode,
            batch_root=batch_root,
            only="summary",
        )
    elif "model" in phases:
        assert api_client is not None
        phase_model(
            cfg,
            events=events,
            raw_root=raw_root,
            api_client=api_client,
            mode=mode,
            batch_root=batch_root,
            only="both",
        )
    if "assemble" in phases or phase == "all":
        return phase_assemble(
            cfg,
            events=events,
            work_rows=work,
            selected=selected,
            sample_dir=sample_dir,
            sample_manifest=sample_manifest,
            identity_sha=identity_sha,
            target_n=target_n,
            out_dir=out_dir,
            raw_root=raw_root,
            event_id=event_id,
            limit=limit,
            months=months,
        )
    print(f"construct-formal phase={phase} complete (no assemble)")
    return out_dir / "instances.json"
