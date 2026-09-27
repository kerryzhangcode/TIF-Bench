"""Stratified sample collect — sole candidate-entry Stage 1.

Stage 1: structural hard filters (including POST_DEADLINE_OBSERVATION) +
Gamma /events/keyset cursor pagination + tag×month hash sample + horizon
price gates (CLOB only for sampled candidates).
Stage 2 monthly target is ``per_month_n`` × months (domains soft-balanced).
Tags are retrieval-only; Stage 2 assigns the final domain label.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from uuid import uuid4

import httpx

from tifbench.io import ensure_dir, read_json, write_json_atomic
from tifbench.polymarket.clock import (
    CLOCK_DEFINITION,
    parse_market_active_start,
    parse_market_closed_time,
)
from tifbench.polymarket.fetch import (
    _parse_outcome,
    _yes_token_id,
    fetch_price_history,
)
from tifbench.polymarket.post_deadline import (
    POST_DEADLINE_OBSERVATION,
    post_deadline_observation_reason,
)
from tifbench.polymarket.prices import (
    formal_price_window_bounds,
    latest_price_at_or_before,
)
from tifbench.polymarket.settled import is_settled_yes_no
from tifbench.schemas import ReasonCode

SELECTION_VERSION = "sample_benchmark"
DEFAULT_MAX_PAGES_PER_TAG = 50
DEFAULT_PAGE_LIMIT = 100
MAX_KEYSET_LIMIT = 100
EVENTS_KEYSET_PATH = "/events/keyset"
EVENTS_LIST_ORDER = "closedTime"
EVENTS_LIST_ASCENDING = "false"
EARLIEST_CUTOFF_HOURS = 504  # paper benchmark: 21 days before close
TAG_MONTH_SAMPLING_CAP_REASON = "PRESCREEN_TAG_MONTH_SAMPLING_CAP"
DEFAULT_TAG_MONTH_SAMPLE_CAP = 50

# Retrieval tags verified via Gamma GET /tags/{id}. NOT domain assignment.
RETRIEVAL_TAGS: tuple[dict[str, Any], ...] = (
    {"tag_id": 2, "label": "Politics", "verified_via_path": "/tags/2"},
    {"tag_id": 100265, "label": "Geopolitics", "verified_via_path": "/tags/100265"},
    {"tag_id": 120, "label": "Finance", "verified_via_path": "/tags/120"},
    {"tag_id": 107, "label": "Business", "verified_via_path": "/tags/107"},
    {"tag_id": 1401, "label": "Tech", "verified_via_path": "/tags/1401"},
    {"tag_id": 74, "label": "Science", "verified_via_path": "/tags/74"},
    {"tag_id": 596, "label": "Culture", "verified_via_path": "/tags/596"},
)
STALE_ENTERTAINMENT_TAG_ID = 315

ListPagesFn = Callable[..., dict[str, Any]]
FetchCutoffHistoryFn = Callable[..., list[dict[str, Any]]]


def sample_section(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["sample_benchmark"]


def sample_months(cfg: dict[str, Any]) -> list[str]:
    return [str(m) for m in sample_section(cfg)["months"]]


def parse_sample_months(
    raw: str | list[str] | None,
    *,
    allowed: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[str] | None:
    """Parse a comma-separated or list month filter; preserve config order.

    Returns ``None`` when ``raw`` is None/empty (meaning: all protocol months).
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        months = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        months = [str(m).strip() for m in raw if str(m).strip()]
    if not months:
        return None
    allow = list(allowed) if allowed is not None else sample_months(cfg or {})
    allow_set = set(allow)
    bad = [m for m in months if m not in allow_set]
    if bad:
        raise ValueError(
            f"months not in sample_benchmark.months: {bad}; allowed={allow}"
        )
    wanted = set(months)
    return [m for m in allow if m in wanted]


def sample_domains(cfg: dict[str, Any]) -> list[str]:
    return [str(d) for d in sample_section(cfg)["domains"]]


def per_month_n(cfg: dict[str, Any]) -> int:
    """Hard monthly select count for Stage 2 (domains are soft-balanced only)."""
    sec = sample_section(cfg)
    if "per_month_n" in sec:
        return int(sec["per_month_n"])
    # Legacy fixtures that only set per_bucket_n.
    return int(sec["per_bucket_n"])


def sample_target_n(cfg: dict[str, Any]) -> int:
    return per_month_n(cfg) * len(sample_months(cfg))


def sample_window(cfg: dict[str, Any]) -> tuple[datetime, datetime]:
    sec = sample_section(cfg)
    start = datetime.fromisoformat(str(sec["window_start_utc"]).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(sec["window_end_utc_exclusive"]).replace("Z", "+00:00"))
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def bucket_key(month: str, domain: str) -> str:
    return f"{month}::{domain}"


def _normalize_cursor(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _events_keyset_params(
    *,
    tag_id: int,
    limit: int,
    after_cursor: str | None = None,
) -> dict[str, Any]:
    """Gamma /events/keyset query params. Never includes offset."""
    if not 1 <= int(limit) <= MAX_KEYSET_LIMIT:
        raise ValueError(f"keyset page limit must be 1..{MAX_KEYSET_LIMIT}")
    params: dict[str, Any] = {
        "closed": "true",
        "tag_id": int(tag_id),
        "limit": int(limit),
        "order": EVENTS_LIST_ORDER,
        "ascending": EVENTS_LIST_ASCENDING,
    }
    cursor = _normalize_cursor(after_cursor)
    if cursor is not None:
        params["after_cursor"] = cursor
    return params


def _events_keyset_url(
    gamma_base: str,
    *,
    tag_id: int,
    limit: int,
    after_cursor: str | None = None,
) -> str:
    base = f"{gamma_base.rstrip('/')}{EVENTS_KEYSET_PATH}"
    query = urlencode(
        _events_keyset_params(tag_id=tag_id, limit=limit, after_cursor=after_cursor)
    )
    return f"{base}?{query}"


def fetch_closed_events_keyset_page(
    client: httpx.Client,
    gamma_base: str,
    *,
    tag_id: int,
    limit: int,
    after_cursor: str | None = None,
) -> dict[str, Any]:
    """Fetch one Gamma /events/keyset page. Returns {events, next_cursor}; no offset."""
    params = _events_keyset_params(
        tag_id=tag_id, limit=limit, after_cursor=after_cursor
    )
    assert "offset" not in params
    r = client.get(
        f"{gamma_base.rstrip('/')}{EVENTS_KEYSET_PATH}",
        params=params,
        timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Gamma /events/keyset (tag_id={tag_id}) did not return an object"
        )
    events = data.get("events")
    if events is None:
        events = []
    if not isinstance(events, list):
        raise RuntimeError(
            f"Gamma /events/keyset (tag_id={tag_id}) events field was not a list"
        )
    return {
        "events": events,
        "next_cursor": _normalize_cursor(data.get("next_cursor")),
    }


def month_from_closed_time(
    closed: datetime,
    *,
    window_start: datetime,
    window_end: datetime,
) -> str | None:
    if closed.tzinfo is None:
        closed = closed.replace(tzinfo=timezone.utc)
    closed = closed.astimezone(timezone.utc)
    if not (window_start <= closed < window_end):
        return None
    return f"{closed.year:04d}-{closed.month:02d}"


def has_explicit_close_start_rules(market: dict[str, Any]) -> bool:
    """Require nonempty resolution text plus parseable close and start clocks.

    Clarity of criteria is a Stage-2 GPT-6 Sol API judgment — Stage 1 only requires
    nonempty rules text (not an arbitrary minimum character count).
    """
    criteria = str(market.get("description") or "").strip()
    if not criteria:
        return False
    if parse_market_closed_time(market) is None:
        return False
    if parse_market_active_start(market) is None:
        return False
    return True


def start_allows_longest_horizon(
    market: dict[str, Any], longest_horizon_hours: int = EARLIEST_CUTOFF_HOURS
) -> bool:
    """Require the market to be active at its earliest forecast cutoff."""
    closed = parse_market_closed_time(market)
    start = parse_market_active_start(market)
    if closed is None or start is None:
        return False
    return start <= closed - timedelta(hours=longest_horizon_hours)


def screen_sample_metadata(market: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Stage-1 objective metadata gates; content eligibility belongs to Stage 2."""
    if not market.get("closed"):
        return "NOT_CLOSED"
    if not is_settled_yes_no(market):
        return "NOT_SETTLED_YES_NO"
    if not has_explicit_close_start_rules(market):
        return "MISSING_EXPLICIT_CLOSE_START_OR_RULES"
    closed = parse_market_closed_time(market)
    if closed is None:
        return "MISSING_MARKET_CLOSE_TIME"
    window_start, window_end = sample_window(cfg)
    if month_from_closed_time(closed, window_start=window_start, window_end=window_end) is None:
        return "OUTSIDE_SAMPLE_WINDOW"
    if not start_allows_longest_horizon(market, max(cfg["experiment"]["horizon_hours"])):
        return "START_AFTER_EARLIEST_CUTOFF"
    if not _yes_token_id(market):
        return "MISSING_YES_TOKEN"
    pdo = post_deadline_observation_reason(market)
    if pdo is not None:
        return pdo
    return None


def price_ok_at_all_cutoffs(
    history_by_horizon: dict[int, list[dict[str, Any]]],
    *,
    closed: datetime,
    horizons: tuple[int, ...],
    max_lag_hours: float,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Require a usable latest-≤cutoff price at every configured horizon."""
    detail: dict[str, Any] = {}
    for hours in horizons:
        cutoff = closed - timedelta(hours=int(hours))
        history = history_by_horizon.get(int(hours), [])
        hit, err = latest_price_at_or_before(history, cutoff, max_lag_hours)
        key = f"tminus{hours}h"
        if err is not None or hit is None:
            code = err.value if isinstance(err, ReasonCode) else "NO_PRICE_AT_CUTOFF"
            detail[key] = {"ok": False, "error": code}
            return False, f"PRICE_FAIL_{key}_{code}", detail
        detail[key] = {
            "ok": True,
            "asof": hit.asof.isoformat(),
            "lag_hours": hit.lag_hours,
            "probability": hit.probability,
        }
    return True, None, detail


def sample_tag_month_pool(
    candidates: list[dict[str, Any]], *, cap_per_tag_month: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stable hash sample: at most ``cap_per_tag_month`` per month×retrieval_tag_id.

    Ranking key is ``sha256(month|retrieval_tag_id|market_id)`` then ``market_id``.
    Returns ``(offered, sampled_out)``. Every input row appears in exactly one list.
    """
    if cap_per_tag_month < 1:
        raise ValueError("cap_per_tag_month must be >= 1")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[
            (str(row.get("sample_month") or ""), str(row.get("retrieval_tag_id") or ""))
        ].append(row)
    offered: list[dict[str, Any]] = []
    sampled_out: list[dict[str, Any]] = []
    for (month, tag), rows in sorted(groups.items()):
        ranked = sorted(
            rows,
            key=lambda r: (
                hashlib.sha256(
                    f"{month}|{tag}|{r.get('market_id')}".encode("utf-8")
                ).hexdigest(),
                str(r.get("market_id") or ""),
            ),
        )
        offered.extend(ranked[:cap_per_tag_month])
        sampled_out.extend(ranked[cap_per_tag_month:])

    def by_market_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: str(r.get("market_id") or ""))

    return by_market_id(offered), by_market_id(sampled_out)


def _nested_markets(event: dict[str, Any]) -> list[dict[str, Any]]:
    markets = event.get("markets")
    if isinstance(markets, list):
        return [m for m in markets if isinstance(m, dict)]
    return []


def _criteria_block(cfg: dict[str, Any]) -> dict[str, Any]:
    gamma = cfg["polymarket"]["gamma_base_url"]
    horizons = tuple(int(h) for h in cfg["experiment"]["horizon_hours"])
    max_lag = float(cfg["polymarket"]["max_price_lag_hours"])
    months = sample_months(cfg)
    domains = sample_domains(cfg)
    n_per = per_month_n(cfg)
    window_start, window_end = sample_window(cfg)
    sec = sample_section(cfg)
    return {
        "selection_version": SELECTION_VERSION,
        "protocol": {
            "window_start_utc": window_start.isoformat(),
            "window_end_utc_exclusive": window_end.isoformat(),
            "clock": CLOCK_DEFINITION,
            "months": months,
            "domain_labels": domains,
            "per_month_n": n_per,
            "n_months": len(months),
            "target_n": sample_target_n(cfg),
            "domain_quota": "soft_prompt_only",
        },
        "retrieval_tags": [
            {
                "tag_id": t["tag_id"],
                "label": t["label"],
                "verified_via": f"{gamma.rstrip('/')}{t['verified_via_path']}",
                "role": "candidate_retrieval_only",
            }
            for t in RETRIEVAL_TAGS
        ],
        "stale_tag_ids_not_used": [STALE_ENTERTAINMENT_TAG_ID],
        "domain_assignment": "stage2_semantic_api_only",
        "stage1": {
            "closed": True,
            "settled_yes_no": True,
            "explicit_close_start_rules": True,
            "nonempty_resolution_text": True,
            "min_rule_chars": None,
            "clarity_deferred_to_stage2": True,
            "start_le_close_minus_hours": max(cfg["experiment"]["horizon_hours"]),
            "requires_yes_token": True,
            "uses_market_closed_time_not_event": True,
            "stage1_mechanical_exclusions": [POST_DEADLINE_OBSERVATION],
            "post_deadline_exclusion_surface": "description",
            "semantic_exclusions_deferred_to_stage2_api": [
                "sports",
                "weather",
                "social_post_count",
                "word_count",
                "short_horizon_asset_price",
                "live_structured_data_or_odds_or_sensor_or_high_frequency",
            ],
            "pagination": "gamma_events_keyset_cursor",
            "pagination_stop": (
                "Stop a tag scan only on empty events, absent next_cursor, or "
                "max_pages cap. Never use offset with /events/keyset. "
                "Do not infer completion from one pre-window nested market. "
                "Capped scans define a bounded, non-exhaustive candidate frame."
            ),
            "tag_month_sample_before_price": {
                "cap": int(sec.get("prescreen_candidates_per_tag_month", DEFAULT_TAG_MONTH_SAMPLE_CAP)),
                "sampling": "sha256(month|retrieval_tag_id|market_id)",
                "sampled_out_reason": TAG_MONTH_SAMPLING_CAP_REASON,
            },
            "price_at_all_horizons": {
                "horizons_hours": list(horizons),
                "max_lag_hours": max_lag,
                "selection": "latest_observation_at_or_before_cutoff",
                "preferred_fetch": "short_clob_window_startTs_endTs",
                "only_after_tag_month_sample": True,
            },
        },
        "stage2": {
            "model": "gpt-6-sol",
            "unit": "month_prescreen_then_month_select",
            "max_candidates_per_month_prompt": int(sec["max_candidates_per_month_prompt"]),
            "max_bucket_prompt_chars": int(sec["max_bucket_prompt_chars"]),
            "prescreen_batched": True,
            "prescreen_max_candidates_per_batch": int(sec["max_candidates_per_month_prompt"]),
            "prescreen_no_silent_truncate": True,
            "one_paid_eligibility_review_per_offered_candidate": True,
            "month_select_n": n_per,
            "month_select_offer": "full_include_pool_heat_ranked",
            "month_select_heat_rank_before_offer": True,
            "domain_balance": "soft_prompt_only",
            "prescreen_offer_cap_per_tag_month": int(sec.get("prescreen_candidates_per_tag_month", DEFAULT_TAG_MONTH_SAMPLE_CAP)),
            "prescreen_offer_sampling": "sha256(month|retrieval_tag_id|market_id)",
            "checks": [
                "true_category_domain",
                "news_grounded_real_world_event",
                "clear_resolution_criteria",
                "nonduplicate_underlying_event",
                "diversity_country_entity_event_type_news_structure",
                "prefer_contested_price_heat",
                "yes_no_balance_without_rigid_quotas",
            ],
            "underlying_default_max": 1,
            "underlying_max_2_requires_rationale": True,
            "underlying_cap_enforced_during_validation": True,
            "avoid_simple_volume_or_ranking_selection": True,
            "cross_bucket_dedupe": True,
            "month_pool_fail_closed_if_oversized": False,
            "month_pool_oversized_uses_batched_prescreen": True,
            "quota_fail_closed_on_shortfall": True,
        },
        "ledger_policy": (
            f"Write sample_ledger.csv only when every sample month has exactly {n_per} "
            f"selected markets (target_n={sample_target_n(cfg)}); domains are soft-balanced "
            "in the month-select prompt only. Preserve any page-cap status as a "
            "non-exhaustive sampling-frame limitation. Ledger is provisional until news "
            "evidence is validated per horizon."
        ),
        "events_list_order": EVENTS_LIST_ORDER,
        "events_list_ascending": EVENTS_LIST_ASCENDING,
    }


def default_out_dir(cfg: dict[str, Any], run_id: str) -> Path:
    return Path(cfg["paths"]["derived"]) / "sample_benchmark" / run_id


def _default_fetch_cutoff_history(
    client: httpx.Client,
    cfg: dict[str, Any],
    *,
    token_id: str,
    closed: datetime,
    hours: int,
) -> list[dict[str, Any]]:
    """Prefer exact short CLOB windows around each cutoff (same as formal freeze)."""
    max_lag = float(cfg["polymarket"]["max_price_lag_hours"])
    lookahead = float(cfg["polymarket"].get("price_fetch_lookahead_hours", 1))
    fidelity = int(cfg["polymarket"].get("price_fidelity", 60))
    clob = str(cfg["polymarket"]["clob_base_url"])
    cutoff = closed - timedelta(hours=int(hours))
    start, end = formal_price_window_bounds(
        cutoff, max_lag_hours=max_lag, lookahead_hours=lookahead
    )
    return fetch_price_history(
        client,
        clob,
        token_id,
        interval=None,
        fidelity=fidelity,
        start_ts=int(start.timestamp()),
        end_ts=int(end.timestamp()),
    )


def collect_qualified_pool(
    cfg: dict[str, Any],
    *,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = DEFAULT_MAX_PAGES_PER_TAG,
    dry_run: bool = False,
    out_dir: Path | None = None,
    list_pages: ListPagesFn | None = None,
    fetch_cutoff_history: FetchCutoffHistoryFn | None = None,
    expand_months: list[str] | None = None,
    base_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage 1: Gamma keyset scan → metadata gates → tag×month hash sample → CLOB prices.

    When ``expand_months`` is set, requires ``base_manifest``: only those months are
    re-scanned / re-sampled / re-priced (with price reuse for known market_ids);
    other months' ``qualified`` rows are kept verbatim so Stage-2 freezes for
    untouched months remain valid.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if not 1 <= page_limit <= MAX_KEYSET_LIMIT:
        raise ValueError(f"page_limit must be 1..{MAX_KEYSET_LIMIT}")
    expand_set: set[str] | None = None
    if expand_months is not None:
        if base_manifest is None:
            raise ValueError("expand_months requires base_manifest")
        expand_set = set(expand_months)
        if not expand_set:
            raise ValueError("expand_months must be non-empty")

    gamma = cfg["polymarket"]["gamma_base_url"]
    clob = cfg["polymarket"]["clob_base_url"]
    horizons = tuple(int(h) for h in cfg["experiment"]["horizon_hours"])
    max_lag = float(cfg["polymarket"]["max_price_lag_hours"])
    window_start, window_end = sample_window(cfg)
    tag_month_cap = int(
        sample_section(cfg).get(
            "prescreen_candidates_per_tag_month", DEFAULT_TAG_MONTH_SAMPLE_CAP
        )
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    out = ensure_dir(out_dir or default_out_dir(cfg, run_id))
    criteria = _criteria_block(cfg)
    criteria["max_pages_per_tag"] = max_pages
    criteria["page_limit"] = page_limit
    criteria["pagination"] = "gamma_events_keyset_cursor"

    if dry_run:
        manifest = {
            "run_id": run_id,
            "mode": "dry_run",
            "stage": "collect",
            "selection_version": SELECTION_VERSION,
            "criteria": criteria,
            "qualified": [],
            "n_qualified": 0,
            "rejection_counts": {},
            "tag_scan": [],
            "scan_complete": None,
            "note": "Dry-run: criteria only; no Gamma/CLOB calls.",
            "ledger_written": False,
        }
        write_json_atomic(out / "collect_manifest.json", manifest)
        write_json_atomic(out / "manifest.json", manifest)
        print(f"Dry-run sample-benchmark criteria written to {out / 'manifest.json'}")
        return manifest

    pages_fn = list_pages or (
        lambda client, tag_id, after_cursor: fetch_closed_events_keyset_page(
            client,
            gamma,
            tag_id=int(tag_id),
            limit=page_limit,
            after_cursor=after_cursor,
        )
    )
    history_fn = fetch_cutoff_history or (
        lambda client, token_id, closed, hours: _default_fetch_cutoff_history(
            client, cfg, token_id=token_id, closed=closed, hours=hours
        )
    )

    rejection_counts: Counter[str] = Counter()
    metadata_eligible: list[dict[str, Any]] = []
    seen_market_ids: set[str] = set()
    pages_fetched: list[dict[str, Any]] = []
    tag_scan: list[dict[str, Any]] = []
    n_price_reused = 0
    offered: list[dict[str, Any]] = []
    sampled_out_audit: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []

    with httpx.Client() as client:
        for tag in RETRIEVAL_TAGS:
            tag_id = int(tag["tag_id"])
            tag_label = str(tag["label"])
            pages_for_tag = 0
            hit_cursor_end = False
            capped = False
            n_events = 0
            n_markets = 0
            n_metadata_eligible_tag = 0
            n_pre_window_markets_seen = 0
            after_cursor: str | None = None

            for _page in range(max_pages):
                page_url = _events_keyset_url(
                    gamma,
                    tag_id=tag_id,
                    limit=page_limit,
                    after_cursor=after_cursor,
                )
                page_params = _events_keyset_params(
                    tag_id=tag_id, limit=page_limit, after_cursor=after_cursor
                )
                assert "offset" not in page_params
                page_data = pages_fn(client, tag_id, after_cursor)
                if not isinstance(page_data, dict):
                    raise RuntimeError(
                        "list_pages must return a keyset page dict "
                        "{'events': [...], 'next_cursor': ...}"
                    )
                events = page_data.get("events")
                if events is None:
                    events = []
                if not isinstance(events, list):
                    raise RuntimeError(
                        "list_pages keyset page 'events' must be a list"
                    )
                next_cursor = _normalize_cursor(page_data.get("next_cursor"))
                pages_for_tag += 1
                page_event_ids: list[str] = []
                page_market_ids: list[str] = []

                if not events:
                    hit_cursor_end = True
                    pages_fetched.append({
                        "tag_id": tag_id,
                        "tag_label": tag_label,
                        "after_cursor": after_cursor,
                        "next_cursor": next_cursor,
                        "limit": page_limit,
                        "url": page_url,
                        "params": page_params,
                        "pagination": "keyset",
                        "n_events": 0,
                        "event_ids": [],
                        "market_ids": [],
                    })
                    break

                flat: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    event_id = str(event.get("id") or "")
                    if event_id:
                        page_event_ids.append(event_id)
                    for market in _nested_markets(event):
                        mid = str(market.get("id") or "")
                        if mid:
                            page_market_ids.append(mid)
                        flat.append((event, market))
                        m_closed = parse_market_closed_time(market)
                        if m_closed is not None and m_closed < window_start:
                            n_pre_window_markets_seen += 1

                flat.sort(key=lambda pair: int(str(pair[1].get("id") or 0)))
                n_events += len(page_event_ids)
                n_markets += len(page_market_ids)
                pages_fetched.append({
                    "tag_id": tag_id,
                    "tag_label": tag_label,
                    "after_cursor": after_cursor,
                    "next_cursor": next_cursor,
                    "limit": page_limit,
                    "url": page_url,
                    "params": page_params,
                    "pagination": "keyset",
                    "n_events": len(page_event_ids),
                    "event_ids": page_event_ids,
                    "market_ids": page_market_ids,
                })

                for event, market in flat:
                    market_id = str(market.get("id") or "")
                    if not market_id:
                        rejection_counts["MISSING_MARKET_ID"] += 1
                        continue
                    if market_id in seen_market_ids:
                        rejection_counts["DUPLICATE_MARKET_ID_ACROSS_TAGS"] += 1
                        continue
                    # Prefer market.closedTime; never treat event.closedTime as substitute.
                    reason = screen_sample_metadata(market, cfg)
                    if reason:
                        rejection_counts[reason] += 1
                        continue
                    closed = parse_market_closed_time(market)
                    assert closed is not None
                    token_id = _yes_token_id(market)
                    assert token_id
                    month = month_from_closed_time(
                        closed, window_start=window_start, window_end=window_end
                    )
                    assert month is not None
                    if expand_set is not None and month not in expand_set:
                        continue
                    start = parse_market_active_start(market)
                    seen_market_ids.add(market_id)
                    metadata_eligible.append({
                        "market_id": market_id,
                        "gamma_event_id": str(event.get("id") or ""),
                        "market_slug": str(market.get("slug") or ""),
                        "condition_id": str(
                            market.get("conditionId") or market.get("condition_id") or ""
                        ),
                        "question": str(market.get("question") or ""),
                        "description": str(market.get("description") or ""),
                        "category_field": str(market.get("category") or ""),
                        "closed_time_utc": closed.isoformat(),
                        "start_time_utc": start.isoformat() if start else None,
                        "sample_month": month,
                        "outcome": _parse_outcome(market),
                        "yes_token_id": token_id,
                        "retrieval_tag_id": tag_id,
                        "retrieval_tag_label": tag_label,
                        "source_page_url": page_url,
                        "source_page_params": page_params,
                        "source_after_cursor": after_cursor,
                        "source_next_cursor": next_cursor,
                        "source_pagination": "keyset",
                        "gamma_event_url": (
                            f"{gamma.rstrip('/')}/events/{event.get('id')}"
                            if event.get("id")
                            else None
                        ),
                        "gamma_market_url": f"{gamma.rstrip('/')}/markets/{market_id}",
                        "clock_definition": CLOCK_DEFINITION,
                        # event.closedTime recorded for audit only; not used for gates.
                        "event_closed_time_raw": event.get("closedTime"),
                        "market_closed_time_raw": market.get("closedTime"),
                        "_closed_dt": closed,
                    })
                    n_metadata_eligible_tag += 1

                # Keyset end: absent next_cursor (empty events handled above).
                # One pre-window nested market must NOT imply later pages lack
                # in-window children (events can mix old and new markets).
                if next_cursor is None:
                    hit_cursor_end = True
                    break
                after_cursor = next_cursor
            else:
                # for-page exhausted without break → hit max_pages with more cursors
                capped = True

            # Complete only when cursor ended; capped = incomplete.
            scan_complete = bool(hit_cursor_end) and not capped
            tag_scan.append({
                "tag_id": tag_id,
                "tag_label": tag_label,
                "pages_fetched": pages_for_tag,
                "max_pages": max_pages,
                "n_events_seen": n_events,
                "n_markets_seen": n_markets,
                "n_metadata_eligible": n_metadata_eligible_tag,
                "n_qualified": 0,  # filled after price gate
                "n_pre_window_markets_seen": n_pre_window_markets_seen,
                "hit_empty_or_absent_next_cursor": hit_cursor_end,
                "capped_by_max_pages": capped,
                "scan_complete": scan_complete,
                "pagination": "keyset",
            })

        # Hash-sample before CLOB: at most tag_month_cap per month×retrieval_tag_id.
        offered, sampled_out_rows = sample_tag_month_pool(
            metadata_eligible, cap_per_tag_month=tag_month_cap
        )
        sampled_out_audit = [
            {
                "market_id": str(r["market_id"]),
                "sample_month": str(r.get("sample_month") or ""),
                "retrieval_tag_id": r.get("retrieval_tag_id"),
                "retrieval_tag_label": r.get("retrieval_tag_label"),
                "reason": TAG_MONTH_SAMPLING_CAP_REASON,
            }
            for r in sampled_out_rows
        ]
        if sampled_out_audit:
            rejection_counts[TAG_MONTH_SAMPLING_CAP_REASON] += len(sampled_out_audit)

        qualified: list[dict[str, Any]] = []
        base_by_id = {
            str(r["market_id"]): r
            for r in (base_manifest or {}).get("qualified") or []
            if r.get("market_id")
        }
        n_price_reused = 0
        for cand in offered:
            closed = cand["_closed_dt"]
            assert isinstance(closed, datetime)
            mid = str(cand["market_id"])
            prior = base_by_id.get(mid)
            if (
                prior
                and isinstance(prior.get("price_at_cutoffs"), dict)
                and prior.get("price_at_cutoffs")
            ):
                row = {k: v for k, v in cand.items() if k != "_closed_dt"}
                row["price_at_cutoffs"] = prior["price_at_cutoffs"]
                row["price_fetch_meta"] = {
                    **(prior.get("price_fetch_meta") or {}),
                    "reused_from_base_collect": True,
                }
                qualified.append(row)
                n_price_reused += 1
                continue
            token_id = str(cand["yes_token_id"])
            history_by_horizon: dict[int, list[dict[str, Any]]] = {}
            fetch_meta: dict[str, Any] = {}
            fetch_failed = False
            for hours in horizons:
                try:
                    hist = history_fn(client, token_id, closed, int(hours))
                except httpx.HTTPError as exc:
                    rejection_counts["PRICE_HISTORY_FETCH_FAILED"] += 1
                    fetch_meta[f"tminus{hours}h"] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }
                    fetch_failed = True
                    break
                history_by_horizon[int(hours)] = hist
                fetch_meta[f"tminus{hours}h"] = {"n_points": len(hist)}
            if fetch_failed:
                continue
            ok, price_reason, price_detail = price_ok_at_all_cutoffs(
                history_by_horizon,
                closed=closed,
                horizons=horizons,
                max_lag_hours=max_lag,
            )
            if not ok:
                rejection_counts[price_reason or "PRICE_AT_CUTOFF_FAILED"] += 1
                continue
            row = {k: v for k, v in cand.items() if k != "_closed_dt"}
            row["price_at_cutoffs"] = price_detail
            row["price_fetch_meta"] = fetch_meta
            qualified.append(row)

    if expand_set is not None and base_manifest is not None:
        kept_qualified = [
            r
            for r in (base_manifest.get("qualified") or [])
            if str(r.get("sample_month") or "") not in expand_set
        ]
        qualified = kept_qualified + qualified
        kept_sampled_out = [
            e
            for e in (base_manifest.get("sampled_out") or [])
            if str(e.get("sample_month") or "") not in expand_set
        ]
        sampled_out_audit = kept_sampled_out + sampled_out_audit

    n_qualified_by_tag = Counter(int(r["retrieval_tag_id"]) for r in qualified)
    for t in tag_scan:
        t["n_qualified"] = int(n_qualified_by_tag.get(int(t["tag_id"]), 0))

    overall_complete = all(t["scan_complete"] for t in tag_scan)
    incomplete_tags = [t["tag_label"] for t in tag_scan if not t["scan_complete"]]

    manifest = {
        "run_id": run_id,
        "mode": "collect",
        "stage": "collect",
        "selection_version": SELECTION_VERSION,
        "criteria": criteria,
        "qualified": qualified,
        "n_qualified": len(qualified),
        "n_metadata_eligible": len(metadata_eligible),
        "n_sampled_for_price": len(offered),
        "sampled_out": sampled_out_audit,
        "n_sampled_out": len(sampled_out_audit),
        "tag_month_sample_cap": tag_month_cap,
        "expand_months": sorted(expand_set) if expand_set else None,
        "n_price_reused_from_base": n_price_reused if expand_set else 0,
        "base_collect_run_id": (
            (base_manifest or {}).get("run_id") if expand_set else None
        ),
        "qualified_by_month": dict(sorted(Counter(r["sample_month"] for r in qualified).items())),
        "qualified_by_retrieval_tag": dict(sorted(Counter(r["retrieval_tag_label"] for r in qualified).items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "pages_fetched": pages_fetched,
        "tag_scan": tag_scan,
        "scan_complete": overall_complete,
        "incomplete_tags": incomplete_tags,
        "scan_completeness_note": (
            None
            if overall_complete
            else (
                "One or more retrieval-tag keyset scans stopped at max_pages before "
                "an empty page or absent next_cursor. Do not treat this pool as a "
                "complete in-window scan. A quota sample from this pool is "
                "non-exhaustive and may miss earlier in-window months."
            )
        ),
        "source": {
            "gamma_events_keyset": (
                f"{gamma.rstrip('/')}{EVENTS_KEYSET_PATH}"
                f"?closed=true&tag_id={{tag_id}}"
                f"&order={EVENTS_LIST_ORDER}&ascending={EVENTS_LIST_ASCENDING}"
                f"&limit={{limit}}&after_cursor={{after_cursor}}"
            ),
            "gamma_events_keyset_params": {
                "closed": "true",
                "tag_id": "{tag_id}",
                "limit": "{limit}",
                "order": EVENTS_LIST_ORDER,
                "ascending": EVENTS_LIST_ASCENDING,
                "after_cursor": "{after_cursor_when_present}",
                "offset": None,
                "note": "offset must never be passed to /events/keyset",
            },
            "pagination": "keyset",
            "gamma_tags_verified": [
                {
                    "tag_id": t["tag_id"],
                    "label": t["label"],
                    "url": f"{gamma.rstrip('/')}{t['verified_via_path']}",
                }
                for t in RETRIEVAL_TAGS
            ],
            "stale_tag_ids_not_used": [STALE_ENTERTAINMENT_TAG_ID],
            "clob_prices_history": f"{clob.rstrip('/')}/prices-history",
            "price_fetch_style": "short_window_startTs_endTs_per_cutoff",
            "price_fetch_after_tag_month_sample": True,
            "max_pages_per_tag": max_pages,
            "page_limit": page_limit,
            "page_limit_max": MAX_KEYSET_LIMIT,
            "pagination_stop_policy": (
                "empty_events_or_absent_next_cursor_or_max_pages_cap;"
                "no_offset;no_oldest_nested_market_pre_window_early_exit"
            ),
        },
        "ledger_written": False,
        "out_dir": str(out),
    }
    write_json_atomic(out / "collect_manifest.json", manifest)
    write_json_atomic(out / "qualified_pool.json", qualified)
    write_json_atomic(out / "manifest.json", manifest)
    print(
        f"Sample collect: n_qualified={len(qualified)} scan_complete={overall_complete} "
        f"→ {out / 'collect_manifest.json'}"
    )
    if expand_set:
        print(
            f"Expanded months {sorted(expand_set)} "
            f"(price_reused={n_price_reused}); other months kept from base collect."
        )
    if incomplete_tags:
        print(
            f"Incomplete tag scans (bounded candidate frame): {incomplete_tags}"
        )
    return manifest


def evaluate_month_fill(
    selected_by_month: dict[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
    *,
    n_per: int | None = None,
) -> dict[str, Any]:
    """Complete only when every sample month has exactly ``per_month_n`` rows."""
    from collections import Counter

    per = int(n_per) if n_per is not None else per_month_n(cfg)
    months = sample_months(cfg)
    fills: dict[str, int] = {}
    shortfalls: dict[str, int] = {}
    overs: dict[str, int] = {}
    by_domain: Counter[str] = Counter()
    for month in months:
        rows = list(selected_by_month.get(month, []))
        n = len(rows)
        fills[month] = n
        if n < per:
            shortfalls[month] = per - n
        elif n > per:
            overs[month] = n - per
        for row in rows:
            dom = str(row.get("domain") or row.get("domain_confirmed") or "")
            if dom:
                by_domain[dom] += 1
    target = per * len(months)
    complete = not shortfalls and not overs and sum(fills.values()) == target
    return {
        "per_month_n": per,
        "n_months": len(months),
        "fills_by_month": fills,
        "shortfalls": shortfalls,
        "overs": overs,
        "by_domain": dict(sorted(by_domain.items())),
        "n_selected_total": sum(fills.values()),
        "target_n": target,
        "quota_complete": complete,
        "domain_quota": "soft_prompt_only",
    }


def write_sample_ledger(
    path: Path,
    selected: list[dict[str, Any]],
) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "event_id",
        "market_id",
        "market_slug",
        "condition_id",
        "sample_month",
        "domain",
        "bucket",
        "outcome",
        "closed_time_utc",
        "question",
        "selection_reason",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(selected, start=1):
            writer.writerow({
                "event_id": row.get("event_id") or f"sample_{SELECTION_VERSION}_{i:03d}",
                "market_id": row.get("market_id", ""),
                "market_slug": row.get("market_slug", ""),
                "condition_id": row.get("condition_id", ""),
                "sample_month": row.get("sample_month", ""),
                "domain": row.get("domain", ""),
                "bucket": row.get("bucket")
                or bucket_key(str(row.get("sample_month", "")), str(row.get("domain", ""))),
                "outcome": row.get("outcome", ""),
                "closed_time_utc": row.get("closed_time_utc", ""),
                "question": row.get("question", ""),
                "selection_reason": row.get("selection_reason", ""),
                "notes": row.get("notes", ""),
            })


def load_collect_manifest(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid collect manifest: {path}")
    return data
