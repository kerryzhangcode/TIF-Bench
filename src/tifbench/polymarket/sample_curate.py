"""Stage 2: GPT-6 Sol semantic curation into a monthly cohort.

Flow:
  1) Month-level batched semantic pre-screen (eligibility + domain assignment).
  2) Per-month selection of exactly ``per_month_n`` markets from the full
     heat-ranked INCLUDE pool (no offer truncate). Domains are soft-balanced
     in the prompt only.

Injectable ``call_model`` keeps tests offline. Live OpenAI is only used when the
CLI runs ``--curate`` without an injected callable (not exercised in unit tests).
"""

from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from tifbench.config import require_api_key
from tifbench.io import ensure_dir, project_root, read_json, write_json_atomic
from tifbench.openai_client import (
    build_responses_body,
    extract_output_text,
    make_client,
    responses_create,
)
from tifbench.polymarket.sample_benchmark import (
    SELECTION_VERSION,
    TAG_MONTH_SAMPLING_CAP_REASON,
    bucket_key,
    collect_qualified_pool,
    default_out_dir,
    evaluate_month_fill,
    parse_sample_months,
    per_month_n,
    sample_domains,
    sample_months,
    sample_section,
    sample_tag_month_pool,
    sample_target_n,
    write_sample_ledger,
)

CallModelFn = Callable[[str, dict[str, Any]], dict[str, Any]]

# Second same-underlying pick requires a concrete distinct-target rationale.
MIN_PAIR_RATIONALE_CHARS = 40

# Explicit hard caps recorded in manifests (no silent truncate).
DEFAULT_MAX_CANDIDATES_PER_BATCH = 40
DEFAULT_MAX_PROMPT_CHARS = 160_000
DEFAULT_PRESCREEN_TAG_MONTH_CAP = 50
PRESCREEN_SAMPLING_CAP_REASON = TAG_MONTH_SAMPLING_CAP_REASON

CURATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "market_id": {"type": "string"},
                    "selection_reason": {"type": "string"},
                    "domain_confirmed": {"type": "string"},
                    "underlying_event_key": {"type": "string"},
                    "country_or_region": {"type": "string"},
                    "entity": {"type": "string"},
                    "event_type": {"type": "string"},
                    "news_structure": {"type": "string"},
                    "pair_allowed": {"type": "boolean"},
                    "pair_rationale": {"type": "string"},
                },
                "required": [
                    "market_id",
                    "selection_reason",
                    "domain_confirmed",
                    "underlying_event_key",
                    "country_or_region",
                    "entity",
                    "event_type",
                    "news_structure",
                    "pair_allowed",
                    "pair_rationale",
                ],
            },
        },
        "excluded": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "market_id": {"type": "string"},
                    "exclusion_reason": {"type": "string"},
                },
                "required": ["market_id", "exclusion_reason"],
            },
        },
        "bucket_notes": {"type": "string"},
    },
    "required": ["selected", "excluded", "bucket_notes"],
}

PRESCREEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "market_id": {"type": "string"},
                    "decision": {"type": "string"},
                    "domain_confirmed": {"type": "string"},
                    "exclusion_reason": {"type": "string"},
                },
                "required": [
                    "market_id",
                    "decision",
                    "domain_confirmed",
                    "exclusion_reason",
                ],
            },
        },
        "batch_notes": {"type": "string"},
    },
    "required": ["decisions", "batch_notes"],
}

MONTH_SELECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected": CURATION_SCHEMA["properties"]["selected"],
        "excluded": CURATION_SCHEMA["properties"]["excluded"],
        "month_notes": {"type": "string"},
    },
    "required": ["selected", "excluded", "month_notes"],
}


def make_sample_event_id(market_id: str) -> str:
    """Stable selected identity: includes real market_id (not a bucket ordinal)."""
    mid = str(market_id).strip()
    if not mid:
        raise ValueError("market_id required for sample event_id")
    return f"sample_{SELECTION_VERSION}_m{mid}"


def load_curation_prompt() -> str:
    path = project_root() / "prompts" / "sample_benchmark_curation.md"
    return path.read_text(encoding="utf-8")


def load_prescreen_prompt() -> str:
    path = project_root() / "prompts" / "sample_benchmark_prescreen.md"
    return path.read_text(encoding="utf-8")


def stage2_caps(cfg: dict[str, Any]) -> tuple[int, int]:
    sec = sample_section(cfg)
    return (
        int(sec.get("max_candidates_per_month_prompt", DEFAULT_MAX_CANDIDATES_PER_BATCH)),
        int(sec.get("max_bucket_prompt_chars", DEFAULT_MAX_PROMPT_CHARS)),
    )


def _format_candidate_lines(candidates: list[dict[str, Any]]) -> str:
    """Format candidates with full resolution rules (no silent truncate).

    Prompt size is enforced by batch planning / char caps; oversized single
    candidates fail closed rather than clipping resolution text.
    """
    lines: list[str] = []
    for c in candidates:
        heat_mid, heat_move, price_bits = candidate_heat_features(c)
        lines.append(
            f"- market_id={c.get('market_id')}\n"
            f"  closed_time_utc={c.get('closed_time_utc')}\n"
            f"  outcome={c.get('outcome')} (1=YES,0=NO)\n"
            f"  retrieval_tag={c.get('retrieval_tag_label')}({c.get('retrieval_tag_id')})\n"
            f"  category={c.get('category_field')}\n"
            f"  domain_hint={c.get('domain_confirmed') or c.get('domain') or ''}\n"
            f"  heat_mid={heat_mid:.4f} heat_move={heat_move:.4f} {price_bits}\n"
            f"  slug={c.get('market_slug')}\n"
            f"  question={c.get('question')}\n"
            f"  rules={str(c.get('description') or '')}"
        )
    return "\n".join(lines) if lines else "(empty pool)"


def candidate_heat_features(row: dict[str, Any]) -> tuple[float, float, str]:
    """Price-based heat from Stage-1 ``price_at_cutoffs`` (no volume).

    ``heat_mid`` is higher when the *mean* |p − 0.5| across usable horizons is
    smaller (prices stay nearer the middle on average).
    ``heat_move`` is max−min probability across those horizons.
    """
    pac = row.get("price_at_cutoffs") or {}
    prices: list[float] = []
    bits: list[str] = []
    for key, cell in sorted(pac.items()) if isinstance(pac, dict) else []:
        if not (key.startswith("tminus") and key.endswith("h")):
            continue
        if not isinstance(cell, dict) or not cell.get("ok"):
            continue
        try:
            p = float(cell.get("probability"))
        except (TypeError, ValueError):
            continue
        prices.append(p)
        bits.append(f"p_{key}={p:.4f}")
    if not prices:
        return 0.0, 0.0, "prices=missing"
    mean_abs = sum(abs(p - 0.5) for p in prices) / len(prices)
    heat_mid = 1.0 - mean_abs
    heat_move = max(prices) - min(prices)
    return float(heat_mid), float(heat_move), " ".join(bits)


def heat_rank_month_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stable heat-rank of the full INCLUDE pool (no offer truncate)."""
    decorated: list[tuple[float, float, str, dict[str, Any]]] = []
    for row in candidates:
        heat_mid, heat_move, _ = candidate_heat_features(row)
        decorated.append((heat_mid, heat_move, str(row.get("market_id") or ""), row))
    decorated.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [t[3] for t in decorated]


def build_month_select_prompt(
    *,
    month: str,
    target_n: int,
    candidates: list[dict[str, Any]],
    domain_labels: list[str],
    prompt_template: str | None = None,
) -> str:
    template = prompt_template or load_curation_prompt()
    return template.format(
        month=month,
        target_n=target_n,
        n_candidates=len(candidates),
        domain_labels="\n".join(f"- {d}" for d in domain_labels),
        candidate_pool=_format_candidate_lines(candidates),
    )


def sample_prescreen_pool(
    qualified: list[dict[str, Any]], *, cap_per_tag_month: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Offer a reproducible, auditable sample per month×retrieval tag (Stage 2 reuse).

    Stage 1 already applies the same cap before CLOB; this remains for curate-only
    runs against older collect manifests. Tags are discovery hints, not final domains.
    """
    return sample_tag_month_pool(qualified, cap_per_tag_month=cap_per_tag_month)


def build_prescreen_prompt(
    *,
    month: str,
    candidates: list[dict[str, Any]],
    domain_labels: list[str],
    batch_index: int,
    n_batches: int,
    prompt_template: str | None = None,
) -> str:
    template = prompt_template or load_prescreen_prompt()
    return template.format(
        month=month,
        n_candidates=len(candidates),
        candidate_pool=_format_candidate_lines(candidates),
        domain_labels=", ".join(domain_labels),
        batch_index=batch_index,
        n_batches=n_batches,
    )


def substantial_pair_rationale(text: str) -> bool:
    """Require a concrete distinct-target rationale for a second same-underlying pick."""
    return len(str(text or "").strip()) >= MIN_PAIR_RATIONALE_CHARS


def _default_openai_call(cfg: dict[str, Any]) -> CallModelFn:
    api_key = require_api_key(str(cfg["models"]["gpt_sol"]["api_key_env"]))
    client = make_client(api_key)
    model = str(cfg["models"]["gpt_sol"]["model"])
    effort = str(cfg["models"]["gpt_sol"].get("reasoning", {}).get("effort", "medium"))

    def _call(prompt: str, meta: dict[str, Any]) -> dict[str, Any]:
        schema = meta.get("json_schema") or MONTH_SELECT_SCHEMA
        schema_name = str(meta.get("schema_name") or "sample_benchmark_month_select")
        body = build_responses_body(
            model=model,
            input_text=prompt,
            reasoning_effort=effort,
            json_schema=schema,
            schema_name=schema_name,
        )
        response = responses_create(client, body)
        text = extract_output_text(response)
        return json.loads(text)

    return _call


def _pool_by_month(
    qualified: list[dict[str, Any]], months: list[str]
) -> dict[str, list[dict[str, Any]]]:
    allowed = set(months)
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qualified:
        month = str(row.get("sample_month") or "")
        if month in allowed:
            by_month[month].append(row)
    return by_month


def clear_month_curate_freezes(out: Path, months: list[str]) -> list[str]:
    """Delete prescreen/month freeze files for the given months (audited overwrite)."""
    removed: list[str] = []
    for month in months:
        key = month.replace("-", "")
        pre_dir = out / "prescreen_calls"
        if pre_dir.is_dir():
            for path in sorted(pre_dir.glob(f"{key}_batch*.json")):
                path.unlink()
                removed.append(str(path))
        month_path = out / "month_calls" / f"{key}.json"
        if month_path.exists():
            month_path.unlink()
            removed.append(str(month_path))
    return removed


def _load_prior_curate_manifest(out: Path) -> dict[str, Any] | None:
    path = out / "curate_manifest.json"
    if not path.exists():
        return None
    raw = read_json(path)
    return raw if isinstance(raw, dict) else None


def _prior_selected_by_month(
    prior: dict[str, Any] | None, months: list[str]
) -> dict[str, list[dict[str, Any]]]:
    by_month: dict[str, list[dict[str, Any]]] = {m: [] for m in months}
    if not prior:
        return by_month
    for row in prior.get("selected") or []:
        if not isinstance(row, dict):
            continue
        month = str(row.get("sample_month") or "")
        if month in by_month:
            by_month[month].append(row)
    return by_month


def _stable_sorted_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda c: str(c.get("market_id") or ""))


def plan_prescreen_batches(
    candidates: list[dict[str, Any]],
    *,
    month: str,
    max_candidates: int,
    max_chars: int,
    domain_labels: list[str],
    prompt_template: str | None = None,
) -> tuple[list[list[dict[str, Any]]], str | None]:
    """Split a month pool into auditable batches that fit prompt caps.

    Never silently drops candidates. If even a single candidate cannot fit,
    returns an empty plan with an explicit fail-closed reason.
    """
    ordered = _stable_sorted_candidates(candidates)
    if not ordered:
        return [], None
    if max_candidates < 1:
        return [], "prescreen_invalid_max_candidates:max<1;fail_closed_no_silent_truncate"

    batches: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(ordered):
        # Prefer full max_candidates slices; shrink only when prompt chars overflow.
        size = min(max_candidates, len(ordered) - i)
        while size >= 1:
            chunk = ordered[i : i + size]
            # n_batches unknown until finished; use placeholder upper bound for sizing.
            approx_batches = max(1, (len(ordered) + max_candidates - 1) // max_candidates)
            prompt = build_prescreen_prompt(
                month=month,
                candidates=chunk,
                domain_labels=domain_labels,
                batch_index=len(batches) + 1,
                n_batches=approx_batches,
                prompt_template=prompt_template,
            )
            if len(prompt) <= max_chars:
                batches.append(chunk)
                i += size
                break
            size //= 2
        else:
            return [], (
                f"prescreen_single_candidate_prompt_too_large:"
                f"market_id={ordered[i].get('market_id')};"
                f"max_chars={max_chars};fail_closed_no_silent_truncate"
            )

    # Rebuild prompts with accurate n_batches for audit consistency on freeze.
    return batches, None


def _validate_prescreen_response(
    raw: dict[str, Any],
    *,
    pool_ids: set[str],
    domain_labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Normalize pre-screen output; require a disposition for every pool id."""
    issues: list[str] = []
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_domains = set(domain_labels)

    for item in raw.get("decisions") or []:
        if not isinstance(item, dict):
            issues.append("non_object_prescreen_row")
            continue
        mid = str(item.get("market_id") or "")
        if not mid or mid not in pool_ids:
            issues.append(f"prescreen_not_in_pool:{mid}")
            continue
        if mid in seen:
            issues.append(f"prescreen_duplicate:{mid}")
            continue
        seen.add(mid)
        decision = str(item.get("decision") or "").strip().upper()
        domain_confirmed = str(item.get("domain_confirmed") or "").strip()
        exclusion_reason = str(item.get("exclusion_reason") or "").strip()
        if decision == "INCLUDE":
            if domain_confirmed not in allowed_domains:
                issues.append(f"prescreen_bad_domain:{mid}:{domain_confirmed}")
                continue
            included.append({
                "market_id": mid,
                "decision": "INCLUDE",
                "domain_confirmed": domain_confirmed,
                "exclusion_reason": "",
            })
        elif decision == "EXCLUDE":
            if not exclusion_reason:
                issues.append(f"prescreen_missing_exclusion_reason:{mid}")
                continue
            excluded.append({
                "market_id": mid,
                "decision": "EXCLUDE",
                "domain_confirmed": "",
                "exclusion_reason": exclusion_reason,
            })
        else:
            issues.append(f"prescreen_bad_decision:{mid}:{decision}")

    missing = pool_ids - seen
    if missing:
        issues.append(f"unscreened_candidates:{','.join(sorted(missing))}")
    return included, excluded, issues


def _validate_month_response(
    raw: dict[str, Any],
    *,
    month: str,
    pool_ids: set[str],
    domain_labels: list[str],
    target_n: int,
    globally_used: set[str],
    underlying_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Validate month-select output; domain_confirmed must be one of protocol labels."""
    issues: list[str] = []
    selected_out: list[dict[str, Any]] = []
    excluded_out: list[dict[str, Any]] = []
    allowed_domains = set(domain_labels)

    for ex in raw.get("excluded") or []:
        if not isinstance(ex, dict):
            continue
        mid = str(ex.get("market_id") or "")
        if mid not in pool_ids:
            issues.append(f"excluded_not_in_pool:{mid}")
            continue
        excluded_out.append({
            "market_id": mid,
            "exclusion_reason": str(ex.get("exclusion_reason") or "UNSPECIFIED"),
        })

    seen_local: set[str] = set()
    local_underlying: dict[str, int] = defaultdict(int)
    for item in raw.get("selected") or []:
        if not isinstance(item, dict):
            issues.append("non_object_selected_row")
            continue
        mid = str(item.get("market_id") or "")
        if not mid or mid not in pool_ids:
            issues.append(f"selected_not_in_pool:{mid}")
            continue
        if mid in globally_used or mid in seen_local:
            issues.append(f"cross_or_local_duplicate:{mid}")
            continue
        domain_confirmed = str(item.get("domain_confirmed") or "").strip()
        if domain_confirmed not in allowed_domains:
            issues.append(f"domain_mismatch:{mid}:{domain_confirmed}")
            continue
        ukey = str(item.get("underlying_event_key") or "").strip() or f"unknown:{mid}"
        pair_allowed = bool(item.get("pair_allowed"))
        pair_rationale = str(item.get("pair_rationale") or "").strip()
        prior = int(underlying_counts.get(ukey, 0)) + int(local_underlying[ukey])
        if prior >= 2:
            issues.append(f"underlying_over_cap:{ukey}:{mid}")
            continue
        if prior >= 1:
            if not (pair_allowed and substantial_pair_rationale(pair_rationale)):
                issues.append(f"underlying_cap:{ukey}:{mid}")
                continue
        row = {
            "market_id": mid,
            "sample_month": month,
            "domain": domain_confirmed,
            "bucket": bucket_key(month, domain_confirmed),
            "selection_reason": str(item.get("selection_reason") or ""),
            "domain_confirmed": domain_confirmed,
            "underlying_event_key": ukey,
            "country_or_region": str(item.get("country_or_region") or ""),
            "entity": str(item.get("entity") or ""),
            "event_type": str(item.get("event_type") or ""),
            "news_structure": str(item.get("news_structure") or ""),
            "pair_allowed": pair_allowed,
            "pair_rationale": pair_rationale,
        }
        selected_out.append(row)
        seen_local.add(mid)
        local_underlying[ukey] += 1
        if len(selected_out) >= target_n:
            break

    if len(selected_out) < target_n:
        issues.append(f"shortfall_after_validation:{len(selected_out)}<{target_n}")
    if len(selected_out) > target_n:
        issues.append(f"over_after_validation:{len(selected_out)}>{target_n}")
        selected_out = selected_out[:target_n]
    selected_ids = {row["market_id"] for row in selected_out}
    excluded_ids = {row["market_id"] for row in excluded_out}
    overlap = selected_ids & excluded_ids
    if overlap:
        issues.append(f"selected_and_excluded:{','.join(sorted(overlap))}")
    missing = pool_ids - selected_ids - excluded_ids
    if missing:
        issues.append(f"unscreened_candidates:{','.join(sorted(missing))}")
    return selected_out, excluded_out, issues


def _freeze_or_call(
    *,
    call_path: Path,
    prompt: str,
    prompt_sha256: str,
    template_sha256: str,
    candidate_ids: list[str],
    model_fn: CallModelFn,
    meta: dict[str, Any],
    cfg: dict[str, Any],
    record_extra: dict[str, Any],
) -> dict[str, Any]:
    model = str(cfg["models"]["gpt_sol"]["model"])
    effort = str(
        cfg["models"]["gpt_sol"].get("reasoning", {}).get("effort", "medium")
    )
    stage = str(meta.get("stage") or record_extra.get("stage") or "")
    schema_name = str(meta.get("schema_name") or "")
    if call_path.exists():
        frozen_call = read_json(call_path)
        # Fail closed on mismatch *or* missing fields (no silent reuse of old caches).
        checks = {
            "prompt_sha256": frozen_call.get("prompt_sha256") == prompt_sha256,
            "candidate_market_ids": (
                frozen_call.get("candidate_market_ids") == candidate_ids
            ),
            "model": "model" in frozen_call and frozen_call.get("model") == model,
            "reasoning_effort": (
                "reasoning_effort" in frozen_call
                and frozen_call.get("reasoning_effort") == effort
            ),
            "prompt_template_sha256": (
                "prompt_template_sha256" in frozen_call
                and frozen_call.get("prompt_template_sha256") == template_sha256
            ),
            "stage": "stage" in frozen_call and frozen_call.get("stage") == stage,
            "schema_name": (
                "schema_name" in frozen_call
                and frozen_call.get("schema_name") == schema_name
            ),
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise RuntimeError(
                f"Frozen call differs ({','.join(bad)}): {call_path}"
            )
        raw = frozen_call.get("response")
        if not isinstance(raw, dict):
            raise RuntimeError(f"Frozen call missing response dict: {call_path}")
        return raw
    raw = model_fn(prompt, meta)
    if not isinstance(raw, dict):
        raw = {}
    write_json_atomic(call_path, {
        "model": model,
        "reasoning_effort": effort,
        "prompt_template_sha256": template_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt": prompt,
        "candidate_market_ids": candidate_ids,
        "response": raw,
        **record_extra,
        # Ensure required reuse keys are always present (fail-closed on old caches).
        "stage": stage or str(record_extra.get("stage") or ""),
        "schema_name": schema_name,
    })
    return raw


def _run_month_prescreen(
    *,
    month: str,
    month_pool: list[dict[str, Any]],
    domains: list[str],
    max_cand: int,
    max_chars: int,
    out: Path,
    cfg: dict[str, Any],
    model_fn: CallModelFn,
    template: str,
    template_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """Return (eligible_by_domain, all_excluded_rows, month_report)."""
    batches, plan_fail = plan_prescreen_batches(
        month_pool,
        month=month,
        max_candidates=max_cand,
        max_chars=max_chars,
        domain_labels=domains,
        prompt_template=template,
    )
    report: dict[str, Any] = {
        "month": month,
        "n_candidates": len(month_pool),
        "max_candidates_per_prescreen_batch": max_cand,
        "max_prescreen_prompt_chars": max_chars,
        "n_batches_planned": len(batches),
        "batch_reports": [],
        "fail_closed_reason": plan_fail,
        "n_included": 0,
        "n_excluded": 0,
    }
    eligible_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_excluded: list[dict[str, Any]] = []
    if plan_fail:
        return eligible_by_domain, all_excluded, report

    n_batches = len(batches)
    for bi, chunk in enumerate(batches, start=1):
        prompt = build_prescreen_prompt(
            month=month,
            candidates=chunk,
            domain_labels=domains,
            batch_index=bi,
            n_batches=n_batches,
            prompt_template=template,
        )
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        candidate_ids = [str(c["market_id"]) for c in chunk]
        call_path = (
            ensure_dir(out / "prescreen_calls")
            / f"{month.replace('-', '')}_batch{bi:02d}.json"
        )
        meta = {
            "stage": "prescreen",
            "month": month,
            "batch_index": bi,
            "n_batches": n_batches,
            "n_candidates": len(chunk),
            "candidate_market_ids": candidate_ids,
            "domain_labels": domains,
            "json_schema": PRESCREEN_SCHEMA,
            "schema_name": "sample_benchmark_prescreen",
        }
        raw = _freeze_or_call(
            call_path=call_path,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            template_sha256=template_sha256,
            candidate_ids=candidate_ids,
            model_fn=model_fn,
            meta=meta,
            cfg=cfg,
            record_extra={
                "stage": "prescreen",
                "month": month,
                "batch_index": bi,
                "n_batches": n_batches,
            },
        )
        pool_ids = set(candidate_ids)
        included, excluded, issues = _validate_prescreen_response(
            raw, pool_ids=pool_ids, domain_labels=domains,
        )
        fail_closed = False
        if any(
            i.startswith((
                "unscreened_candidates:",
                "prescreen_not_in_pool:",
                "prescreen_duplicate:",
                "prescreen_bad_decision:",
                "prescreen_bad_domain:",
                "prescreen_missing_exclusion_reason:",
            ))
            for i in issues
        ):
            fail_closed = True
            issues.append("fail_closed_incomplete_agent_prescreen")
            included = []
            # Keep excluded empty on fail-closed so we do not silently accept partial.
            excluded = []

        pool_index = {str(c["market_id"]): c for c in chunk}
        if not fail_closed:
            for row in included:
                base = pool_index.get(row["market_id"], {})
                eligible_by_domain[row["domain_confirmed"]].append({**base, **row})
            for row in excluded:
                all_excluded.append({**row, "month": month, "prescreen_batch": bi})

        batch_report = {
            "month": month,
            "batch_index": bi,
            "n_batches": n_batches,
            "n_candidates": len(chunk),
            "candidate_market_ids": candidate_ids,
            "n_included": len(included) if not fail_closed else 0,
            "n_excluded": len(excluded) if not fail_closed else 0,
            "validation_issues": issues,
            "fail_closed": fail_closed,
            "prescreen_call_record": str(call_path),
            "prompt_sha256": prompt_sha256,
            "batch_notes": str(raw.get("batch_notes") or ""),
            "included": included if not fail_closed else [],
            "excluded": excluded if not fail_closed else [],
        }
        report["batch_reports"].append(batch_report)
        if fail_closed:
            report["fail_closed_reason"] = "fail_closed_incomplete_agent_prescreen"
            # Stop further batches for this month; prior batch includes stay (auditable).
            break

    report["n_included"] = sum(len(v) for v in eligible_by_domain.values())
    report["n_excluded"] = len(all_excluded)
    return eligible_by_domain, all_excluded, report


def curate_sample_benchmark(
    cfg: dict[str, Any],
    *,
    collect_manifest: dict[str, Any] | None = None,
    collect_manifest_path: Path | None = None,
    n_per: int | None = None,
    dry_run: bool = False,
    out_dir: Path | None = None,
    call_model: CallModelFn | None = None,
    prompt_template: str | None = None,
    prescreen_prompt_template: str | None = None,
    only_months: list[str] | None = None,
    refresh_months: bool = False,
) -> dict[str, Any]:
    """Stage 2: select exactly per_month_n markets per sample month when possible.

    ``only_months``: re-run GPT only for those months; merge prior ``selected`` /
    reports for other months from ``out_dir/curate_manifest.json`` when present.
    ``refresh_months``: delete freeze files for the active months before calling.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    out = ensure_dir(out_dir or default_out_dir(cfg, run_id))
    per = int(n_per) if n_per is not None else per_month_n(cfg)
    max_cand, max_chars = stage2_caps(cfg)
    domains = sample_domains(cfg)
    all_months = sample_months(cfg)
    active_months = list(only_months) if only_months is not None else list(all_months)
    active_set = set(active_months)
    if not active_set.issubset(set(all_months)):
        raise ValueError(f"only_months must be subset of {all_months}")

    if dry_run and collect_manifest is None and collect_manifest_path is None:
        manifest = {
            "run_id": run_id,
            "mode": "dry_run",
            "stage": "curate",
            "selection_version": SELECTION_VERSION,
            "per_month_n": per,
            "target_n": sample_target_n(cfg),
            "months": sample_months(cfg),
            "max_candidates_per_month_prompt": max_cand,
            "max_bucket_prompt_chars": max_chars,
            "prescreen": {
                "batched": True,
                "max_candidates_per_batch": max_cand,
                "max_prompt_chars": max_chars,
                "no_silent_truncate": True,
            },
            "month_selection": {
                "per_month_n": per,
                "offer": "full_include_pool_heat_ranked",
                "domain_quota": "soft_prompt_only",
            },
            "note": "Dry-run: curation criteria only; no OpenAI calls.",
            "ledger_written": False,
        }
        write_json_atomic(out / "curate_manifest.json", manifest)
        write_json_atomic(out / "manifest.json", manifest)
        print(f"Dry-run sample-benchmark curate criteria → {out / 'manifest.json'}")
        return manifest

    if collect_manifest is None:
        if collect_manifest_path is None:
            raise ValueError("curate requires collect_manifest or collect_manifest_path")
        collect_manifest = read_json(collect_manifest_path)
    assert collect_manifest is not None
    qualified = list(collect_manifest.get("qualified") or [])
    prescreen_cap = int(sample_section(cfg).get(
        "prescreen_candidates_per_tag_month", DEFAULT_PRESCREEN_TAG_MONTH_CAP
    ))
    prescreen_offered, prescreen_sampled_out = sample_prescreen_pool(
        qualified, cap_per_tag_month=prescreen_cap
    )

    if dry_run:
        # Rehearse Stage 2 pool sizes / batch plans; no model calls.
        by_month = _pool_by_month(prescreen_offered, sample_months(cfg))
        prescreen_template = prescreen_prompt_template or load_prescreen_prompt()
        month_reports: list[dict[str, Any]] = []
        for month in sample_months(cfg):
            candidates = list(by_month.get(month, []))
            batches, plan_fail = plan_prescreen_batches(
                candidates,
                month=month,
                max_candidates=max_cand,
                max_chars=max_chars,
                domain_labels=domains,
                prompt_template=prescreen_template,
            )
            month_reports.append({
                "month": month,
                "n_candidates": len(candidates),
                "n_batches_planned": len(batches),
                "batch_sizes": [len(b) for b in batches],
                "fail_closed_reason": plan_fail,
                "month_select_target_n": per,
                "month_select_offer": "full_include_pool_heat_ranked",
                "skipped_model_call": True,
            })
        manifest = {
            "run_id": run_id,
            "mode": "dry_run",
            "stage": "curate",
            "selection_version": SELECTION_VERSION,
            "collect_run_id": collect_manifest.get("run_id"),
            "n_qualified_pool": len(qualified),
            "n_prescreen_offered": len(prescreen_offered),
            "n_prescreen_sampled_out": len(prescreen_sampled_out),
            "per_month_n": per,
            "target_n": sample_target_n(cfg),
            "max_candidates_per_month_prompt": max_cand,
            "max_bucket_prompt_chars": max_chars,
            "prescreen_month_reports": month_reports,
            "note": "Dry-run: batch-plan rehearsal only; no OpenAI calls.",
            "ledger_written": False,
        }
        write_json_atomic(out / "curate_manifest.json", manifest)
        write_json_atomic(out / "manifest.json", manifest)
        print(f"Dry-run sample-benchmark curate pool check → {out / 'manifest.json'}")
        return manifest

    pool_index = {str(r["market_id"]): r for r in qualified if r.get("market_id")}
    by_month = _pool_by_month(prescreen_offered, all_months)
    effective_template = prompt_template or load_curation_prompt()
    template_sha256 = hashlib.sha256(effective_template.encode("utf-8")).hexdigest()
    prescreen_template = prescreen_prompt_template or load_prescreen_prompt()
    prescreen_template_sha256 = hashlib.sha256(
        prescreen_template.encode("utf-8")
    ).hexdigest()

    refreshed: list[str] = []
    if refresh_months:
        refreshed = clear_month_curate_freezes(out, active_months)
        if refreshed:
            print(f"Refreshed {len(refreshed)} freeze file(s) for months {active_months}")

    prior = _load_prior_curate_manifest(out) if only_months is not None else None
    prior_by_month = _prior_selected_by_month(prior, all_months)
    prior_prescreen_reports = {
        str(r.get("month")): r
        for r in (prior or {}).get("prescreen_month_reports") or []
        if isinstance(r, dict) and r.get("month")
    }
    prior_month_reports = {
        str(r.get("month")): r
        for r in (prior or {}).get("month_select_reports") or []
        if isinstance(r, dict) and r.get("month")
    }

    model_fn = call_model or _default_openai_call(cfg)
    selected_by_month: dict[str, list[dict[str, Any]]] = {}
    month_select_reports: list[dict[str, Any]] = []
    prescreen_month_reports: list[dict[str, Any]] = []
    globally_used: set[str] = set()
    underlying_counts: dict[str, int] = defaultdict(int)
    all_selected: list[dict[str, Any]] = []
    all_excluded: list[dict[str, Any]] = [
        {
            "market_id": str(r["market_id"]),
            "month": str(r.get("sample_month") or ""),
            "retrieval_tag_id": r.get("retrieval_tag_id"),
            "stage": "prescreen_offer_sample",
            "exclusion_reason": PRESCREEN_SAMPLING_CAP_REASON,
        }
        for r in prescreen_sampled_out
    ]
    # Carry prior exclusions for inactive months (audit continuity).
    if prior and only_months is not None:
        for ex in prior.get("excluded") or []:
            if not isinstance(ex, dict):
                continue
            mo = str(ex.get("month") or "")
            if mo and mo not in active_set:
                all_excluded.append(ex)

    eligible_by_month_domain: dict[str, dict[str, list[dict[str, Any]]]] = {}

    # Seed prior selections for months we are not re-running (save GPT tokens).
    for month in all_months:
        if month in active_set:
            continue
        kept = list(prior_by_month.get(month) or [])
        selected_by_month[month] = kept
        for row in kept:
            mid = str(row.get("market_id") or "")
            if mid:
                globally_used.add(mid)
            ukey = str(row.get("underlying_event_key") or "")
            if ukey:
                underlying_counts[ukey] += 1
            all_selected.append(row)
        pre_rep = prior_prescreen_reports.get(month) or {
            "month": month,
            "n_candidates": 0,
            "n_batches_planned": 0,
            "batch_reports": [],
            "status": "carried_from_prior",
            "fail_closed_reason": None,
            "n_included": 0,
            "n_excluded": 0,
            "carried_from_prior": True,
        }
        pre_rep = {**pre_rep, "carried_from_prior": True}
        prescreen_month_reports.append(pre_rep)
        sel_rep = prior_month_reports.get(month) or {
            "month": month,
            "n_eligible_pool": len(kept),
            "n_candidates_offered": len(kept),
            "n_selected": len(kept),
            "validation_issues": [],
            "skipped_model_call": True,
            "status": "carried_from_prior",
            "selected_market_ids": [str(r.get("market_id") or "") for r in kept],
            "excluded": [],
        }
        sel_rep = {**sel_rep, "carried_from_prior": True, "n_selected": len(kept)}
        month_select_reports.append(sel_rep)

    # Phase A: batched month pre-screen (once per candidate; no 6× paid review).
    for month in all_months:
        if month not in active_set:
            continue
        month_pool = list(by_month.get(month, []))
        if not month_pool:
            eligible_by_month_domain[month] = {d: [] for d in domains}
            prescreen_month_reports.append({
                "month": month,
                "n_candidates": 0,
                "n_batches_planned": 0,
                "batch_reports": [],
                "status": "empty_month",
                "fail_closed_reason": None,
                "n_included": 0,
                "n_excluded": 0,
            })
            continue
        eligible_by_domain, month_excluded, month_report = _run_month_prescreen(
            month=month,
            month_pool=month_pool,
            domains=domains,
            max_cand=max_cand,
            max_chars=max_chars,
            out=out,
            cfg=cfg,
            model_fn=model_fn,
            template=prescreen_template,
            template_sha256=prescreen_template_sha256,
        )
        eligible_by_month_domain[month] = {
            d: list(eligible_by_domain.get(d, [])) for d in domains
        }
        all_excluded.extend(month_excluded)
        prescreen_month_reports.append(month_report)

    # Phase B: one month-select call per sample month (exactly ``per`` picks).
    # Offer the full heat-ranked INCLUDE pool (40 is only the Phase A batch size).
    for month in all_months:
        if month not in active_set:
            continue
        month_prescreen = next(
            (r for r in prescreen_month_reports if r["month"] == month), None
        )
        if month_prescreen and month_prescreen.get("fail_closed_reason"):
            selected_by_month[month] = []
            month_select_reports.append({
                "month": month,
                "n_eligible_pool": 0,
                "n_candidates_offered": 0,
                "n_selected": 0,
                "validation_issues": [str(month_prescreen["fail_closed_reason"])],
                "skipped_model_call": True,
                "status": "prescreen_fail_closed",
                "fail_closed_reason": month_prescreen["fail_closed_reason"],
                "selected_market_ids": [],
                "excluded": [],
            })
            continue

        eligible: list[dict[str, Any]] = []
        for domain in domains:
            for row in eligible_by_month_domain.get(month, {}).get(domain, []):
                mid = str(row["market_id"])
                if mid in globally_used:
                    continue
                eligible.append(row)
        if len(eligible) < per:
            selected_by_month[month] = []
            month_select_reports.append({
                "month": month,
                "n_eligible_pool": len(eligible),
                "n_candidates_offered": 0,
                "n_selected": 0,
                "validation_issues": [
                    f"NO_CANDIDATES_OR_SHORTFALL:{len(eligible)}<{per}"
                ],
                "skipped_model_call": True,
                "status": "shortfall",
                "fail_closed_reason": f"month_shortfall:{len(eligible)}<{per}",
                "selected_market_ids": [],
                "excluded": [],
            })
            continue

        offered = heat_rank_month_candidates(eligible)

        # Prompt size gate on the full offered set (fail closed; no silent truncate).
        prompt = build_month_select_prompt(
            month=month,
            target_n=per,
            candidates=offered,
            domain_labels=domains,
            prompt_template=effective_template,
        )
        if len(prompt) > max_chars:
            selected_by_month[month] = []
            month_select_reports.append({
                "month": month,
                "n_eligible_pool": len(eligible),
                "n_candidates_offered": len(offered),
                "n_selected": 0,
                "validation_issues": [
                    f"month_select_prompt_too_large:chars={len(prompt)}>max={max_chars}"
                ],
                "skipped_model_call": True,
                "status": "prompt_too_large",
                "fail_closed_reason": (
                    f"month_select_prompt_too_large:chars={len(prompt)}"
                    f">max={max_chars};fail_closed_no_silent_truncate"
                ),
                "selected_market_ids": [],
                "excluded": [],
            })
            continue

        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        call_path = ensure_dir(out / "month_calls") / f"{month.replace('-', '')}.json"
        candidate_ids = [str(c["market_id"]) for c in offered]
        meta = {
            "stage": "month",
            "month": month,
            "n_candidates": len(offered),
            "candidate_market_ids": candidate_ids,
            "json_schema": MONTH_SELECT_SCHEMA,
            "schema_name": "sample_benchmark_month_select",
            "n_eligible_pool": len(eligible),
            "target_n": per,
        }
        raw = _freeze_or_call(
            call_path=call_path,
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            template_sha256=template_sha256,
            candidate_ids=candidate_ids,
            model_fn=model_fn,
            meta=meta,
            cfg=cfg,
            record_extra={
                "stage": "month",
                "month": month,
                "n_eligible_pool": len(eligible),
            },
        )
        if not raw.get("selected") and not raw.get("excluded") and not raw.get("month_notes"):
            raw = {
                "selected": raw.get("selected") or [],
                "excluded": raw.get("excluded") or [],
                "month_notes": raw.get("month_notes") or "invalid_model_payload",
            }
        pool_ids = set(candidate_ids)
        selected, excluded, issues = _validate_month_response(
            raw,
            month=month,
            pool_ids=pool_ids,
            domain_labels=domains,
            target_n=per,
            globally_used=globally_used,
            underlying_counts=underlying_counts,
        )
        if any(
            i.startswith((
                "selected_and_excluded:",
                "unscreened_candidates:",
                "excluded_not_in_pool:",
            ))
            for i in issues
        ):
            selected = []
            issues.append("fail_closed_incomplete_agent_screen")
        enriched: list[dict[str, Any]] = []
        for row in selected:
            base = pool_index.get(row["market_id"], {})
            merged = {
                **base,
                **row,
                "event_id": make_sample_event_id(row["market_id"]),
                "notes": (
                    f"{SELECTION_VERSION} | month={month} | "
                    f"domain={row['domain']} | "
                    f"underlying={row['underlying_event_key']} | "
                    f"{row['selection_reason']}"
                ),
            }
            enriched.append(merged)
            globally_used.add(row["market_id"])
            underlying_counts[row["underlying_event_key"]] += 1
            all_selected.append(merged)
        for ex in excluded:
            all_excluded.append({**ex, "month": month, "stage": "month"})
        selected_by_month[month] = enriched
        month_select_reports.append({
            "month": month,
            "n_eligible_pool": len(eligible),
            "n_candidates_offered": len(offered),
            "n_selected": len(enriched),
            "n_excluded_reported": len(excluded),
            "validation_issues": issues,
            "skipped_model_call": False,
            "status": "ok" if len(enriched) == per else "incomplete",
            "month_call_record": str(call_path),
            "prompt_sha256": prompt_sha256,
            "month_notes": str(raw.get("month_notes") or raw.get("bucket_notes") or ""),
            "selected_market_ids": [r["market_id"] for r in enriched],
            "excluded": excluded,
            "by_domain": {
                d: sum(1 for r in enriched if r.get("domain") == d) for d in domains
            },
        })

    # Stable month order in reports (carried + refreshed mixed).
    prescreen_by = {str(r.get("month")): r for r in prescreen_month_reports}
    month_by = {str(r.get("month")): r for r in month_select_reports}
    prescreen_month_reports = [
        prescreen_by[m] for m in all_months if m in prescreen_by
    ]
    month_select_reports = [month_by[m] for m in all_months if m in month_by]

    fill = evaluate_month_fill(selected_by_month, cfg, n_per=per)
    collect_scan_complete = collect_manifest.get("scan_complete")
    incomplete_collect = collect_scan_complete is False
    ledger_written = False
    ledger_path: Path | None = None
    if fill["quota_complete"]:
        ledger_path = out / "sample_ledger.csv"
        write_sample_ledger(ledger_path, all_selected)
        ledger_written = True
        shortfall_note = None
    else:
        shortfall_note = (
            "Ledger NOT written: require exactly "
            f"{per} markets in each of {fill['n_months']} months "
            f"(target {fill['target_n']}). "
            f"shortfalls={fill['shortfalls']} overs={fill['overs']}"
        )

    id_counts: dict[str, int] = defaultdict(int)
    for row in all_selected:
        id_counts[str(row["market_id"])] += 1
    dup_ids = sorted([mid for mid, n in id_counts.items() if n > 1])

    fail_closed_prescreen = [
        r["month"] for r in prescreen_month_reports if r.get("fail_closed_reason")
    ]
    fail_closed_months = [
        r["month"] for r in month_select_reports if r.get("fail_closed_reason")
    ]

    manifest = {
        "run_id": run_id,
        "mode": "curate",
        "stage": "curate",
        "selection_version": SELECTION_VERSION,
        "collect_run_id": collect_manifest.get("run_id"),
        "curation_model": str(cfg["models"]["gpt_sol"]["model"]),
        "curation_prompt_template_sha256": template_sha256,
        "prescreen_prompt_template_sha256": prescreen_template_sha256,
        "only_months": active_months if only_months is not None else None,
        "refresh_months": bool(refresh_months),
        "refreshed_freeze_files": refreshed,
        "collect_scan_complete": collect_manifest.get("scan_complete"),
        "candidate_frame_capped": incomplete_collect,
        "candidate_frame_note": (
            "Quota-selected from a bounded, non-exhaustive candidate scan; "
            "do not describe the candidate universe as complete."
            if incomplete_collect else None
        ),
        "n_qualified_pool": len(qualified),
        "n_prescreen_offered": len(prescreen_offered),
        "n_prescreen_sampled_out": len(prescreen_sampled_out),
        "per_month_n": per,
        "target_n": fill["target_n"],
        "max_candidates_per_month_prompt": max_cand,
        "max_bucket_prompt_chars": max_chars,
        "prescreen": {
            "batched": True,
            "offer_cap_per_tag_month": prescreen_cap,
            "sampling": "sha256(month|retrieval_tag_id|market_id)",
            "sampled_out_reason": PRESCREEN_SAMPLING_CAP_REASON,
            "max_candidates_per_batch": max_cand,
            "max_prompt_chars": max_chars,
            "no_silent_truncate": True,
            "one_paid_review_per_candidate": True,
            "fail_closed_months": fail_closed_prescreen,
        },
        "month_selection": {
            "per_month_n": per,
            "offer": "full_include_pool_heat_ranked",
            "heat_rank": "heat_mid,heat_move,market_id",
            "domain_quota": "soft_prompt_only",
            "no_silent_truncate": True,
            "model_disposition_required_for_offered_only": True,
        },
        "prescreen_month_reports": prescreen_month_reports,
        "month_select_reports": month_select_reports,
        "fail_closed_month_select": fail_closed_months,
        "selected": all_selected,
        "excluded": all_excluded,
        "fill": fill,
        "cross_bucket_duplicate_market_ids": dup_ids,
        "cross_bucket_dedupe_ok": not dup_ids,
        "ledger_written": ledger_written,
        "ledger_path": str(ledger_path) if ledger_written else None,
        "shortfall_note": shortfall_note,
        "out_dir": str(out),
    }
    write_json_atomic(out / "curate_manifest.json", manifest)
    write_json_atomic(out / "selected.json", all_selected)
    write_json_atomic(out / "manifest.json", manifest)
    if ledger_written:
        print(
            f"Sample curate: quota-complete ({fill['n_selected_total']} markets); "
            f"wrote ledger {ledger_path} (manifest {out / 'curate_manifest.json'})"
        )
        if incomplete_collect:
            print("Candidate retrieval was page-capped; sampling frame is non-exhaustive.")
    else:
        print(
            f"Sample curate SHORTFALL: n_selected={fill['n_selected_total']}/"
            f"{fill['target_n']}; ledger not written. {out / 'curate_manifest.json'}"
        )
        if shortfall_note:
            print(shortfall_note)
        if fail_closed_prescreen:
            print(f"Fail-closed month pre-screen: {fail_closed_prescreen}")
        if fail_closed_months:
            print(f"Fail-closed month select: {fail_closed_months}")
    return manifest


def run_sample_benchmark(
    cfg: dict[str, Any],
    *,
    collect: bool = False,
    curate: bool = False,
    dry_run: bool = False,
    max_pages: int = 50,
    page_limit: int = 100,
    out_dir: Path | None = None,
    collect_manifest_path: Path | None = None,
    list_pages: Any = None,
    fetch_cutoff_history: Any = None,
    call_model: CallModelFn | None = None,
    only_months: list[str] | None = None,
    refresh_months: bool = False,
    expand_months: list[str] | None = None,
) -> dict[str, Any]:
    """CLI orchestration for --dry-run / --collect / --curate."""
    if dry_run and not collect and not curate:
        # Criteria-only for the full protocol.
        return collect_qualified_pool(cfg, dry_run=True, out_dir=out_dir)

    result: dict[str, Any] = {}
    collect_manifest: dict[str, Any] | None = None
    run_out = out_dir

    if collect:
        base_manifest = None
        if expand_months is not None:
            if collect_manifest_path is None:
                raise ValueError("--expand-months requires --from-collect MANIFEST")
            base_manifest = read_json(collect_manifest_path)
            if run_out is None and base_manifest.get("out_dir"):
                run_out = Path(str(base_manifest["out_dir"]))
        collect_manifest = collect_qualified_pool(
            cfg,
            page_limit=page_limit,
            max_pages=max_pages,
            dry_run=dry_run,
            out_dir=run_out,
            list_pages=list_pages,
            fetch_cutoff_history=fetch_cutoff_history,
            expand_months=expand_months,
            base_manifest=base_manifest,
        )
        result["collect"] = collect_manifest
        run_out = Path(collect_manifest["out_dir"]) if collect_manifest.get("out_dir") else run_out

    if curate:
        if dry_run and not collect_manifest and collect_manifest_path is None:
            return curate_sample_benchmark(cfg, dry_run=True, out_dir=run_out)
        if collect_manifest is None and collect_manifest_path is not None:
            collect_manifest = read_json(collect_manifest_path)
        if collect_manifest is None:
            raise ValueError(
                "--curate requires --collect in the same run or --from-collect MANIFEST"
            )
        # Write curate artifacts alongside collect when sharing out_dir.
        curate_out = run_out
        curate_manifest = curate_sample_benchmark(
            cfg,
            collect_manifest=collect_manifest,
            dry_run=dry_run,
            out_dir=curate_out,
            call_model=call_model,
            only_months=only_months,
            refresh_months=refresh_months,
        )
        result["curate"] = curate_manifest
        return curate_manifest if not collect else result

    if not collect and not curate:
        raise ValueError("Specify --dry-run, --collect, and/or --curate")
    return result.get("collect") or result
