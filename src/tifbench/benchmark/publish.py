"""Publishable news-grounded forecasting benchmark (JSONL + schema)."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tifbench.evidence.formal import (
    _rank_key,
    combined_evidence_score,
    recency_score,
)
from tifbench.io import ensure_dir, project_root, read_json, write_json, write_jsonl
from tifbench.polymarket.clock import (
    parse_gamma_timestamp,
    parse_market_active_start,
    parse_market_closed_time,
)
from tifbench.schemas import FormalForecastInstance, FormalNewsItem

BENCHMARK_VERSION = "formal"
SCHEMA_NAME = "schema.json"
BENCHMARK_NAME = "benchmark.jsonl"
PACKAGE_SCHEMA = Path(__file__).resolve().parent / SCHEMA_NAME

_ISO_AWARE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def default_benchmark_dir(cfg: dict[str, Any] | None = None) -> Path:
    if cfg and cfg.get("paths", {}).get("benchmark"):
        return Path(cfg["paths"]["benchmark"])
    if cfg and cfg.get("paths", {}).get("curated"):
        return Path(cfg["paths"]["curated"]).resolve().parent / "benchmark"
    return project_root() / "data" / "benchmark"


def load_schema() -> dict[str, Any]:
    return read_json(PACKAGE_SCHEMA)


def to_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc_iso(value: str) -> datetime:
    if not isinstance(value, str) or not _ISO_AWARE.match(value):
        raise ValueError(f"invalid UTC ISO timestamp: {value!r}")
    text = value.replace("Z", "+00:00")
    ts = datetime.fromisoformat(text)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp missing timezone: {value!r}")
    return ts.astimezone(timezone.utc)


def outcome_label(outcome: int) -> str:
    if outcome == 1:
        return "YES"
    if outcome == 0:
        return "NO"
    raise ValueError(f"unsupported outcome int: {outcome!r}")


def outcome_int(label: str) -> int:
    if label == "YES":
        return 1
    if label == "NO":
        return 0
    raise ValueError(f"unsupported outcome label: {label!r}")


def _public_article_url(raw: str | None) -> str | None:
    """Return a determinate public article URL, or None — never invent."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if not (text.startswith("http://") or text.startswith("https://")):
        return None
    return text


def _market_public_url(market: dict[str, Any]) -> str | None:
    for key in ("url", "eventUrl", "marketUrl"):
        url = _public_article_url(market.get(key))
        if url is not None:
            return url
    return None


def _resolution_utc(market: dict[str, Any]) -> str | None:
    """Formal resolution time only when an explicit reliable field exists.

    Gamma endDate / umaEndDate / closedTime are not treated as resolution time.
    """
    for key in ("resolvedTime", "resolutionTime", "resolved_at", "resolution_at"):
        ts = parse_gamma_timestamp(market.get(key))
        if ts is not None:
            return to_utc_iso(ts)
    return None


def instance_to_benchmark_row(
    instance: FormalForecastInstance,
    market: dict[str, Any],
) -> dict[str, Any]:
    """Convert one formal construction instance into a publishable benchmark row.

    Ranking / freshness use the formal half-life (RECENCY_HALF_LIFE_DAYS = 20).
    """
    market_id = market.get("id")
    if market_id is None or str(market_id).strip() == "":
        raise ValueError(f"{instance.instance_id}: frozen market.json missing id")
    start = parse_market_active_start(market)
    close = parse_market_closed_time(market)
    if start is None:
        raise ValueError(f"{instance.instance_id}: frozen market.json missing startDate/createdAt")
    if close is None:
        raise ValueError(f"{instance.instance_id}: frozen market.json missing closedTime")
    if instance.market_close_time is not None and instance.market_close_time != close:
        raise ValueError(f"{instance.instance_id}: market_close_time != frozen closedTime")
    if not instance.news_items:
        raise ValueError(f"{instance.instance_id}: empty evidence pool")

    ranked = sorted(
        instance.news_items,
        key=lambda n: _rank_key(
            n.relevance_score,
            n.published_at,
            n.source_id,
            instance.forecast_at,
        ),
    )
    if [n.source_id for n in ranked] != [n.source_id for n in instance.news_items]:
        raise ValueError(
            f"{instance.instance_id}: news_items order diverges from "
            f"formal ranking function"
        )

    evidence: list[dict[str, Any]] = []
    for rank, news in enumerate(ranked, start=1):
        evidence.append(
            _evidence_item(
                news,
                rank=rank,
                forecast_at=instance.forecast_at,
            )
        )

    slug = market.get("slug")
    if slug is not None:
        slug = str(slug).strip() or None

    row = {
        "instance_id": f"m{market_id}_tminus{instance.horizon_hours}h",
        "market": {
            "market_id": str(market_id),
            "slug": slug,
            "question": instance.question,
            "resolution_criteria": instance.resolution_criteria,
            "start_utc": to_utc_iso(start),
            "close_utc": to_utc_iso(close),
            "resolution_utc": _resolution_utc(market),
            "url": _market_public_url(market),
        },
        "forecast": {
            "cutoff_utc": to_utc_iso(instance.forecast_at),
            "horizon_days": float(instance.horizon_hours) / 24.0,
        },
        "polymarket": {
            "yes_probability": float(instance.market_probability),
            "observed_at_utc": to_utc_iso(instance.market_probability_asof),
        },
        "outcome": outcome_label(instance.outcome),
        "evidence": evidence,
    }
    validate_benchmark_row(row)
    return row


def _evidence_item(
    news: FormalNewsItem,
    *,
    rank: int,
    forecast_at: datetime,
) -> dict[str, Any]:
    if news.relevance_score not in (3, 4, 5):
        raise ValueError(f"{news.source_id}: evidence_score must be 3–5, got {news.relevance_score}")
    if news.published_at > forecast_at:
        raise ValueError(f"{news.source_id}: published_at after forecast cutoff")
    if news.modified_at > forecast_at:
        raise ValueError(f"{news.source_id}: modified_at after forecast cutoff")
    url = _public_article_url(news.article_url)
    if url is None:
        # Schema requires string URI; refuse rather than invent a URL.
        raise ValueError(f"{news.source_id}: article URL undetermined (refusing to invent)")
    freshness = recency_score(news.published_at, forecast_at)
    ranking = combined_evidence_score(
        news.relevance_score,
        news.published_at,
        forecast_at,
    )
    text = news.summary
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{news.source_id}: empty frozen summary text")
    return {
        "article_id": news.source_id,
        "rank": rank,
        "title": news.title,
        "source": news.source,
        "published_at_utc": to_utc_iso(news.published_at),
        "url": url,
        "text": text,
        "evidence_score": int(news.relevance_score),
        "freshness_score": float(freshness),
        "ranking_score": float(ranking),
    }


def _check_additional_properties(node: Any, schema: dict[str, Any], path: str) -> None:
    if schema.get("type") == "object" or "properties" in schema:
        if not isinstance(node, dict):
            raise ValueError(f"{path}: expected object")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(node) - set(props)
            if extra:
                raise ValueError(f"{path}: unexpected properties {sorted(extra)}")
        for key in schema.get("required", []):
            if key not in node:
                raise ValueError(f"{path}: missing required field {key!r}")
        for key, value in node.items():
            if key in props:
                _validate_node(value, props[key], f"{path}.{key}")
        return
    if schema.get("type") == "array" or "items" in schema:
        if not isinstance(node, list):
            raise ValueError(f"{path}: expected array")
        min_items = schema.get("minItems")
        if min_items is not None and len(node) < min_items:
            raise ValueError(f"{path}: expected at least {min_items} items")
        item_schema = schema.get("items", {})
        for i, item in enumerate(node):
            _validate_node(item, item_schema, f"{path}[{i}]")
        return


def _type_ok(value: Any, type_spec: str | list[str]) -> bool:
    allowed = type_spec if isinstance(type_spec, list) else [type_spec]
    for t in allowed:
        if t == "null" and value is None:
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
    return False


def _validate_node(node: Any, schema: dict[str, Any], path: str) -> None:
    if "type" in schema and not _type_ok(node, schema["type"]):
        raise ValueError(f"{path}: type mismatch for {schema['type']}, got {type(node).__name__}")
    if node is None:
        return
    if "enum" in schema and node not in schema["enum"]:
        raise ValueError(f"{path}: value {node!r} not in enum {schema['enum']}")
    if "minimum" in schema and isinstance(node, (int, float)) and node < schema["minimum"]:
        raise ValueError(f"{path}: {node} < minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(node, (int, float)) and node > schema["maximum"]:
        raise ValueError(f"{path}: {node} > maximum {schema['maximum']}")
    fmt = schema.get("format")
    if fmt == "date-time" and isinstance(node, str):
        parse_utc_iso(node)
    if fmt == "uri" and isinstance(node, str):
        if _public_article_url(node) is None:
            raise ValueError(f"{path}: invalid uri {node!r}")
    if schema.get("type") in ("object", "array") or "properties" in schema or "items" in schema:
        _check_additional_properties(node, schema, path)


def validate_benchmark_row(row: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    schema = schema or load_schema()
    _validate_node(row, schema, "$")
    # Temporal integrity beyond JSON Schema.
    cutoff = parse_utc_iso(row["forecast"]["cutoff_utc"])
    close = parse_utc_iso(row["market"]["close_utc"])
    start = parse_utc_iso(row["market"]["start_utc"])
    if not (start <= cutoff <= close):
        raise ValueError(
            f"{row.get('instance_id')}: require start_utc <= cutoff_utc <= close_utc"
        )
    observed = parse_utc_iso(row["polymarket"]["observed_at_utc"])
    if observed > cutoff:
        raise ValueError(f"{row.get('instance_id')}: polymarket observation after cutoff")
    ranks = [item["rank"] for item in row["evidence"]]
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError(f"{row.get('instance_id')}: evidence ranks must be 1..n contiguous")
    scores = [item["ranking_score"] for item in row["evidence"]]
    if scores != sorted(scores, reverse=True):
        # Allow equal scores; only require non-increasing order.
        for left, right in zip(scores, scores[1:]):
            if left < right:
                raise ValueError(f"{row.get('instance_id')}: evidence not ordered by ranking_score")
    for item in row["evidence"]:
        published = parse_utc_iso(item["published_at_utc"])
        if published > cutoff:
            raise ValueError(
                f"{row.get('instance_id')}: evidence {item['article_id']} after cutoff"
            )


def row_content_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def export_benchmark_from_instances(
    instances_path: Path,
    out_dir: Path,
    *,
    expected_n: int | None = None,
    required_horizons: tuple[int, ...] | None = None,
    min_articles: int = 3,
) -> Path:
    """Export complete market panels from curated instances."""
    raw_instances = read_json(instances_path)
    if not isinstance(raw_instances, list):
        raise ValueError(f"{instances_path}: expected a JSON array")
    instances = [FormalForecastInstance.model_validate(x) for x in raw_instances]
    if required_horizons is not None:
        by_event: dict[str, list[FormalForecastInstance]] = {}
        for instance in instances:
            by_event.setdefault(instance.event_id, []).append(instance)
        required = set(required_horizons)
        instances = [
            instance
            for panel in by_event.values()
            if len(panel) == len(required)
            and {item.horizon_hours for item in panel} == required
            and all(len(item.news_items) >= min_articles for item in panel)
            for instance in panel
        ]
    rows: list[dict[str, Any]] = []
    for instance in instances:
        market_path = instance.provenance.get("market")
        if not market_path:
            raise ValueError(f"{instance.instance_id}: missing provenance.market")
        market = read_json(Path(market_path))
        rows.append(instance_to_benchmark_row(instance, market))
    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"expected {expected_n} benchmark rows, got {len(rows)}")
    return write_benchmark_dir(out_dir, rows)


def write_benchmark_dir(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(out_dir)
    schema = load_schema()
    for row in rows:
        validate_benchmark_row(row, schema)
    _validate_unique_instances(rows)
    write_json(out_dir / SCHEMA_NAME, schema)
    out_path = out_dir / BENCHMARK_NAME
    write_jsonl(out_path, rows)
    return out_path


def load_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    from tifbench.io import read_jsonl

    rows = read_jsonl(path)
    # Frozen older cohorts may carry an earlier schema. Read their own schema
    # for input validation; all newly written cohorts use the current schema.
    sibling_schema = path.parent / SCHEMA_NAME
    schema = read_json(sibling_schema) if sibling_schema.exists() else load_schema()
    for row in rows:
        validate_benchmark_row(row, schema)
    _validate_unique_instances(rows)
    return rows


def _validate_unique_instances(rows: list[dict[str, Any]]) -> None:
    ids = [row["instance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate instance_id in benchmark")
    market_horizons = [
        (row["market"]["market_id"], row["forecast"]["horizon_days"])
        for row in rows
    ]
    if len(market_horizons) != len(set(market_horizons)):
        raise ValueError("duplicate market × horizon in benchmark")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
