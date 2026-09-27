"""Pydantic schemas and reason-code enums for the evaluation pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ReasonCode(str, Enum):
    NO_PRICE_AT_OR_BEFORE_FORECAST = "NO_PRICE_AT_OR_BEFORE_FORECAST"
    PRICE_LAG_EXCEEDED = "PRICE_LAG_EXCEEDED"
    FORECAST_EVIDENCE_LEAKAGE = "FORECAST_EVIDENCE_LEAKAGE"
    MISSING_MARKET_CLOSE_TIME = "MISSING_MARKET_CLOSE_TIME"
    INSUFFICIENT_TIME_SAFE_NEWS = "INSUFFICIENT_TIME_SAFE_NEWS"


class EvidenceItem(BaseModel):
    source_id: str
    title: str = ""
    url: str = ""
    published_at: datetime
    text: str
    is_terminal_or_resolution: bool = False


class ForecastEvidencePacket(BaseModel):
    """Forecast-only evidence. Must never contain future/outcome leakage."""

    event_id: str
    forecast_at: datetime
    items: list[EvidenceItem] = Field(default_factory=list)
    brief: str = ""

    @field_validator("items")
    @classmethod
    def no_terminal_flags(cls, items: list[EvidenceItem]) -> list[EvidenceItem]:
        for item in items:
            if item.is_terminal_or_resolution:
                raise ValueError(
                    f"forecast evidence item {item.source_id} marked terminal/resolution"
                )
        return items


class CuratedEvent(BaseModel):
    """Sample→formal cohort row (question/criteria come from frozen market.json)."""

    event_id: str
    underlying_event_id: str = ""
    question: str
    resolution_criteria: str = ""
    category: str = ""
    actual_terminal_time: datetime | None = None
    outcome: int
    exclusion_reason: str = ""


class ForecastResult(BaseModel):
    event_id: str
    model: str
    requested_model: str
    resolved_model: str | None = None
    prompt_version: str
    probability_yes: float
    forecast_at: datetime
    run_id: str
    raw_path: str | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    cost_source: str | None = None


class FormalNewsItem(BaseModel):
    source_id: str
    title: str
    rss_url: str
    article_url: str
    source: str
    published_at: datetime
    modified_at: datetime
    body_sha256: str
    body_path: str
    relevance_score: int = Field(ge=0, le=5)
    relevance_quote: str
    summary: str
    model: str
    relevance_request_sha256: str
    summary_request_sha256: str


class FormalForecastInstance(BaseModel):
    instance_id: str
    event_id: str
    underlying_event_id: str
    horizon_hours: int
    forecast_at: datetime
    market_close_time: datetime | None = None
    clock_definition: str = ""
    # Auxiliary only: never used for formal cutoffs, prices, or inclusion.
    actual_terminal_time: datetime | None = None
    evidence_start: datetime | None = None
    outcome: int
    market_probability: float
    market_probability_asof: datetime
    question: str
    resolution_criteria: str
    news_items: list[FormalNewsItem]
    construction_version: str
    provenance: dict[str, str] = Field(default_factory=dict)
