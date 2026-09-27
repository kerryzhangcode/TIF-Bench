"""Frozen Google News RSS retrieval helpers for formal construction."""

from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode, urlparse

from tifbench.schemas import EvidenceItem

STOPWORDS = frozenset(
    "will the a an by before after in on at of to for and or is be as with from "
    "happen final effective agreement announced".split()
)
MARKET_TERMS = ("polymarket", "kalshi", "metaculus", "betting odds", "prediction market")


def _terms(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in STOPWORDS
    ]


def rss_url(query: str, start: datetime, cutoff: datetime) -> str:
    search = (
        f"{query} after:{start.date().isoformat()} "
        f"before:{(cutoff + timedelta(days=1)).date().isoformat()}"
    )
    return "https://news.google.com/rss/search?" + urlencode({
        "q": search, "hl": "en-US", "gl": "US", "ceid": "US:en",
    })


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def parse_rss(
    xml_text: str,
    *,
    start: datetime,
    cutoff: datetime,
    query_terms: set[str],
    exclude_market_terms: bool = True,
    inclusive_cutoff: bool = False,
    audit: list[dict[str, str]] | None = None,
) -> list[EvidenceItem]:
    """Parse and reject missing/naive/future timestamps and market commentary."""
    root = ET.fromstring(xml_text)
    items: list[EvidenceItem] = []
    for node in root.findall(".//channel/item"):
        title = _clean(node.findtext("title") or "")
        link = (node.findtext("link") or "").strip()
        source = _clean(node.findtext("source") or "")
        description = _clean(node.findtext("description") or "")
        published_raw = node.findtext("pubDate")

        def record(status: str) -> None:
            if audit is not None:
                audit.append({
                    "title": title, "url": link, "pubDate": published_raw or "",
                    "status": status,
                })

        if not title or not link or not published_raw:
            record("MISSING_RSS_FIELD")
            continue
        try:
            published = parsedate_to_datetime(published_raw)
        except (TypeError, ValueError):
            record("INVALID_RSS_TIMESTAMP")
            continue
        if published.tzinfo is None:
            record("NAIVE_RSS_TIMESTAMP")
            continue
        published = published.astimezone(timezone.utc)
        if not (
            start <= published <= cutoff if inclusive_cutoff
            else start <= published < cutoff
        ):
            record("OUTSIDE_FORECAST_WINDOW")
            continue
        content = f"{title} {source} {description}".lower()
        if exclude_market_terms and any(term in content for term in MARKET_TERMS):
            record("PREDICTION_MARKET_SOURCE")
            continue
        if query_terms and len(query_terms & set(_terms(f"{title} {description}"))) < 2:
            record("LEXICAL_PREFILTER")
            continue
        if urlparse(link).scheme not in ("https", "http"):
            record("INVALID_RSS_URL")
            continue
        source_id = "rss_" + hashlib.sha256(link.encode("utf-8")).hexdigest()[:16]
        items.append(EvidenceItem(
            source_id=source_id, title=title, url=link,
            published_at=published, text=f"{source}: {description}"[:1000],
        ))
        record("CANDIDATE")
    return items
