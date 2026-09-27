"""Conservative extraction of publisher article text and temporal metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

MAX_HTML_BYTES = 2_000_000
HTTP_BACKOFF_DEFAULT_SECONDS = 2.0
HTTP_BACKOFF_CAP_SECONDS = 5.0
BROWSER_TIMEOUT_MS = 15_000
_BROWSER_LOCK = threading.Lock()
# Set after a hard Chromium-missing failure so we do not re-launch 90× per horizon.
_BROWSER_UNAVAILABLE: str | None = None


def _chromium_executable_exists(playwright: object) -> bool:
    try:
        exe = Path(playwright.chromium.executable_path)  # type: ignore[attr-defined]
    except Exception:
        return False
    return exe.is_file()


def _prepare_playwright_browsers_path(playwright: object) -> None:
    """Drop a broken PLAYWRIGHT_BROWSERS_PATH (common under Cursor sandbox).

    Cursor may point the env at an empty cache dir. Prefer the default
    ``~/Library/Caches/ms-playwright`` (or platform equivalent) when the
    override has no Chromium binary.
    """
    if _chromium_executable_exists(playwright):
        return
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
        return
    os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)


def reset_browser_availability_for_tests() -> None:
    """Test helper: clear the process-level browser unavailable latch."""
    global _BROWSER_UNAVAILABLE
    _BROWSER_UNAVAILABLE = None


class ArticleFailure(ValueError):
    """A rejected publisher page with a reportable construction reason."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def non_article_page_reason(final_url: str, html_text: str) -> str | None:
    """Identify clear access gates without treating article topics as gate signals."""
    parsed = urlparse(final_url)
    host = (parsed.hostname or "").lower()
    lowered = html_text.lower()
    title_match = re.search(r"<title[^>]*>(.*?)</title>", lowered, re.S)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    if (host.startswith("consent.") or "/collectconsent" in parsed.path.lower()
            or ("before you continue to google" in title)):
        return "CONSENT_PAGE"
    if ("__cf_chl_" in parsed.query.lower() or title == "just a moment..."
            or title.startswith("attention required! | cloudflare")):
        return "ACCESS_BLOCKED"
    if (re.search(r"<[^>]+class\s*=\s*['\"][^'\"]*(?:articlebody-meteredpaywallpreview|paywall-preview)",
                  lowered)):
        return "PAYWALL_PREVIEW"
    return None


def public_http_url(url: str) -> bool:
    """Accept http(s) URLs with a hostname. No DNS/IP SSRF filtering.

    Local construct pipelines fetch publisher pages behind VPN/fake-ip DNS;
    resolving and requiring ``is_global`` IPs rejects Clash ``198.18.0.0/15``
    and blocks otherwise-valid article fetches.
    """
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def _bounded_backoff_seconds(response: httpx.Response | None) -> float:
    if response is None:
        return HTTP_BACKOFF_DEFAULT_SECONDS
    raw = response.headers.get("retry-after")
    if raw is None:
        return HTTP_BACKOFF_DEFAULT_SECONDS
    try:
        return min(max(float(raw), 0.0), HTTP_BACKOFF_CAP_SECONDS)
    except ValueError:
        return HTTP_BACKOFF_DEFAULT_SECONDS


def _http_get_html(client: httpx.Client, url: str, *, max_redirects: int = 5) -> tuple[str, str]:
    """Follow http(s) redirects; cap response size."""
    current = url
    for _ in range(max_redirects + 1):
        if not public_http_url(current):
            raise ValueError("non-public article URL")
        with client.stream("GET", current, follow_redirects=False) as response:
            if response.status_code in (301, 302, 303, 307, 308):
                target = response.headers.get("location")
                if not target:
                    raise ValueError("redirect without location")
                current = urljoin(current, target)
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                raise ValueError("article is not HTML")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_HTML_BYTES:
                    raise ValueError("article HTML exceeds 2 MB")
                chunks.append(chunk)
            return current, b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
    raise ValueError("too many article redirects")


def fetch_html_with_browser(url: str, *, timeout_ms: int = BROWSER_TIMEOUT_MS) -> tuple[str, str]:
    """Open a public URL in headless Chromium, wait for DOM, return HTML.

    Optional dependency: ``pip install '.[browser]' && playwright install chromium``.
    No login, paywall bypass, CAPTCHA solving, or stealth.
    """
    global _BROWSER_UNAVAILABLE
    if not public_http_url(url):
        raise ValueError("non-public article URL")
    if _BROWSER_UNAVAILABLE is not None:
        raise ValueError(_BROWSER_UNAVAILABLE)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        _BROWSER_UNAVAILABLE = (
            "browser unavailable: install optional dependency with "
            "`pip install '.[browser]' && playwright install chromium`"
        )
        raise ValueError(_BROWSER_UNAVAILABLE) from exc

    def _fetch_once() -> tuple[str, str]:
        with sync_playwright() as playwright:
            _prepare_playwright_browsers_path(playwright)
        # Re-enter after possibly clearing PLAYWRIGHT_BROWSERS_PATH so the
        # driver resolves the default ms-playwright cache.
        with sync_playwright() as playwright:
            if not _chromium_executable_exists(playwright):
                raise RuntimeError("Executable doesn't exist")
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                return page.url, page.content()
            finally:
                browser.close()

    try:
        with _BROWSER_LOCK:
            if _BROWSER_UNAVAILABLE is not None:
                raise ValueError(_BROWSER_UNAVAILABLE)
            try:
                final_url, html_text = _fetch_once()
            except Exception as exc:
                msg = str(exc)
                missing = (
                    "Executable doesn't exist" in msg
                    or "BrowserType.launch" in msg
                )
                if missing and "PLAYWRIGHT_BROWSERS_PATH" in os.environ:
                    os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
                    final_url, html_text = _fetch_once()
                elif missing:
                    _BROWSER_UNAVAILABLE = (
                        "browser unavailable: Chromium executable missing; run "
                        "`python -m playwright install chromium`"
                    )
                    raise ValueError(_BROWSER_UNAVAILABLE) from exc
                else:
                    raise
    except ValueError:
        raise
    except Exception as exc:
        msg = str(exc)
        if "Executable doesn't exist" in msg or "BrowserType.launch" in msg:
            _BROWSER_UNAVAILABLE = (
                "browser unavailable: Chromium executable missing; run "
                "`python -m playwright install chromium`"
            )
            raise ValueError(_BROWSER_UNAVAILABLE) from exc
        raise ValueError(f"browser fetch failed: {type(exc).__name__}: {exc}") from exc
    if not public_http_url(final_url):
        raise ValueError("non-public article URL")
    if len(html_text.encode("utf-8", errors="replace")) > MAX_HTML_BYTES:
        raise ValueError("article HTML exceeds 2 MB")
    if "<html" not in html_text.lower() and "<body" not in html_text.lower():
        raise ValueError("article is not HTML")
    return final_url, html_text


def _browser_fallback(url: str, *, reason: str) -> tuple[str, str]:
    try:
        return fetch_html_with_browser(url)
    except ValueError as exc:
        if reason in ("HTTP 401", "HTTP 403"):
            raise ArticleFailure("ACCESS_BLOCKED", f"{reason}; {exc}") from exc
        raise ValueError(f"{reason}; {exc}") from exc


def fetch_public_html(client: httpx.Client, url: str, *, max_redirects: int = 5) -> tuple[str, str]:
    """HTTP-primary fetch with a single Playwright fallback for hard blocks.

    - 401/403: one browser attempt
    - 429: one bounded HTTP backoff/retry, then one browser attempt if still failing
    - timeout: one HTTP retry, then one browser attempt if still failing
    """
    try:
        return _http_get_html(client, url, max_redirects=max_redirects)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            return _browser_fallback(url, reason=f"HTTP {status}")
        if status == 429:
            time.sleep(_bounded_backoff_seconds(exc.response))
            try:
                return _http_get_html(client, url, max_redirects=max_redirects)
            except (httpx.HTTPStatusError, httpx.TimeoutException):
                return _browser_fallback(url, reason="HTTP 429 after retry")
        raise
    except httpx.TimeoutException:
        try:
            return _http_get_html(client, url, max_redirects=max_redirects)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            return _browser_fallback(url, reason="HTTP timeout after retry")


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.jsonld: list[str] = []
        self.paragraphs: list[str] = []
        self._article_depth = 0
        self._skip_depth = 0
        self._capture_p = False
        self._p_parts: list[str] = []
        self._capture_jsonld = False
        self._jsonld_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
            if tag == "script" and "ld+json" in (values.get("type") or ""):
                self._capture_jsonld = True
                self._jsonld_parts = []
            return
        if self._skip_depth:
            return
        if tag == "article":
            self._article_depth += 1
        if tag == "meta":
            key = (values.get("property") or values.get("name") or "").lower()
            if key and values.get("content"):
                self.meta[key] = values["content"] or ""
        if tag == "link" and (values.get("rel") or "").lower() == "canonical":
            self.meta["canonical"] = values.get("href") or ""
        if tag == "p" and self._article_depth:
            self._capture_p = True
            self._p_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip_depth:
            if tag == "script" and self._capture_jsonld:
                self.jsonld.append("".join(self._jsonld_parts))
                self._capture_jsonld = False
            self._skip_depth -= 1
            return
        if tag == "p" and self._capture_p:
            paragraph = re.sub(r"\s+", " ", "".join(self._p_parts)).strip()
            if paragraph:
                self.paragraphs.append(paragraph)
            self._capture_p = False
        if tag == "article" and self._article_depth:
            self._article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._capture_jsonld:
            self._jsonld_parts.append(data)
        if self._capture_p and not self._skip_depth:
            self._p_parts.append(data)


def _article_nodes(value: object) -> list[dict]:
    if isinstance(value, list):
        return [node for item in value for node in _article_nodes(item)]
    if not isinstance(value, dict):
        return []
    nodes = [value]
    for key in ("@graph", "mainEntity"):
        if key in value:
            nodes.extend(_article_nodes(value[key]))
    return nodes


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else None


def _normalize_body(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _trafilatura_body(html_text: str) -> str | None:
    """Optional fallback extractor. Returns None if unavailable or empty."""
    try:
        import trafilatura
    except ImportError:
        return None
    try:
        extracted = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=False,
        )
    except Exception:
        return None
    if not isinstance(extracted, str):
        return None
    body = _normalize_body(extracted)
    return body or None


def extract_article(
    html_text: str,
    final_url: str,
    *,
    min_body_chars: int = 400,
) -> dict[str, object]:
    """Extract article body with primary rules, then trafilatura if too short.

    Primary (unchanged): JSON-LD ``articleBody``, else ``<article><p>`` only.
    Timestamps / canonical always come from the primary parse so temporal gates
    stay stable. Trafilatura only replaces the body when primary length is below
    ``min_body_chars``.
    """
    parser = _ArticleParser()
    parser.feed(html_text)
    nodes: list[dict] = []
    for raw in parser.jsonld:
        try:
            nodes.extend(_article_nodes(json.loads(raw)))
        except json.JSONDecodeError:
            continue
    article_nodes = [n for n in nodes if "Article" in str(n.get("@type", ""))]
    node = next((n for n in article_nodes if n.get("articleBody")),
                article_nodes[0] if article_nodes else {})
    jsonld_body = node.get("articleBody") if isinstance(node.get("articleBody"), str) else ""
    if jsonld_body:
        body = _normalize_body(jsonld_body)
        body_source = "jsonld"
    else:
        body = _normalize_body("\n".join(parser.paragraphs))
        body_source = "article" if body else "empty"
    published = _timestamp(node.get("datePublished") or
                           parser.meta.get("article:published_time") or
                           parser.meta.get("datepublished"))
    modified = _timestamp(node.get("dateModified") or
                          parser.meta.get("article:modified_time") or
                          parser.meta.get("datemodified"))
    canonical = urljoin(final_url, str(node.get("url") or parser.meta.get("canonical") or final_url))
    if len(body) < int(min_body_chars):
        fallback = _trafilatura_body(html_text)
        if fallback is not None and len(fallback) > len(body):
            body = fallback
            body_source = "trafilatura"
    return {
        "body": body,
        "published_at": published,
        "modified_at": modified,
        "article_url": canonical,
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "body_source": body_source,
    }


def temporal_article_reason(article: dict[str, object], cutoff: datetime,
                            rss_published: datetime, *, min_body_chars: int = 400) -> str | None:
    body = article["body"]
    published = article["published_at"]
    modified = article["modified_at"]
    if not isinstance(body, str) or len(body) < min_body_chars:
        return "NO_RELIABLE_ARTICLE_BODY"
    if not isinstance(published, datetime) or not isinstance(modified, datetime):
        return "MISSING_PUBLISHER_VERSION_TIME"
    if published > modified or published > cutoff or modified > cutoff:
        return "ARTICLE_UPDATED_AFTER_CUTOFF"
    if rss_published > cutoff or abs((published - rss_published).total_seconds()) > 48 * 3600:
        return "RSS_PUBLISHER_TIME_CONFLICT"
    if not public_http_url(str(article["article_url"])):
        return "UNSAFE_ARTICLE_URL"
    return None
