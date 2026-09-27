"""Google News URL decode: paced signature GETs + batched batchexecute POST.

Per article still needs its own HTML signature fetch. The rate-limited
``batchexecute`` RPC accepts many ``Fbv4je`` / ``garturlreq`` calls in one POST,
so N URLs cost ~N GETs + ceil(N/batch_size) POSTs instead of 2N round-trips.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from google_news_api.exceptions import HTTPError, RateLimitError

_BATCH_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
_ARTICLE_BASE = "https://news.google.com/rss/articles/"
_SIG_RE = re.compile(r'data-n-a-sg="([^"]+)"')
_TS_RE = re.compile(r'data-n-a-ts="([^"]+)"')
# google_news_api.decode_url uses this fingerprint. A Chrome UA typically lands
# on consent.google.com without data-n-a-sg (even with follow_redirects).
_DECODE_HEADERS = {
    "User-Agent": "python-requests/2.32.3",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

_GET_LOCK = threading.Lock()
_LAST_GET_MONOTONIC = 0.0
_POST_LOCK = threading.Lock()
_LAST_POST_MONOTONIC = 0.0



def is_google_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, HTTPError) and getattr(exc, "status_code", None) == 429:
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _retry_after_seconds(exc: BaseException, attempt: int) -> float:
    if isinstance(exc, RateLimitError) and exc.retry_after is not None:
        return max(float(exc.retry_after), 1.0)
    if isinstance(exc, httpx.HTTPStatusError):
        raw = exc.response.headers.get("Retry-After")
        if raw:
            try:
                return max(float(raw), 1.0)
            except ValueError:
                pass
    return min(60.0, 2.0 ** max(attempt, 1))


def require_publisher_url(publisher_url: str) -> str:
    host = urlparse(publisher_url).hostname or ""
    if (
        urlparse(publisher_url).scheme not in ("http", "https")
        or not host
        or host == "google.com"
        or host.endswith(".google.com")
        or host == "googleusercontent.com"
        or host.endswith(".googleusercontent.com")
    ):
        raise ValueError(
            "ARTICLE_URL_UNRESOLVED: decoder did not return a publisher URL"
        )
    return str(publisher_url)


def _article_id(rss_link: str) -> str:
    path = urlparse(rss_link).path.split("/")
    if len(path) < 4 or path[1] != "rss" or path[2] != "articles":
        raise ValueError(f"ARTICLE_URL_UNRESOLVED: not a Google News RSS article URL")
    return path[-1]


def _pace(lock: threading.Lock, last_attr: str, min_interval_sec: float) -> None:
    global _LAST_GET_MONOTONIC, _LAST_POST_MONOTONIC
    with lock:
        if last_attr == "get":
            last = _LAST_GET_MONOTONIC
        elif last_attr == "post":
            last = _LAST_POST_MONOTONIC
        else:
            raise ValueError(f"Unknown pace target: {last_attr}")
        gap = float(min_interval_sec) - (time.monotonic() - last)
        if gap > 0:
            time.sleep(gap)
        now = time.monotonic()
        if last_attr == "get":
            _LAST_GET_MONOTONIC = now
        elif last_attr == "post":
            _LAST_POST_MONOTONIC = now


def _fetch_signature(
    client: httpx.Client,
    rss_link: str,
    *,
    country: str,
    language: str,
    get_min_interval_sec: float,
    timeout: float,
) -> tuple[str, str, str]:
    """Return (article_id, timestamp, signature)."""
    article_id = _article_id(rss_link)
    language_base = language.split("-")[0]
    language_full = language if "-" in language else f"{language}-{country}"
    _pace(_GET_LOCK, "get", get_min_interval_sec)
    response = client.get(
        f"{_ARTICLE_BASE}{article_id}",
        params={
            "hl": language_full,
            "gl": country,
            "ceid": f"{country}:{language_base}",
        },
        headers=_DECODE_HEADERS,
        timeout=timeout,
    )
    if response.status_code == 429:
        raise HTTPError("HTTP 429: Too Many Requests", status_code=429)
    if not (200 <= response.status_code < 400):
        raise HTTPError(
            f"HTTP {response.status_code}: signature page",
            status_code=response.status_code,
        )
    if "consent.google.com" in str(response.url):
        raise ValueError(
            "ARTICLE_URL_UNRESOLVED: Google consent interstitial on signature page"
        )
    sig = _SIG_RE.search(response.text)
    ts = _TS_RE.search(response.text)
    if not sig or not ts:
        raise ValueError(
            "ARTICLE_URL_UNRESOLVED: missing decode signature on Google News page"
        )
    return article_id, ts.group(1), sig.group(1)


def _garturl_rpc(
    article_id: str,
    timestamp: str,
    signature: str,
    *,
    country: str,
    language_base: str,
) -> list[Any]:
    payload = (
        "["
        '"garturlreq",'
        "["
        '["X","X",["X","X"],null,null,1,1,'
        f'"{country}:{language_base}",'
        "null,1,null,null,null,null,null,0,1],"
        '"X","X",1,[1,1,1],1,1,null,0,0,null,0],'
        f'"{article_id}",{timestamp},"{signature}"'
        "]"
    )
    return ["Fbv4je", payload]


def _parse_batch_response(text: str, n: int) -> list[str | None]:
    try:
        parsed = json.loads(text.split("\n\n", 1)[1])[:-2]
    except (json.JSONDecodeError, IndexError) as exc:
        raise ValueError(
            f"ARTICLE_URL_UNRESOLVED: bad batchexecute response: {exc}"
        ) from exc
    out: list[str | None] = []
    for i in range(n):
        try:
            out.append(str(json.loads(parsed[i][2])[1]))
        except (IndexError, TypeError, json.JSONDecodeError, KeyError):
            out.append(None)
    return out


def _batchexecute_post(
    client: httpx.Client,
    rpcs: list[list[Any]],
    *,
    post_min_interval_sec: float,
    timeout: float,
    max_retries: int,
) -> list[str | None]:
    attempts = max(0, int(max_retries)) + 1
    body = f"f.req={quote(json.dumps([rpcs]))}"
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        _pace(_POST_LOCK, "post", post_min_interval_sec)
        try:
            response = client.post(
                _BATCH_URL,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                },
                content=body,
                timeout=timeout,
            )
            if response.status_code == 429:
                raise HTTPError("HTTP 429: Too Many Requests", status_code=429)
            response.raise_for_status()
            return _parse_batch_response(response.text, len(rpcs))
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= attempts or not is_google_rate_limit(exc):
                raise
            time.sleep(_retry_after_seconds(exc, attempt + 1))
    raise RuntimeError(f"batchexecute unreachable: {last_exc}")


def decode_publisher_urls_batch(
    rss_links: list[str],
    *,
    country: str = "US",
    language: str = "en",
    get_min_interval_sec: float = 0.75,
    get_concurrency: int = 4,
    post_min_interval_sec: float = 1.0,
    batch_size: int = 50,
    max_retries: int = 5,
    timeout: float = 20.0,
) -> dict[str, str | BaseException]:
    """Decode many RSS article URLs. Values are publisher URL or the failure exception."""
    if not rss_links:
        return {}
    language_base = language.split("-")[0]
    results: dict[str, str | BaseException] = {}
    # Deduplicate while preserving first-seen order.
    unique: list[str] = list(dict.fromkeys(rss_links))

    params: dict[str, tuple[str, str, str]] = {}
    errors: dict[str, BaseException] = {}

    def _one(link: str) -> None:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                params[link] = _fetch_signature(
                    client,
                    link,
                    country=country,
                    language=language,
                    get_min_interval_sec=get_min_interval_sec,
                    timeout=timeout,
                )
        except Exception as exc:
            errors[link] = exc

    workers = max(1, min(int(get_concurrency), len(unique)))
    if workers == 1:
        for link in unique:
            _one(link)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, link) for link in unique]
            for fut in as_completed(futs):
                fut.result()

    for link, exc in errors.items():
        results[link] = exc

    ready_links = [link for link in unique if link in params]
    chunk = max(1, int(batch_size))
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for i in range(0, len(ready_links), chunk):
            group = ready_links[i : i + chunk]
            rpcs = [
                _garturl_rpc(
                    *params[link],
                    country=country,
                    language_base=language_base,
                )
                for link in group
            ]
            try:
                decoded = _batchexecute_post(
                    client,
                    rpcs,
                    post_min_interval_sec=post_min_interval_sec,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            except Exception as exc:
                for link in group:
                    results[link] = exc
                continue
            for link, publisher in zip(group, decoded):
                if not publisher:
                    results[link] = ValueError(
                        "ARTICLE_URL_UNRESOLVED: empty batchexecute result"
                    )
                    continue
                try:
                    results[link] = require_publisher_url(publisher)
                except Exception as exc:
                    results[link] = exc
    return results
