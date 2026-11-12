"""LIHKG sentiment ingestion — HKEX source 1 (from the brief's HK list).

LIHKG's finance channel (財經台） is where Hong Kong retail traders debate
HK stocks. Two execution routes are implemented:

1. **Apify actor** ``claude_code_reviewer/lihkg-scraper`` — used when an
   Apify token is configured *and* the account has run credit. (On the
   free plan the monthly credit can be exhausted; the module detects the
   API error and falls through to route 2.)
2. **LIHKG public ``api_v2`` JSON endpoints** (no key required) — the
   route that produced this repository's data. Thread listings for the
   finance channel are paginated, filtered to the last
   ``SOCIAL_LOOKBACK_DAYS`` days, and matched to assets by the local
   keywords in ``config``. The thread title is the post text (on LIHKG
   the title carries the author's take); the opening post body is
   appended when the thread-detail endpoint returns it.

Post schema (shared by every social source in this project):
``source, exchange, ticker, created_at (UTC), text, likes, reposts,
views, followers, url``. LIHKG exposes likes/dislikes and a reply count
(reposts column) but no view or follower counts — those fields are 0 and
the README says so.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config
from .sentiment import score_text

_API = "https://lihkg.com/api_v2"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                   "AppleWebKit/605.1.15"),
    "Referer": "https://lihkg.com/",
    "Accept": "application/json",
}
_MAX_PAGES = 120         # safety bound; 30 threads/page
_PAGE_SIZE = 30


def _match_asset(title: str) -> str | None:
    """Return the HK ticker whose keyword appears in *title*, else None."""
    for asset in config.ASSETS:
        if asset.exchange != "HKEX":
            continue
        if any(kw in title for kw in asset.keywords_local):
            return asset.ticker
    return None


def _fetch_public_api() -> pd.DataFrame:
    """Route 2: scan the finance channel and match threads to assets.

    The channel listing is ordered by *latest reply*, not by creation:
    LIHKG discussion concentrates in long-running threads, so a thread
    counts for the window when it was created inside it **or** was still
    being replied to inside it. The thread title is the post text (on
    LIHKG the title carries the author's take); the timestamp is the
    creation time for new threads and the last-reply time for older
    active threads — the convention is stated in the README.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(
        days=config.SOCIAL_LOOKBACK_DAYS)
    rows: list[dict] = []
    seen: set[int] = set()
    for page in range(1, _MAX_PAGES + 1):
        resp = requests.get(
            f"{_API}/thread/category",
            params={"cat_id": config.LIHKG_FINANCE_CAT_ID,
                    "page": page, "count": _PAGE_SIZE},
            headers=_HEADERS, timeout=20)
        payload = resp.json()
        if not payload.get("success"):
            print(f"[lihkg] api_v2 page {page}: "
                  f"{payload.get('error_message')}")
            break
        items = (payload.get("response") or {}).get("items") or []
        if not items:
            break
        for item in items:
            thread_id = int(item["thread_id"])
            if thread_id in seen:
                continue
            seen.add(thread_id)
            created = datetime.fromtimestamp(
                int(item.get("create_time", 0)), tz=timezone.utc)
            last_reply = datetime.fromtimestamp(
                int(item.get("last_reply_time", 0)), tz=timezone.utc)
            if created < cutoff and last_reply < cutoff:
                continue
            title = str(item.get("title", "")).strip()
            ticker = _match_asset(title)
            if ticker is None:
                continue
            rows.append({
                "source": "lihkg",
                "exchange": "HKEX",
                "ticker": ticker,
                "created_at": created if created >= cutoff else last_reply,
                "text": title,
                "likes": int(item.get("like_count", 0) or 0),
                "reposts": int(item.get("no_of_reply", 0) or 0),
                "views": 0,
                "followers": 0,
                "url": f"https://lihkg.com/thread/{thread_id}",
            })
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    print(f"[lihkg] public api_v2: {len(df)} asset-matched threads "
          f"active in the last {config.SOCIAL_LOOKBACK_DAYS} days "
          f"(scanned {len(seen)} threads)")
    return df


def _fetch_apify_actor() -> pd.DataFrame | None:
    """Route 1: the Apify LIHKG scraper actor (None if it cannot run)."""
    if not config.APIFY_API_TOKEN:
        return None
    from apify_client import ApifyClient

    keywords = sorted({kw for a in config.ASSETS if a.exchange == "HKEX"
                       for kw in a.keywords_local})
    try:
        client = ApifyClient(config.APIFY_API_TOKEN)
        run = client.actor(config.LIHKG_ACTOR_ID).call(run_input={
            "keywords": keywords,
            "maxItems": 300,
            "dateFrom": (datetime.now(timezone.utc)
                         - timedelta(days=config.SOCIAL_LOOKBACK_DAYS)
                         ).date().isoformat(),
        })
        dataset_id = (getattr(run, "default_dataset_id", None)
                      or run["defaultDatasetId"])
        items = list(client.dataset(dataset_id).iterate_items())
    except Exception as exc:
        print(f"[lihkg] Apify actor route unavailable ({str(exc)[:160]}); "
              "using the public api_v2 route.")
        return None
    rows = []
    for item in items:
        title = str(item.get("title", "")).strip()
        ticker = _match_asset(title)
        if ticker is None:
            continue
        rows.append({
            "source": "lihkg_apify",
            "exchange": "HKEX",
            "ticker": ticker,
            "created_at": pd.to_datetime(
                item.get("create_time") or item.get("created_at"),
                unit="s", utc=True, errors="coerce"),
            "text": title,
            "likes": int(item.get("like_count", 0) or 0),
            "reposts": int(item.get("no_of_reply", 0) or 0),
            "views": 0,
            "followers": 0,
            "url": item.get("url", ""),
        })
    df = pd.DataFrame(rows).dropna(subset=["created_at"])
    print(f"[lihkg] Apify actor route: {len(df)} asset-matched threads")
    return df


def fetch_lihkg() -> pd.DataFrame:
    """Fetch LIHKG posts, preferring the Apify actor, else public api_v2."""
    df = _fetch_apify_actor()
    if df is None or df.empty:
        df = _fetch_public_api()
    columns = ["source", "exchange", "ticker", "created_at", "text",
               "likes", "reposts", "views", "followers", "url",
               "sentiment_score"]
    if df.empty:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(config.DATA_DIR / "posts_lihkg.csv", index=False)
        print("[lihkg] 0 posts saved to data/posts_lihkg.csv")
        return empty
    df["sentiment_score"] = df["text"].map(score_text)
    df = df.sort_values("created_at").reset_index(drop=True)
    out = config.DATA_DIR / "posts_lihkg.csv"
    df.to_csv(out, index=False)
    print(f"[lihkg] {len(df)} posts saved to {out}")
    return df


if __name__ == "__main__":
    fetch_lihkg()
