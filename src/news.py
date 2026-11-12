"""Per-asset news ingestion (context + a daily news-sentiment feature).

Source order, per the brief's specification:

1. **Exa** (``exa.com``) — the brief's named news source; used whenever
   ``EXA_API_KEY`` is configured, via the Exa REST search API with
   per-asset queries restricted to the last ``NEWS_LOOKBACK_DAYS`` days.
2. **Tavily** — fallback search API, used when no Exa key is configured
   (this repository's executed run: no EXA_API_KEY was present locally,
   a Tavily key was; the substitution is documented in the README).

For every asset the most recent/relevant headlines are stored with
timestamps, and each headline is scored with the same multilingual
sentiment scorer as the social posts, so news can complement social
sentiment in the feature set (daily-aggregated, coverage-flagged).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config
from .sentiment import score_text

_COLUMNS = ["ticker", "exchange", "source", "published_at", "title", "url",
            "sentiment_score"]


def _fetch_exa(asset: config.Asset, since: datetime) -> list[dict]:
    resp = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": config.EXA_API_KEY,
                 "Content-Type": "application/json"},
        json={
            "query": f"{asset.name_en} stock news",
            "type": "news",
            "numResults": 5,
            "startPublishedDate": since.isoformat(),
            "contents": {"text": False},
        }, timeout=30)
    resp.raise_for_status()
    return [{
        "ticker": asset.ticker, "exchange": asset.exchange, "source": "exa",
        "published_at": r.get("publishedDate"),
        "title": r.get("title", ""), "url": r.get("url", ""),
    } for r in resp.json().get("results", [])]


def _fetch_tavily(asset: config.Asset) -> list[dict]:
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "query": f"{asset.name_en} ({asset.ticker}) stock news",
            "topic": "news",
            "days": config.NEWS_LOOKBACK_DAYS,
            "max_results": 5,
            "search_depth": "basic",
        }, timeout=30)
    resp.raise_for_status()
    return [{
        "ticker": asset.ticker, "exchange": asset.exchange,
        "source": "tavily",
        "published_at": r.get("published_date"),
        "title": r.get("title", ""), "url": r.get("url", ""),
    } for r in resp.json().get("results", [])]


def fetch_news() -> pd.DataFrame:
    """Fetch recent news for every asset from the first available API."""
    since = datetime.now(timezone.utc) - timedelta(
        days=config.NEWS_LOOKBACK_DAYS)
    if config.EXA_API_KEY:
        engine, fetch = "exa", lambda a: _fetch_exa(a, since)
    elif config.TAVILY_API_KEY:
        engine, fetch = "tavily", _fetch_tavily
        print("[news] EXA_API_KEY not configured — using the Tavily "
              "fallback (documented in README).")
    else:
        print("[news] no news API key configured — skipping news.")
        df = pd.DataFrame(columns=_COLUMNS)
        df.to_csv(config.DATA_DIR / "news.csv", index=False)
        return df
    rows: list[dict] = []
    for asset in config.ASSETS:
        try:
            got = fetch(asset)
        except Exception as exc:
            print(f"[news] {asset.ticker} FAILED via {engine}: "
                  f"{str(exc)[:140]}")
            continue
        rows.extend(got)
        print(f"[news] {asset.ticker}: {len(got)} headlines via {engine}")
    df = pd.DataFrame(rows, columns=_COLUMNS[:-1])
    if df.empty:
        df["sentiment_score"] = pd.Series(dtype=float)
        df = df[_COLUMNS]
    else:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True,
                                            errors="coerce")
        df = df.dropna(subset=["published_at", "title"])
        df["sentiment_score"] = df["title"].map(score_text)
        df = df.sort_values("published_at", ascending=False)
    out = config.DATA_DIR / "news.csv"
    df.to_csv(out, index=False)
    print(f"[news] {len(df)} headlines saved to {out}")
    return df


if __name__ == "__main__":
    fetch_news()
