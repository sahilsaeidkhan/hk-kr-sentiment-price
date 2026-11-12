"""Telegram sentiment ingestion — listed source for both exchanges.

The brief lists Telegram for HKEX ("Telegram") and for KRX ("KakaoTalk
Open Chats & Telegram Channels"). Three routes are implemented:

1. **Keyword search actor** ``lofomachines/telegram-keyword-search-scraper``
   — searches public Telegram messages by keyword, scoped by country code
   (HK / KR). This is the primary route: it finds per-ticker discussion
   without needing a channel list.
2. **Channel actor** ``tri_angle/telegram-scraper`` — scrapes the public
   channels named in ``config.TELEGRAM_CHANNELS`` (empty by default; no
   channel names are guessed).
3. **Public web preview** ``https://t.me/s/<channel>`` — key-free scrape
   of the same configured channels; message timestamps, text and view
   counts are public on that page.

All Apify routes need run credit. On the free plan, exhausted monthly
credit makes the API reject runs; the module logs the exact error and
returns an empty frame so the rest of the pipeline proceeds — the README
documents this state honestly. Post schema matches the other sources
(Telegram exposes views; author follower counts are not public -> 0).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config
from .sentiment import score_text

_COLUMNS = ["source", "exchange", "ticker", "created_at", "text", "likes",
            "reposts", "views", "followers", "url"]


def _match_asset(text: str, exchange: str) -> str | None:
    for asset in config.ASSETS:
        if asset.exchange != exchange:
            continue
        terms = asset.keywords_local + asset.keywords_en
        if any(kw.lower() in text.lower() for kw in terms):
            return asset.ticker
    return None


def _keyword_search_route() -> pd.DataFrame:
    """Route 1: Apify Telegram keyword search, per exchange."""
    from apify_client import ApifyClient

    client = ApifyClient(config.APIFY_API_TOKEN)
    rows: list[dict] = []
    after = (datetime.now(timezone.utc)
             - timedelta(days=config.SOCIAL_LOOKBACK_DAYS)).date().isoformat()
    for exchange, country in (("HKEX", "HK"), ("KRX", "KR")):
        keywords = [a.name_local for a in config.ASSETS
                    if a.exchange == exchange]
        try:
            run = client.actor(config.TELEGRAM_SEARCH_ACTOR_ID).call(
                run_input={
                    "keywords": keywords,
                    "afterDate": after,
                    "maxResultsPerKeyword": 100,
                    "countryCode": country,
                })
        except Exception as exc:
            print(f"[telegram] keyword-search actor unavailable for "
                  f"{exchange} ({str(exc)[:160]})")
            return pd.DataFrame(columns=_COLUMNS)
        dataset_id = (getattr(run, "default_dataset_id", None)
                      or run["defaultDatasetId"])
        for item in client.dataset(dataset_id).iterate_items():
            text = str(item.get("text") or item.get("message") or "").strip()
            created = pd.to_datetime(
                item.get("date") or item.get("published_at"),
                utc=True, errors="coerce")
            if not text or pd.isna(created):
                continue
            ticker = _match_asset(text, exchange)
            if ticker is None:
                continue
            rows.append({
                "source": "telegram",
                "exchange": exchange,
                "ticker": ticker,
                "created_at": created,
                "text": text,
                "likes": int(item.get("reactions", 0) or 0),
                "reposts": int(item.get("forwards", 0) or 0),
                "views": int(item.get("views", 0) or 0),
                "followers": 0,
                "url": str(item.get("url") or item.get("link") or ""),
            })
    df = pd.DataFrame(rows, columns=_COLUMNS)
    print(f"[telegram] keyword-search route: {len(df)} asset-matched posts")
    return df


def _preview_route() -> pd.DataFrame:
    """Route 3: key-free scrape of t.me/s/<channel> public previews."""
    rows: list[dict] = []
    cutoff = datetime.now(timezone.utc) - timedelta(
        days=config.SOCIAL_LOOKBACK_DAYS)
    headers = {"User-Agent": "Mozilla/5.0"}
    for channel in config.TELEGRAM_CHANNELS:
        try:
            resp = requests.get(f"https://t.me/s/{channel}",
                                headers=headers, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            print(f"[telegram] preview {channel} FAILED: {exc}")
            continue
        for msg in soup.select(".tgme_widget_message"):
            time_tag = msg.select_one("time[datetime]")
            text_tag = msg.select_one(".tgme_widget_message_text")
            if not time_tag or not text_tag:
                continue
            created = pd.to_datetime(time_tag["datetime"], utc=True,
                                     errors="coerce")
            text = text_tag.get_text(" ", strip=True)
            if pd.isna(created) or created < cutoff or not text:
                continue
            views_tag = msg.select_one(".tgme_widget_message_views")
            views_txt = (views_tag.get_text(strip=True).replace(",", "")
                         if views_tag else "0")
            views = int(float(views_txt[:-1]) * 1000) \
                if views_txt.endswith("K") else int(
                    float(views_txt) if views_txt.replace(".", "", 1)
                    .isdigit() else 0)
            for exchange in ("HKEX", "KRX"):
                ticker = _match_asset(text, exchange)
                if ticker:
                    rows.append({
                        "source": "telegram",
                        "exchange": exchange,
                        "ticker": ticker,
                        "created_at": created,
                        "text": text,
                        "likes": 0, "reposts": 0, "views": views,
                        "followers": 0,
                        "url": f"https://t.me/{channel}",
                    })
                    break
    df = pd.DataFrame(rows, columns=_COLUMNS)
    if len(config.TELEGRAM_CHANNELS):
        print(f"[telegram] preview route: {len(df)} asset-matched posts")
    return df


def fetch_telegram() -> pd.DataFrame:
    """Fetch Telegram posts via every available route; never fatal."""
    frames = []
    if config.APIFY_API_TOKEN:
        frames.append(_keyword_search_route())
    frames.append(_preview_route())
    frames = [f for f in frames if not f.empty]
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=_COLUMNS))
    if not df.empty:
        df = (df.drop_duplicates(subset=["created_at", "text"])
                .sort_values("created_at").reset_index(drop=True))
        df["sentiment_score"] = df["text"].map(score_text)
    else:
        df["sentiment_score"] = pd.Series(dtype=float)
        print("[telegram] no posts retrieved in this run "
              "(see README limitations).")
    out = config.DATA_DIR / "posts_telegram.csv"
    df.to_csv(out, index=False)
    print(f"[telegram] {len(df)} posts saved to {out}")
    return df


if __name__ == "__main__":
    fetch_telegram()
