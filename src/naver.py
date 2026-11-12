"""Naver Finance discussion-board ingestion — KRX source 1 (brief's list).

Naver Finance's per-stock discussion boards (종목토론) are the centre of
Korean retail-investor sentiment. The boards are publicly readable in a
browser; Naver has, however, migrated them into a JavaScript application
whose post list is no longer present in the served HTML, and the JSON
endpoints the app calls are undocumented.

This module therefore tries, in order:

1. The classic board URL (``finance.naver.com/item/board.naver``) parsed
   with BeautifulSoup — this works whenever Naver serves the classic
   server-rendered table (it still does for some user agents/regions).
2. A small set of candidate JSON endpoints on ``api.stock.naver.com`` /
   ``m.stock.naver.com`` that the current app is known to call.

Every attempt is logged. If none returns posts, the function returns an
empty frame and the pipeline continues with the remaining sources — the
limitation is documented in the README rather than hidden. Post schema
matches the other social sources; Naver boards expose a view count and
like/dislike counts but no follower counts.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config
from .sentiment import score_text

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Referer": "https://finance.naver.com/",
}
_JSON_CANDIDATES = (
    "https://api.stock.naver.com/discussion/stock/{code}?page={page}&pageSize=20",
    "https://m.stock.naver.com/api/discussion/domesticStock/{code}?page={page}&pageSize=20",
    "https://api.stock.naver.com/discussion/domesticStock/{code}?page={page}",
)
_DATE_RE = re.compile(r"(\d{4})[.\-/](\d{2})[.\-/](\d{2})\s+(\d{2}):(\d{2})")


def _parse_classic_board(html: str, ticker: str) -> list[dict]:
    """Parse the classic server-rendered board table, if present."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    for tr in soup.select("table.type5 tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue
        title_cell = cells[1]
        link = title_cell.find("a")
        title = (link.get("title") if link else title_cell.get_text(" ")
                 ).strip()
        date_match = _DATE_RE.search(cells[2].get_text(" ", strip=True))
        if not title or not date_match:
            continue
        y, mo, d, hh, mm = (int(g) for g in date_match.groups())
        created = datetime(y, mo, d, hh, mm,
                           tzinfo=timezone(timedelta(hours=9)))
        def _num(idx: int) -> int:
            txt = cells[idx].get_text(strip=True).replace(",", "")
            return int(txt) if txt.isdigit() else 0
        rows.append({
            "source": "naver",
            "exchange": "KRX",
            "ticker": ticker,
            "created_at": created.astimezone(timezone.utc),
            "text": title,
            "likes": _num(5) if len(cells) > 5 else 0,
            "reposts": 0,
            "views": _num(3),
            "followers": 0,
            "url": ("https://finance.naver.com/item/board.naver?code="
                    f"{ticker.split('.')[0]}"),
        })
    return rows


def _try_json_endpoints(code: str, ticker: str) -> list[dict]:
    """Probe the candidate JSON endpoints once each; return any posts."""
    for template in _JSON_CANDIDATES:
        url = template.format(code=code, page=1)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=12)
            if resp.status_code != 200:
                continue
            payload = resp.json()
        except Exception:
            continue
        items = (payload.get("posts") or payload.get("items")
                 or payload.get("result") or [])
        rows = []
        for item in items if isinstance(items, list) else []:
            title = str(item.get("title") or item.get("content") or "")
            stamp = (item.get("createDate") or item.get("createdAt")
                     or item.get("regTime"))
            created = pd.to_datetime(stamp, utc=True, errors="coerce")
            if not title or pd.isna(created):
                continue
            rows.append({
                "source": "naver",
                "exchange": "KRX",
                "ticker": ticker,
                "created_at": created,
                "text": title,
                "likes": int(item.get("likeCount", 0) or 0),
                "reposts": 0,
                "views": int(item.get("viewCount", 0) or 0),
                "followers": 0,
                "url": url,
            })
        if rows:
            print(f"[naver] {ticker}: JSON endpoint served {len(rows)} posts")
            return rows
    return []


def fetch_naver() -> pd.DataFrame:
    """Fetch discussion-board posts for every KRX asset (best effort)."""
    cutoff = datetime.now(timezone.utc) - timedelta(
        days=config.SOCIAL_LOOKBACK_DAYS)
    rows: list[dict] = []
    for asset in config.ASSETS:
        if asset.exchange != "KRX":
            continue
        got: list[dict] = []
        for page in range(1, 4):  # ~20 posts/page; stop early when stale
            url = config.NAVER_BOARD_URL.format(code=asset.code, page=page)
            try:
                resp = requests.get(url, headers=_HEADERS, timeout=20)
                page_rows = _parse_classic_board(resp.text, asset.ticker)
            except Exception as exc:
                print(f"[naver] {asset.ticker} page {page} FAILED: {exc}")
                page_rows = []
            if not page_rows:
                break
            got.extend(page_rows)
            if min(r["created_at"] for r in page_rows) < cutoff:
                break
        if not got:
            got = _try_json_endpoints(asset.code, asset.ticker)
        got = [r for r in got if r["created_at"] >= cutoff]
        print(f"[naver] {asset.ticker}: {len(got)} posts in window")
        rows.extend(got)
    df = pd.DataFrame(rows, columns=[
        "source", "exchange", "ticker", "created_at", "text", "likes",
        "reposts", "views", "followers", "url"])
    if not df.empty:
        df["sentiment_score"] = df["text"].map(score_text)
        df = df.sort_values("created_at").reset_index(drop=True)
    else:
        df["sentiment_score"] = pd.Series(dtype=float)
        print("[naver] no posts retrieved — Naver served no parseable "
              "board content in this run (see README limitations).")
    out = config.DATA_DIR / "posts_naver.csv"
    df.to_csv(out, index=False)
    print(f"[naver] {len(df)} posts saved to {out}")
    return df


if __name__ == "__main__":
    fetch_naver()
