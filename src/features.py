"""Dataset construction: join daily sentiment with daily price bars.

The join is persisted in SQLite (``data/dataset.db``: ``price_daily``,
``sentiment_daily``, ``dataset_joined``) and exported to
``data/dataset.csv`` — one row per (ticker, trading day).

Daily sentiment aggregation (per ticker, sources combined by post count):
``sent_mean``, ``bull_ratio`` (share of posts scoring > 0.05),
``post_volume``, ``engagement`` (likes + reposts). News headlines are
aggregated the same way into ``news_sent`` / ``news_count``.

No-lookahead rules (auditable, same discipline as Task 1):
* features for day *t* use only posts/prices dated <= *t*;
* posts are bucketed to their **UTC calendar date** and joined onto the
  ticker's trading days — a post lands on the first trading day on or
  after its UTC date, so weekend/holiday chatter informs the next
  session, never a past one;
* unobserved days carry sentiment 0 with ``sent_observed = 0`` — coverage
  is explicit, never silently imputed.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import config


def _daily_social(posts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate posts to (ticker, UTC date) sentiment statistics."""
    if posts is None or posts.empty:
        return pd.DataFrame(columns=["ticker", "date", "sent_mean",
                                     "bull_ratio", "post_volume",
                                     "engagement"])
    d = posts.copy()
    d["date"] = (pd.to_datetime(d["created_at"], utc=True)
                   .dt.tz_localize(None).dt.normalize())
    return (d.groupby(["ticker", "date"]).agg(
        sent_mean=("sentiment_score", "mean"),
        bull_ratio=("sentiment_score", lambda s: float((s > 0.05).mean())),
        post_volume=("sentiment_score", "size"),
        engagement=("likes", "sum"),
        _reposts=("reposts", "sum"),
    ).reset_index()
        .assign(engagement=lambda g: g["engagement"] + g["_reposts"])
        .drop(columns="_reposts"))


def _daily_news(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headlines to (ticker, UTC date) news sentiment."""
    if news is None or news.empty:
        return pd.DataFrame(columns=["ticker", "date", "news_sent",
                                     "news_count"])
    d = news.copy()
    d["date"] = (pd.to_datetime(d["published_at"], utc=True)
                   .dt.tz_localize(None).dt.normalize())
    return (d.groupby(["ticker", "date"]).agg(
        news_sent=("sentiment_score", "mean"),
        news_count=("sentiment_score", "size"),
    ).reset_index())


def _roll_to_trading_days(daily: pd.DataFrame,
                          trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Re-bucket calendar-day aggregates onto a ticker's trading days.

    Each calendar date maps to the first trading day >= that date; rows
    mapping to the same trading day are averaged (sentiment) / summed
    (counts), which preserves the no-lookahead property.
    """
    if daily.empty:
        return daily
    pos = np.searchsorted(trading_days.values,
                          daily["date"].values.astype("datetime64[ns]"))
    keep = pos < len(trading_days)
    d = daily.loc[keep].copy()
    d["date"] = trading_days[pos[keep]]
    agg = {"sent_mean": "mean", "bull_ratio": "mean", "post_volume": "sum",
           "engagement": "sum", "news_sent": "mean", "news_count": "sum"}
    cols = {k: v for k, v in agg.items() if k in d.columns}
    return d.groupby(["ticker", "date"], as_index=False).agg(cols)


def build_dataset(prices: pd.DataFrame, posts: pd.DataFrame,
                  news: pd.DataFrame) -> pd.DataFrame:
    """Join prices with daily social + news sentiment, per ticker."""
    social = _daily_social(posts)
    news_daily = _daily_news(news)
    frames: list[pd.DataFrame] = []
    for ticker, price in prices.groupby("ticker"):
        days = pd.DatetimeIndex(sorted(price["date"].unique()))
        soc = _roll_to_trading_days(social[social["ticker"] == ticker], days)
        nws = _roll_to_trading_days(news_daily[news_daily["ticker"] == ticker],
                                    days)
        df = price.merge(soc, on=["ticker", "date"], how="left")
        df = df.merge(nws, on=["ticker", "date"], how="left")
        df["sent_observed"] = df["post_volume"].notna().astype(int)
        df["news_observed"] = df["news_count"].notna().astype(int)
        for col in ("sent_mean", "bull_ratio", "news_sent"):
            df[col] = df[col].fillna(0.0)
        for col in ("post_volume", "engagement", "news_count"):
            df[col] = df[col].fillna(0).astype(int)
        frames.append(df)
    joined = (pd.concat(frames, ignore_index=True)
                .sort_values(["ticker", "date"]).reset_index(drop=True))
    with sqlite3.connect(config.DATA_DIR / "dataset.db") as con:
        prices.to_sql("price_daily", con, if_exists="replace", index=False)
        social.to_sql("sentiment_daily", con, if_exists="replace",
                      index=False)
        joined.to_sql("dataset_joined", con, if_exists="replace",
                      index=False)
    joined.to_csv(config.DATA_DIR / "dataset.csv", index=False)
    cov = joined.groupby("ticker")["sent_observed"].mean()
    print("[dataset] SQLite join saved (data/dataset.db); social-sentiment "
          "coverage by ticker:")
    for ticker, share in cov.items():
        print(f"    {ticker}: {share:.0%} of trading days")
    return joined


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price + sentiment features and the n+1 target, per ticker.

    Target convention: the forecast origin is the close of day *t*; the
    single target is the next-trading-day return ``close(t+1)/close(t)-1``.
    """
    out: list[pd.DataFrame] = []
    for _, d in df.groupby("ticker"):
        d = d.copy().sort_values("date").reset_index(drop=True)
        close = d["close"]
        ret1 = close.pct_change()
        for k in (1, 3, 5, 7):
            d[f"ret_{k}d"] = close.pct_change(k)
        d["vol_7d"] = ret1.rolling(7).std()
        d["vol_14d"] = ret1.rolling(14).std()
        d["sent_ma3"] = d["sent_mean"].rolling(3).mean()
        d["sent_ma7"] = d["sent_mean"].rolling(7).mean()
        d["sent_momentum"] = d["sent_ma3"] - d["sent_ma7"]
        d["bull_ma7"] = d["bull_ratio"].rolling(7).mean()
        d["news_ma3"] = d["news_sent"].rolling(3).mean()

        def _z30(s: pd.Series) -> pd.Series:
            z = (s - s.rolling(30).mean()) / s.rolling(30).std()
            return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        d["postvol_z30"] = _z30(d["post_volume"].astype(float))
        d["pricevol_z30"] = _z30(np.log1p(d["volume"].astype(float)))
        d["dow"] = d["date"].dt.dayofweek
        d["target_next"] = close.shift(-1) / close - 1.0
        d["target_date"] = d["date"].shift(-1)
        out.append(d)
    return pd.concat(out, ignore_index=True)


FEATURE_COLUMNS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_7d", "vol_7d", "vol_14d",
    "sent_mean", "sent_ma3", "sent_ma7", "sent_momentum",
    "bull_ratio", "bull_ma7", "post_volume", "postvol_z30",
    "news_sent", "news_ma3", "pricevol_z30", "dow",
    "sent_observed", "news_observed",
]
