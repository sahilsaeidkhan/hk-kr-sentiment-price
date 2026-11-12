"""Per-ticker extraction — the brief's deliverable #3.

One row per asset with: ticker, asset name, current price, aggregate
sentiment (social, last ``SOCIAL_LOOKBACK_DAYS`` days, engagement-weighted
across sources), sentiment direction, the model's position stance for
n+1, and support / resistance levels.

Support/resistance method (stated in the README): classic floor-trader
pivot points from the most recent completed month of bars give
``pivot``, ``r1``, ``s1``; swing highs/lows (local extrema over a
5-session window, last 60 sessions) refine them — resistance is the
nearest swing high above the current price (fallback: R1), support the
nearest swing low below it (fallback: S1).

"Position" is the pipeline's own model stance from the long/short rule
applied to the n+1 prediction (LONG / SHORT / FLAT) — a model output,
not investment advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def support_resistance(price: pd.DataFrame) -> dict:
    """Nearest support/resistance for one ticker's price history."""
    d = price.sort_values("date").reset_index(drop=True)
    current = float(d["close"].iloc[-1])
    recent = d.tail(21)                       # ~one trading month
    high, low, close_prev = (float(recent["high"].max()),
                             float(recent["low"].min()),
                             float(recent["close"].iloc[-2]))
    pivot = (high + low + close_prev) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    window = d.tail(60)
    highs, lows = window["high"].values, window["low"].values
    swing_highs = [highs[i] for i in range(2, len(highs) - 2)
                   if highs[i] == max(highs[i - 2:i + 3])]
    swing_lows = [lows[i] for i in range(2, len(lows) - 2)
                  if lows[i] == min(lows[i - 2:i + 3])]
    above = [h for h in swing_highs if h > current]
    below = [l for l in swing_lows if l < current]
    resistance = float(min(above)) if above else float(r1)
    support = float(max(below)) if below else float(s1)
    return {"support": support, "resistance": resistance,
            "pivot": float(pivot)}


def _direction(score: float) -> str:
    if score > 0.05:
        return "BULLISH"
    if score < -0.05:
        return "BEARISH"
    return "NEUTRAL"


def build_extraction(prices: pd.DataFrame, posts: pd.DataFrame,
                     news: pd.DataFrame,
                     predictions: pd.DataFrame) -> pd.DataFrame:
    """Assemble the per-ticker extraction table."""
    pred_by_ticker = predictions.set_index("ticker")
    rows: list[dict] = []
    for asset in config.ASSETS:
        price = prices[prices["ticker"] == asset.ticker]
        if price.empty:
            continue
        latest = price.sort_values("date").iloc[-1]
        asset_posts = (posts[posts["ticker"] == asset.ticker]
                       if posts is not None else pd.DataFrame())
        if len(asset_posts):
            weights = 1.0 + asset_posts["likes"] + asset_posts["reposts"]
            sentiment = float(np.average(asset_posts["sentiment_score"],
                                         weights=weights))
            mentions = int(len(asset_posts))
            sources = ", ".join(sorted(asset_posts["source"].unique()))
        else:
            sentiment, mentions, sources = 0.0, 0, ""
        headlines = (int((news["ticker"] == asset.ticker).sum())
                     if news is not None and len(news) else 0)
        sr = support_resistance(price)
        pred = pred_by_ticker.loc[asset.ticker]
        rows.append({
            "ticker": asset.ticker,
            "asset_name": asset.name_en,
            "name_local": asset.name_local,
            "exchange": asset.exchange,
            "currency": asset.currency,
            "current_price": round(float(latest["close"]), 3),
            "price_date": str(latest["date"].date()),
            "social_mentions": mentions,
            "sentiment_score": round(sentiment, 3),
            "sentiment_direction": _direction(sentiment),
            "sources": sources,
            "news_headlines": headlines,
            "model_direction_nplus1": pred["predicted_direction"],
            "predicted_return_nplus1": round(
                float(pred["predicted_return"]), 5),
            "position": {"UP": "LONG", "DOWN": "SHORT"}.get(
                pred["predicted_direction"], "FLAT"),
            "support": round(sr["support"], 3),
            "resistance": round(sr["resistance"], 3),
            "pivot": round(sr["pivot"], 3),
        })
    df = pd.DataFrame(rows)
    out = config.OUTPUTS_DIR / "per_ticker_extraction.csv"
    df.to_csv(out, index=False)
    print(f"[extract] per-ticker extraction saved to {out}")
    return df
