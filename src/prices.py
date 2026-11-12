"""Daily price ingestion for all configured assets (Yahoo Finance).

No API key is required. Nine months of daily OHLCV are pulled per ticker
(the brief requires at least six for out-of-sample validation). Each
ticker is saved to its own CSV and all tickers are combined into
``data/prices.csv`` with a ``ticker`` column.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from . import config


def _fetch_one(ticker: str) -> pd.DataFrame:
    """Download one ticker's daily bars and normalise the schema."""
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(config.PRICE_HISTORY_MONTHS * 31))
    raw = yf.download(
        ticker, start=start.isoformat(), end=end.isoformat(),
        auto_adjust=False, progress=False, timeout=30,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["ticker"] = ticker
    df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
    # Yahoo can append a stub row for the current (unfinished) session
    # with NaN prices — drop anything without a real close.
    return df.dropna(subset=["close"]).reset_index(drop=True)


def fetch_prices() -> pd.DataFrame:
    """Fetch every configured asset; failures are logged, not fatal."""
    frames: list[pd.DataFrame] = []
    for asset in config.ASSETS:
        try:
            df = _fetch_one(asset.ticker)
        except Exception as exc:  # keep the run alive for the other tickers
            print(f"[prices] {asset.ticker} FAILED: {exc}")
            continue
        df.to_csv(config.DATA_DIR / f"prices_{asset.ticker}.csv", index=False)
        print(f"[prices] {asset.ticker}: {len(df)} rows "
              f"{df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
        frames.append(df)
    if not frames:
        raise RuntimeError("no price data could be fetched for any ticker")
    combined = (pd.concat(frames, ignore_index=True)
                  .sort_values(["ticker", "date"]).reset_index(drop=True))
    combined.to_csv(config.DATA_DIR / "prices.csv", index=False)
    print(f"[prices] combined: {len(combined)} rows across "
          f"{combined['ticker'].nunique()} tickers -> data/prices.csv")
    return combined


if __name__ == "__main__":
    fetch_prices()
