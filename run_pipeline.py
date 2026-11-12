"""End-to-end pipeline for the CrowdWisdomTrading (TZURONI) Task 2 brief.

    python run_pipeline.py

Stages:
  1. Prices        — 9 months of daily OHLCV for 5 HKEX + 5 KRX tickers
                     (Yahoo Finance, no key required).
  2. Social posts  — LIHKG finance channel (HKEX) and Naver Finance
                     discussion boards (KRX) from the brief's source
                     lists, plus Telegram for both markets. Each source
                     is attempted independently; a source that cannot
                     run logs why and the pipeline continues.
  3. News          — per-asset headlines via Exa (Tavily fallback),
                     scored with the same sentiment scorer.
  4. Dataset       — daily sentiment joined to price bars in SQLite,
                     with explicit coverage flags (no hidden imputation).
  5. Model         — pooled RandomForest on price + sentiment features,
                     expanding-window walk-forward validation (no
                     shuffling, no lookahead), long/short strategy
                     evaluation on out-of-sample predictions only.
  6. Deliverables  — per-ticker extraction, frozen n+1 predictions,
                     metrics, and the self-contained visual report.

Every number in the report is computed by this run from the files in
``data/``. API keys are read from ``.env`` only and never printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (config, extract, features, lihkg, model, naver,  # noqa: E402
                 news, prices, report, telegram_apify)


def _coverage_note(posts_by_source: dict[str, int],
                   news_count: int) -> str:
    """One-paragraph, factual coverage summary for the report header."""
    parts = [f"{name}: {count} posts" for name, count in
             posts_by_source.items()]
    return ("Social posts collected for the last "
            f"{config.SOCIAL_LOOKBACK_DAYS} days — "
            + (", ".join(parts) if parts else "none")
            + f". News headlines: {news_count}. Prices: 9 months of daily "
              "bars per ticker via Yahoo Finance. Sources that returned "
              "no posts in this run are documented in the README "
              "(limitations), not silently dropped.")


def main() -> None:
    price = prices.fetch_prices()

    social_frames: dict[str, pd.DataFrame] = {}
    for name, fetch in (("lihkg", lihkg.fetch_lihkg),
                        ("naver", naver.fetch_naver),
                        ("telegram", telegram_apify.fetch_telegram)):
        try:
            social_frames[name] = fetch()
        except Exception as exc:  # a dead source must not kill the run
            print(f"[{name}] FAILED with an unexpected error: {exc}")
            social_frames[name] = pd.DataFrame()
    posts = pd.concat(
        [f for f in social_frames.values() if f is not None],
        ignore_index=True) if social_frames else pd.DataFrame()
    if not posts.empty:
        posts = posts.sort_values("created_at").reset_index(drop=True)
        posts.to_csv(config.DATA_DIR / "posts_all.csv", index=False)
        print(f"[social] {len(posts)} posts total -> data/posts_all.csv")

    news_df = news.fetch_news()

    joined = features.build_dataset(price, posts, news_df)
    featured = features.add_features(joined)
    featured.to_csv(config.DATA_DIR / "features.csv", index=False)

    metrics, oos_preds = model.walk_forward(featured)
    metrics.to_csv(config.OUTPUTS_DIR / "walk_forward_metrics.csv",
                   index=False)
    oos_preds.to_csv(config.OUTPUTS_DIR / "oos_predictions.csv", index=False)
    strategy = model.strategy_metrics(oos_preds)

    predictions = model.final_predictions(featured)
    predictions.to_csv(config.OUTPUTS_DIR / "predictions_nplus1.csv",
                       index=False)
    extraction = extract.build_extraction(price, posts, news_df, predictions)

    coverage = _coverage_note(
        {name: int(len(f)) for name, f in social_frames.items()},
        int(len(news_df)))
    report.build_report(extraction, predictions, metrics, strategy,
                        oos_preds, coverage)

    pooled = metrics[metrics["scope"] == "pooled"]
    print("\n=== WALK-FORWARD (OUT-OF-SAMPLE) ===")
    if not pooled.empty:
        r = pooled.iloc[0]
        print(f"n={int(r['n_oos'])}  direction accuracy "
              f"{r['directional_accuracy']:.1%} (baseline "
              f"{r['baseline_directional_accuracy']:.1%})  "
              f"RMSE {r['rmse']:.4f}  MAE {r['mae']:.4f}")
    if strategy:
        s = strategy["strategy"]
        print(f"Long/short rule: Sharpe {s['sharpe']:.2f} | "
              f"max drawdown {s['max_drawdown']:.1%} | "
              f"{strategy['oos_start']} -> {strategy['oos_end']}")
    print("\n=== n+1 PREDICTIONS ===")
    print(predictions[["ticker", "current_price", "predicted_return",
                       "predicted_direction", "confidence"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
