# HKEX & KRX Social-Sentiment Price Prediction

A Python pipeline that collects retail-investor sentiment for five Hong
Kong (HKEX) and five South Korean (KRX) large-cap stocks, combines it
with price history and recent news, and trains a model to predict each
asset's **next-trading-day (n+1)** price direction and return — validated
with rigorous walk-forward, out-of-sample evaluation.

Built for the CrowdWisdomTrading intern task (via TZURONI), Task 2.
Task 1 (X sentiment → BTC) is a separate repository; this project reuses
its proven patterns — modular ingestion, coverage-flagged sentiment
joins, walk-forward validation, fully computed reports — for the
two-exchange equity setting.

> Research prototype. Nothing here is financial advice.

## Assets

| Exchange | Tickers |
|---|---|
| HKEX | 0700.HK Tencent · 9988.HK Alibaba · 1810.HK Xiaomi · 3690.HK Meituan · 1211.HK BYD |
| KRX | 005930.KS Samsung Electronics · 000660.KS SK Hynix · 035420.KS NAVER · 005380.KS Hyundai Motor · 068270.KS Celltrion |

## Architecture

```
run_pipeline.py            end-to-end orchestration
src/
  config.py                assets, windows, source/actor IDs, tokens (env only)
  prices.py                Yahoo Finance daily OHLCV, 9 months, 10 tickers
  lihkg.py                 HKEX source 1: LIHKG finance channel
  naver.py                 KRX source 1: Naver Finance discussion boards
  telegram_apify.py        both markets: Telegram (keyword search + channels)
  news.py                  per-asset news: Exa primary, Tavily fallback
  sentiment.py             multilingual scoring (VADER / ZH lexicon / KO lexicon)
  features.py              SQLite join + feature engineering (no lookahead)
  model.py                 walk-forward validation + n+1 forecasts
  extract.py               per-ticker extraction (price, sentiment, S/R, stance)
  report.py                charts + self-contained report/report.html
tools/check_predictions.py scores the frozen n+1 forecasts after the next session
data/                      prices, posts, news, joined dataset (CSV + SQLite)
outputs/                   metrics, OOS predictions, n+1 forecasts, extraction
report/                    report.html and report.pdf (print rendering)
```

## Data sources (per the brief's lists)

Two *different* listed sources are implemented per exchange, plus news:

**HKEX**
1. **LIHKG** （財經台 finance channel) — two routes: the Apify actor
   `claude_code_reviewer/lihkg-scraper`, and LIHKG's public `api_v2`
   JSON endpoints (no key). Post timestamps (UTC), title text, likes,
   reply counts are collected; LIHKG exposes no view/follower counts.
2. **Telegram** — Apify keyword-search actor
   `lofomachines/telegram-keyword-search-scraper` (country-scoped HK),
   channel actor `tri_angle/telegram-scraper`, and a key-free
   `t.me/s/<channel>` preview scraper for channels configured in
   `src/config.py` (none are guessed by default).

**KRX**
1. **Naver Finance Stock Discussion Boards** (종목토론) — direct scrape
   of the classic board pages (`finance.naver.com/item/board.naver`)
   with BeautifulSoup, plus probes of the JSON endpoints the current
   app uses. Timestamps are converted from KST to UTC.
2. **Telegram** — the same three routes as HKEX, country-scoped KR
   (the brief lists "KakaoTalk Open Chats & Telegram Channels";
   KakaoTalk open chats are not publicly readable, Telegram is the
   executable half of that entry).

**News (both exchanges)** — **Exa** (`api.exa.ai/search`) is the primary
source, per the brief; **Tavily** is the automatic fallback when no Exa
key is configured. Headlines are scored with the same sentiment scorer
and enter the feature set as a daily news-sentiment series.

**Prices** — Yahoo Finance via `yfinance` (no key), 9 months of daily
OHLCV per ticker (brief minimum: 6), symbols `0700.HK`-style / `.KS`.

Sentiment scoring is script-aware: VADER for English text, and curated
financial-forum lexicons for Traditional Chinese (LIHKG) and Korean
(Naver/Telegram) text — directional market vocabulary, normalised to
[-1, +1]. Every lexicon term is listed in `src/sentiment.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add your own free-tier tokens
python run_pipeline.py
```

`.env` variables: `APIFY_API_TOKEN`, `EXA_API_KEY`, `TAVILY_API_KEY` —
all optional; the pipeline degrades gracefully and logs what it skipped.
**Keys are never committed** (`.env` is git-ignored) and never printed.

## Method

* **Join.** Posts are bucketed to their UTC date and rolled forward to
  the ticker's next trading day, then joined with price bars in SQLite
  (`data/dataset.db`). Days without posts carry sentiment 0 with an
  explicit `sent_observed` flag — coverage is auditable, never imputed.
* **Features** (per ticker-day): lagged returns (1/3/5/7d), rolling
  volatility (7/14d), daily mean sentiment, sentiment 3d/7d moving
  averages and momentum, bullish ratio, post-volume and price-volume
  z-scores, 3-day news sentiment, day of week, coverage flags.
* **Model.** One pooled `RandomForestRegressor` (300 trees, depth 6)
  over all ten tickers; target = next-trading-day return.
* **Validation.** Expanding-window **walk-forward** over calendar time
  (10-trading-day test blocks, refit each block). A training row is used
  only if its own target date precedes the block start — no shuffling,
  no lookahead. Baselines: zero-return and directional persistence.
* **Trading rule.** Long/short by the sign of the out-of-sample
  prediction, equal-weight across tickers per day, no costs (stated
  limitation); Sharpe annualised with √252.
* **Support/resistance.** Nearest swing high/low (5-session extrema,
  last 60 sessions) around the current price, with classic pivot
  R1/S1 as fallbacks.
* **n+1 forecasts** are frozen in `outputs/predictions_nplus1.csv`;
  after the next session, `tools/check_predictions.py` scores them
  against realised closes (direction accuracy + MAE). The confidence
  column is `|pred| / (|pred| + std across trees)` — a ranking score,
  not a calibrated probability.

## Results of the executed run (2026-11-09)

Prices: 1,855 daily bars (2026-02-04 → 2026-11-06). News: 38 recent
headlines, all ten tickers. Walk-forward, out-of-sample (n = 1,030):

| Scope | Direction acc. | Persistence baseline | RMSE | MAE |
|---|---|---|---|---|
| Pooled | 49.3% | 47.2% | 0.0393 | 0.0273 |
| HKEX | 48.4% | 47.7% | 0.0275 | 0.0206 |
| KRX | 50.1% | 46.7% | 0.0484 | 0.0342 |

Long/short rule on the same out-of-sample predictions: Sharpe −1.95,
max drawdown −19.9% (2026-06-09 → 2026-11-05) — i.e. **the model does
not beat the market in this run**, and the numbers are reported exactly
as computed. The dominant reason is data coverage, below.

## Limitations — read before judging the numbers

* **Apify credit exhausted.** The free-plan monthly credit ($5) was
  already used up when this run executed (cycle resets 2026-11-30), so
  the LIHKG actor and Telegram keyword-search routes could not run.
  They are implemented and will execute unchanged on a funded account
  or after the reset. LIHKG was instead executed via its public
  `api_v2` route: 899 finance-channel threads active in the 10-day
  window were scanned, and **none had a title naming one of the five
  HK assets** — LIHKG discussion of these names happens inside general
  megathreads, so per-ticker title attribution legitimately yields
  zero in this window. Social-sentiment coverage in the model is
  therefore 0% for this run; results above are price + news driven.
* **Naver boards returned no parseable posts.** Naver has migrated the
  discussion boards into a JavaScript app; the classic server-rendered
  table and the probed JSON endpoints served no posts to this client.
  `src/naver.py` implements both routes and logs the outcome; a
  browser-rendered fetch or a future documented endpoint would slot
  into the same module.
* **News ran on the Tavily fallback** because no `EXA_API_KEY` was
  configured locally; the Exa path is implemented and takes over
  automatically when the key is present.
* Social history is 10 days by the brief's design; daily equity
  prediction is close to a random walk at this horizon, and weak
  metrics are expected and reported as-is.
* Lexicon scoring captures directional vocabulary, not sarcasm,
  negation, or forum slang nuance. LIHKG provides no view/follower
  counts, so those fields are 0 for that source (engagement = likes
  and replies).
* Strategy evaluation ignores transaction costs, borrow fees for
  shorts, and HK/KR market microstructure.

## Deliverables map (brief → file)

| Brief item | File |
|---|---|
| Pipeline code | `run_pipeline.py`, `src/` |
| Scraped posts / prices / sentiment | `data/posts_*.csv`, `data/prices*.csv`, `data/dataset.csv` |
| Per-ticker extraction | `outputs/per_ticker_extraction.csv` + report tables |
| Validation metrics | `outputs/walk_forward_metrics.csv`, `outputs/oos_predictions.csv` |
| n+1 predictions | `outputs/predictions_nplus1.csv` |
| Visual report | `report/report.html`, `report/report.pdf` |
| Post-hoc forecast check | `tools/check_predictions.py` |
