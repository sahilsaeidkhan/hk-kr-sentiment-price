"""Central configuration for the HKEX/KRX sentiment price-prediction pipeline.

All secrets are read from a local ``.env`` file (never hard-coded, never
committed — see ``.gitignore``). Every other module imports its settings
from here so the whole pipeline can be re-pointed (assets, windows, actor
IDs) from one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
CHARTS_DIR = OUTPUTS_DIR / "charts"
for _d in (DATA_DIR, OUTPUTS_DIR, REPORT_DIR, CHARTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# --- API tokens (from .env only) --------------------------------------------
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
EXA_API_KEY = os.getenv("EXA_API_KEY", "").strip()

# --- Time windows -------------------------------------------------------------
SOCIAL_LOOKBACK_DAYS = 10      # assignment scope: social posts, last 10 days
PRICE_HISTORY_MONTHS = 9       # >= 6 months required; 9 gives validation room
NEWS_LOOKBACK_DAYS = 10

# --- Assets -------------------------------------------------------------------
@dataclass(frozen=True)
class Asset:
    """One listed asset plus the search vocabulary used per source."""

    ticker: str                       # Yahoo Finance symbol
    code: str                         # local exchange code (e.g. "0700")
    name_en: str
    name_local: str
    exchange: str                     # "HKEX" | "KRX"
    currency: str                     # "HKD" | "KRW"
    keywords_local: tuple[str, ...]   # forum search/match terms (local language)
    keywords_en: tuple[str, ...]      # news / Telegram search terms (English)


ASSETS: tuple[Asset, ...] = (
    # --- Hong Kong (HKEX) ---
    Asset("0700.HK", "0700", "Tencent Holdings", "騰訊控股", "HKEX", "HKD",
          ("騰訊", "騰訊控股", "700"), ("Tencent", "$0700.HK", "0700.HK")),
    Asset("9988.HK", "9988", "Alibaba Group", "阿里巴巴", "HKEX", "HKD",
          ("阿里巴巴", "阿里", "9988"), ("Alibaba", "$9988.HK", "9988.HK")),
    Asset("1810.HK", "1810", "Xiaomi Corporation", "小米集團", "HKEX", "HKD",
          ("小米", "小米集團", "1810"), ("Xiaomi", "$1810.HK", "1810.HK")),
    Asset("3690.HK", "3690", "Meituan", "美團", "HKEX", "HKD",
          ("美團", "3690"), ("Meituan", "$3690.HK", "3690.HK")),
    Asset("1211.HK", "1211", "BYD Company", "比亞迪股份", "HKEX", "HKD",
          ("比亞迪", "1211"), ("BYD", "$1211.HK", "1211.HK")),
    # --- South Korea (KRX) ---
    Asset("005930.KS", "005930", "Samsung Electronics", "삼성전자", "KRX", "KRW",
          ("삼성전자", "005930"), ("Samsung Electronics", "005930.KS")),
    Asset("000660.KS", "000660", "SK Hynix", "SK하이닉스", "KRX", "KRW",
          ("SK하이닉스", "하이닉스", "000660"), ("SK Hynix", "000660.KS")),
    Asset("035420.KS", "035420", "NAVER Corporation", "네이버", "KRX", "KRW",
          ("네이버", "035420"), ("NAVER Corp", "035420.KS")),
    Asset("005380.KS", "005380", "Hyundai Motor", "현대차", "KRX", "KRW",
          ("현대차", "현대자동차", "005380"), ("Hyundai Motor", "005380.KS")),
    Asset("068270.KS", "068270", "Celltrion", "셀트리온", "KRX", "KRW",
          ("셀트리온", "068270"), ("Celltrion", "068270.KS")),
)

ASSET_BY_TICKER: dict[str, Asset] = {a.ticker: a for a in ASSETS}

# --- Sentiment sources ----------------------------------------------------------
# HKEX, per the brief's list: LIHKG + Telegram.
#   * LIHKG finance channel (財經台, cat_id 15) — executed via LIHKG's public
#     api_v2 JSON endpoints; the Apify actor below is the alternative route.
#   * Telegram — Apify keyword-search actor (listed source for both markets).
LIHKG_FINANCE_CAT_ID = 15
LIHKG_ACTOR_ID = "claude_code_reviewer/lihkg-scraper"
TELEGRAM_SEARCH_ACTOR_ID = "lofomachines/telegram-keyword-search-scraper"
TELEGRAM_CHANNEL_ACTOR_ID = "tri_angle/telegram-scraper"
# Public Telegram channels to scrape via t.me/s/ preview + the channel actor.
# Left empty by default: no channel is guessed. Add verified public channel
# usernames here and both Telegram routes will pick them up automatically.
TELEGRAM_CHANNELS: tuple[str, ...] = ()

# KRX, per the brief's list: Naver Finance discussion boards + Telegram.
NAVER_BOARD_URL = "https://finance.naver.com/item/board.naver?code={code}&page={page}"

# --- Modelling ------------------------------------------------------------------
INITIAL_TRAIN_DAYS = 70        # walk-forward: first test block starts here
WF_STEP_DAYS = 10              # walk-forward test block size (trading days)
TRADING_DAYS_PER_YEAR = 252    # HKEX/KRX annualisation
RANDOM_STATE = 42
