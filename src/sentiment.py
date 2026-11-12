"""Multilingual sentiment scoring for forum posts.

Three scoring paths, chosen by the script actually present in the text:

* **Latin / English text** — VADER compound score (validated lexicon model,
  the same approach as the Task-1 pipeline).
* **Chinese text** (LIHKG posts, Traditional Chinese) — a curated
  financial-forum lexicon: bullish/bearish market vocabulary with a few
  common intensifiers. Score = (pos - neg) / (pos + neg), in [-1, +1].
* **Korean text** (Naver / Telegram posts) — a curated financial-forum
  lexicon on the same normalised scale.

Lexicon scoring is deliberately simple and fully auditable: every term is
listed below. It captures directional market language ("the words a post
uses about price direction") rather than general emotion, which is the
signal this pipeline needs. Its limits — sarcasm, slang, negation — are
stated in the README.
"""
from __future__ import annotations

import re
import unicodedata

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_VADER = SentimentIntensityAnalyzer()

# --- Chinese financial-forum lexicon (Traditional, LIHKG usage) ---------------
_ZH_POSITIVE = (
    "爆升", "大升", "急升", "上升", "升穿", "創新高", "破頂", "反彈", "回升",
    "利好", "好消息", "睇好", "買入", "掃貨", "加倉", "增持", "抄底", "撈底",
    "強勢", "牛市", "牛", "盈利", "賺", "增長", "超預期", "突破", "升",
)
_ZH_NEGATIVE = (
    "暴跌", "大跌", "急跌", "下跌", "跌穿", "創新低", "破底", "回調", "回落",
    "利淡", "壞消息", "睇淡", "賣出", "沽貨", "走貨", "減持", "割肉", "止蝕",
    "弱勢", "熊市", "熊", "虧損", "蝕", "衰退", "暴雷", "插水", "崩", "跌",
)
# --- Korean financial-forum lexicon (Naver 종목토론 usage) ---------------------
_KO_POSITIVE = (
    "급등", "상승", "호재", "매수", "반등", "신고가", "강세", "상향", "흑자",
    "수익", "증가", "성장", "돌파", "급반등", "상한가", "호실적", "기대감",
    "사자", "오른다", "상승세",
)
_KO_NEGATIVE = (
    "급락", "하락", "악재", "매도", "폭락", "신저가", "약세", "하향", "적자",
    "손실", "감소", "우려", "하한가", "부진", "팔자", "내린다", "하락세",
    "폭등락",
)

_HANGUL = re.compile(r"[\uac00-\ud7a3]")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def detect_script(text: str) -> str:
    """Return 'ko', 'zh' or 'en' based on the dominant script in *text*."""
    if _HANGUL.search(text):
        return "ko"
    if _CJK.search(text):
        return "zh"
    return "en"


def _lexicon_score(text: str, positive: tuple[str, ...],
                   negative: tuple[str, ...]) -> float:
    """Normalised lexicon score in [-1, +1]; 0.0 when nothing matches.

    Longer phrases are counted first so that, e.g., 大升 is not double
    counted as 升 (matched spans are blanked out after each hit).
    """
    work = text
    pos = neg = 0
    for term in sorted(positive, key=len, reverse=True):
        hits = work.count(term)
        if hits:
            pos += hits
            work = work.replace(term, " ")
    for term in sorted(negative, key=len, reverse=True):
        hits = work.count(term)
        if hits:
            neg += hits
            work = work.replace(term, " ")
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def score_text(text: str) -> float:
    """Sentiment score in [-1, +1] for one post, in any of the 3 scripts."""
    if not text or not str(text).strip():
        return 0.0
    text = unicodedata.normalize("NFC", str(text))
    script = detect_script(text)
    if script == "ko":
        return _lexicon_score(text, _KO_POSITIVE, _KO_NEGATIVE)
    if script == "zh":
        return _lexicon_score(text, _ZH_POSITIVE, _ZH_NEGATIVE)
    return float(_VADER.polarity_scores(text)["compound"])
