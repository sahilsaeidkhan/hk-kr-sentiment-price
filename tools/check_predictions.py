"""Score the frozen n+1 predictions against realised next-day closes.

Usage (run *after* the next trading session following a pipeline run)::

    python tools/check_predictions.py

Reads ``outputs/predictions_nplus1.csv`` (written by ``run_pipeline.py``),
fetches each ticker's closes from Yahoo Finance, finds the first close
strictly after the prediction's origin date, and writes
``outputs/predictions_nplus1_scored.csv`` with actual returns, direction
hits, and absolute errors, plus a printed summary. The frozen prediction
file is never modified — forecasts stay auditable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def main() -> int:
    pred_path = config.OUTPUTS_DIR / "predictions_nplus1.csv"
    if not pred_path.exists():
        print("predictions_nplus1.csv not found — run run_pipeline.py first.")
        return 1
    preds = pd.read_csv(pred_path)
    rows = []
    for _, row in preds.iterrows():
        ticker = row["ticker"]
        origin = pd.Timestamp(row["origin_date"])
        raw = yf.download(ticker, start=origin.date().isoformat(),
                          period="10d", auto_adjust=False, progress=False,
                          timeout=30)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw.reset_index().rename(columns={"Date": "date"})
        closes["date"] = pd.to_datetime(closes["date"]).dt.tz_localize(None)
        after = closes[closes["date"] > origin].sort_values("date")
        if after.empty:
            print(f"{ticker}: next close not available yet.")
            continue
        actual_close = float(after["Close"].iloc[0])
        actual_ret = actual_close / float(row["current_price"]) - 1.0
        rows.append({
            **row.to_dict(),
            "actual_date": str(after["date"].iloc[0].date()),
            "actual_close": actual_close,
            "actual_return": actual_ret,
            "direction_hit": int(np.sign(actual_ret)
                                 == np.sign(row["predicted_return"])),
            "abs_error": abs(actual_ret - float(row["predicted_return"])),
        })
    if not rows:
        print("No ticker could be scored yet — try again after the "
              "next session.")
        return 0
    scored = pd.DataFrame(rows)
    out = config.OUTPUTS_DIR / "predictions_nplus1_scored.csv"
    scored.to_csv(out, index=False)
    print(f"Scored {len(scored)} predictions -> {out}")
    print(f"Direction accuracy: {scored['direction_hit'].mean():.1%} | "
          f"MAE: {scored['abs_error'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
