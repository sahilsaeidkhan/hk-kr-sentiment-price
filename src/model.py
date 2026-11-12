"""Walk-forward validation and n+1 forecasting.

Design (rigorous, no lookahead, no shuffling):

* All ten tickers are pooled into one modelling frame; a row is a
  (ticker, trading day) with the features of :mod:`src.features`.
* **Expanding-window walk-forward over calendar time:** test blocks of
  ``WF_STEP_DAYS`` trading days (per-ticker date index); a row enters the
  training set for a test block only if its target date — the *next*
  trading day — is strictly before the block's first date. The model is
  refitted at every block.
* Model: ``RandomForestRegressor`` (300 trees, depth 6), the same family
  as the Task-1 pipeline, fit on pooled rows so the HKEX and KRX names
  share one model; metrics are reported pooled, per exchange and per
  ticker.
* Baselines: zero-return forecast and directional persistence (sign of
  the last 1-day return).
* Trading evaluation uses **out-of-sample predictions only**: a
  long/short rule (long when the predicted next-day return is positive,
  short when negative, equal weight per ticker-day, no costs — stated
  limitation) versus buy-and-hold over the same days. Sharpe annualised
  with sqrt(252).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config
from .features import FEATURE_COLUMNS


def _make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        n_jobs=-1, random_state=config.RANDOM_STATE)


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def walk_forward(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the walk-forward evaluation.

    Returns ``(metrics, oos_predictions)`` where *metrics* has one row per
    scope (pooled / per exchange / per ticker).
    """
    feat = (d.dropna(subset=FEATURE_COLUMNS)
              .sort_values(["date", "ticker"]).reset_index(drop=True))
    all_dates = np.sort(feat["date"].unique())
    pred_rows: list[dict] = []
    start_idx = config.INITIAL_TRAIN_DAYS
    while start_idx < len(all_dates):
        block_dates = all_dates[start_idx:start_idx + config.WF_STEP_DAYS]
        block_start = block_dates[0]
        # Exact no-lookahead rule: a row is trainable for this block only
        # if its target date (its own ticker's next trading day, stored
        # in target_date) is strictly before the block's first date.
        train = feat[(feat["target_date"] < block_start)
                     & feat["target_next"].notna()]
        block = feat[feat["date"].isin(block_dates)]
        block = block[block["target_next"].notna()]
        if len(train) < 200 or block.empty:
            start_idx += config.WF_STEP_DAYS
            continue
        model = _make_model()
        model.fit(train[FEATURE_COLUMNS], train["target_next"])
        preds = model.predict(block[FEATURE_COLUMNS])
        for (_, row), p in zip(block.iterrows(), preds):
            pred_rows.append({
                "date": row["date"], "ticker": row["ticker"],
                "exchange": config.ASSET_BY_TICKER[row["ticker"]].exchange,
                "y_true": float(row["target_next"]), "y_pred": float(p),
                "persist_sign": float(np.sign(row["ret_1d"])),
            })
        start_idx += config.WF_STEP_DAYS

    preds_df = pd.DataFrame(pred_rows)
    metrics = _metrics_table(preds_df)
    return metrics, preds_df


def _scope_metrics(sub: pd.DataFrame, scope: str, name: str) -> dict:
    y, p = sub["y_true"].values, sub["y_pred"].values
    nz = sub[sub["y_true"] != 0]
    dir_acc = float((np.sign(nz["y_pred"]) == np.sign(nz["y_true"])).mean())
    base = float((nz["persist_sign"] == np.sign(nz["y_true"])).mean())
    return {
        "scope": scope, "name": name, "n_oos": int(len(sub)),
        "rmse": _rmse(y, p), "mae": float(mean_absolute_error(y, p)),
        "naive_rmse": _rmse(y, np.zeros_like(y)),
        "naive_mae": float(mean_absolute_error(y, np.zeros_like(y))),
        "directional_accuracy": dir_acc,
        "baseline_directional_accuracy": base,
    }


def _metrics_table(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    rows = [_scope_metrics(preds, "pooled", "ALL")]
    for exchange, sub in preds.groupby("exchange"):
        rows.append(_scope_metrics(sub, "exchange", exchange))
    for ticker, sub in preds.groupby("ticker"):
        rows.append(_scope_metrics(sub, "ticker", ticker))
    return pd.DataFrame(rows)


def strategy_metrics(preds: pd.DataFrame) -> dict:
    """Long/short rule on out-of-sample predictions vs buy-and-hold."""
    if preds.empty:
        return {}
    d = preds.copy()
    d["position"] = np.sign(d["y_pred"])
    d["strat_ret"] = d["position"] * d["y_true"]
    # Equal-weight daily portfolio across the tickers predicted that day.
    daily = d.groupby("date").agg(strat=("strat_ret", "mean"),
                                  bh=("y_true", "mean"))

    def _stats(r: pd.Series) -> dict:
        mu, sd = float(r.mean()), float(r.std(ddof=1))
        downside = r[r < 0].std(ddof=1)
        ann = np.sqrt(config.TRADING_DAYS_PER_YEAR)
        equity = (1 + r).cumprod()
        return {
            "sharpe": float(mu / sd * ann) if sd else 0.0,
            "sortino": float(mu / downside * ann)
                       if downside and not np.isnan(downside) else 0.0,
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
            "total_return": float(equity.iloc[-1] - 1),
        }

    return {
        "oos_days": int(len(daily)),
        "oos_start": str(daily.index.min().date()),
        "oos_end": str(daily.index.max().date()),
        "n_predictions": int(len(d)),
        "strategy": _stats(daily["strat"]),
        "buy_hold": _stats(daily["bh"]),
        "_daily": daily,
    }


def final_predictions(d: pd.DataFrame) -> pd.DataFrame:
    """Refit on all labelled rows; predict n+1 from each ticker's latest row.

    Confidence is a transparent score in (0, 1):
    ``|pred| / (|pred| + tree_std)`` — the predicted move relative to the
    forest's own disagreement (std across trees). It ranks predictions;
    it is not a calibrated probability, and the README says so.
    """
    feat = d.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    train = feat[feat["target_next"].notna()]
    model = _make_model()
    model.fit(train[FEATURE_COLUMNS], train["target_next"])
    rows: list[dict] = []
    for ticker, group in feat.groupby("ticker"):
        last = group.sort_values("date").iloc[-1]
        x = last[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
        tree_preds = np.array([t.predict(x)[0] for t in model.estimators_])
        pred_ret = float(tree_preds.mean())
        tree_std = float(tree_preds.std())
        confidence = abs(pred_ret) / (abs(pred_ret) + tree_std + 1e-9)
        rows.append({
            "ticker": ticker,
            "exchange": config.ASSET_BY_TICKER[ticker].exchange,
            "origin_date": str(last["date"].date()),
            "current_price": float(last["close"]),
            "predicted_return": pred_ret,
            "predicted_price": float(last["close"] * (1 + pred_ret)),
            "predicted_direction": ("UP" if pred_ret > 0 else
                                    "DOWN" if pred_ret < 0 else "FLAT"),
            "tree_std": tree_std,
            "confidence": float(confidence),
        })
    return (pd.DataFrame(rows)
              .sort_values(["exchange", "ticker"]).reset_index(drop=True))
