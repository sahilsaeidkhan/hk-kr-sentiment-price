"""Visual report generation: charts (matplotlib) + a self-contained HTML.

``report/report.html`` embeds its charts as base64 PNGs, so the single
file can be opened anywhere, attached, or printed to PDF. Layout: one
section per exchange with the per-ticker extraction table sorted by
social mentions (then sentiment), the n+1 prediction table, validation
metrics, and a methods/limitations section — every number comes from the
run's own output files, nothing is hand-entered.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_sentiment(extraction: pd.DataFrame, exchange: str) -> str:
    """Horizontal bars: sentiment score per ticker, coloured by sign."""
    d = extraction[extraction["exchange"] == exchange]
    d = d.sort_values("sentiment_score")
    fig, ax = plt.subplots(figsize=(7.2, 2.6 + 0.35 * len(d)))
    colours = ["#2e7d32" if v > 0.05 else "#c62828" if v < -0.05
               else "#9e9e9e" for v in d["sentiment_score"]]
    ax.barh(d["ticker"], d["sentiment_score"], color=colours)
    ax.axvline(0, color="#444", linewidth=0.8)
    for i, (score, mentions) in enumerate(
            zip(d["sentiment_score"], d["social_mentions"])):
        ax.text(score + (0.02 if score >= 0 else -0.02), i,
                f"{score:+.2f} · {mentions} posts", va="center",
                ha="left" if score >= 0 else "right", fontsize=8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Engagement-weighted social sentiment (-1 bearish … +1 bullish)")
    ax.set_title(f"{exchange} — social sentiment by ticker")
    return _fig_to_b64(fig)


def chart_equity(strategy: dict) -> str:
    """Equity curves of the OOS long/short rule vs buy-and-hold."""
    daily = strategy["_daily"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(daily.index, (1 + daily["strat"]).cumprod(),
            label="Model long/short (out-of-sample)", linewidth=1.4)
    ax.plot(daily.index, (1 + daily["bh"]).cumprod(),
            label="Buy & hold (same days)", linewidth=1.0, alpha=0.75)
    ax.set_ylabel("Growth of 1.0")
    ax.set_title("Walk-forward out-of-sample equity curves (equal-weight, no costs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_b64(fig)


def chart_pred_vs_actual(preds: pd.DataFrame) -> str:
    """Scatter of OOS predicted vs actual next-day returns."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(preds["y_pred"], preds["y_true"], s=7, alpha=0.45)
    lim = float(np.nanmax(np.abs(preds[["y_pred", "y_true"]].values))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#c62828", linewidth=0.9)
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.axvline(0, color="#888", linewidth=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Predicted next-day return")
    ax.set_ylabel("Actual next-day return")
    ax.set_title("Out-of-sample predictions vs actuals (pooled)")
    return _fig_to_b64(fig)


def _table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[key]}</td>" for key, _ in columns)
        body.append(f"<tr>{cells}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem auto;
       max-width: 980px; color: #1a1a2e; padding: 0 1rem; }
h1 { font-size: 1.55rem; margin-bottom: .2rem; }
h2 { margin-top: 2.2rem; border-bottom: 2px solid #3949ab; padding-bottom: .3rem; }
h3 { margin-top: 1.4rem; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; }
th, td { border: 1px solid #cfd8dc; padding: .38rem .5rem; text-align: left; }
th { background: #e8eaf6; }
tr:nth-child(even) { background: #f7f8fc; }
.meta { color: #555; font-size: .9rem; }
.card { background: #f4f6fb; border: 1px solid #dfe3f0; border-radius: 8px;
        padding: .8rem 1rem; margin: 1rem 0; }
img.chart { max-width: 100%; }
.small { font-size: .85rem; color: #444; }
"""


def build_report(extraction: pd.DataFrame, predictions: pd.DataFrame,
                 metrics: pd.DataFrame, strategy: dict,
                 preds: pd.DataFrame, coverage_note: str) -> str:
    """Write ``report/report.html`` and return its path."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []
    for exchange in ("HKEX", "KRX"):
        ex = (extraction[extraction["exchange"] == exchange]
              .sort_values(["social_mentions", "sentiment_score"],
                           ascending=False))
        sections.append(f"<h2>{exchange}</h2>")
        sections.append(
            f'<img class="chart" src="data:image/png;base64,'
            f'{chart_sentiment(extraction, exchange)}" alt="sentiment chart"/>')
        sections.append("<h3>Per-ticker extraction (sorted by social mentions)</h3>")
        sections.append(_table(ex, [
            ("ticker", "Ticker"), ("asset_name", "Asset"),
            ("current_price", "Current price"), ("social_mentions", "Posts"),
            ("sentiment_score", "Sentiment"), ("sentiment_direction", "Direction"),
            ("position", "Model position (n+1)"), ("support", "Support"),
            ("resistance", "Resistance"), ("news_headlines", "News (10d)"),
        ]))
    pred_view = predictions.copy()
    pred_view["predicted_return"] = pred_view["predicted_return"].map(
        lambda v: f"{v:+.2%}")
    pred_view["confidence"] = pred_view["confidence"].map(
        lambda v: f"{v:.2f}")
    m = metrics.set_index(["scope", "name"]) if len(metrics) else metrics
    def _m(scope: str, name: str, key: str, fmt: str = "{:.1%}") -> str:
        try:
            return fmt.format(float(m.loc[(scope, name), key]))
        except Exception:
            return "n/a"
    s, b = strategy.get("strategy", {}), strategy.get("buy_hold", {})
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>HKEX &amp; KRX Social-Sentiment Price Prediction — Visual Report</title>
<style>{_CSS}</style></head><body>
<h1>Social Sentiment → Next-Day Price Prediction: HKEX &amp; KRX</h1>
<p class="meta">Generated {generated} by <code>run_pipeline.py</code> ·
CrowdWisdomTrading intern task (via TZURONI), Task 2.</p>
<div class="card"><strong>Data coverage.</strong> {coverage_note}</div>
{''.join(sections)}
<h2>n+1 predictions — next trading day</h2>
{_table(pred_view, [
    ("ticker", "Ticker"), ("exchange", "Exchange"),
    ("current_price", "Current price"), ("predicted_return", "Predicted return"),
    ("predicted_price", "Predicted price"), ("predicted_direction", "Direction"),
    ("confidence", "Confidence*")])}
<p class="small">* Confidence = |predicted return| / (|predicted return| +
forest disagreement across trees). A ranking score, not a calibrated
probability.</p>
<h2>Validation — walk-forward, out-of-sample only</h2>
<p>Pooled directional accuracy <strong>{_m("pooled", "ALL", "directional_accuracy")}</strong>
vs persistence baseline {_m("pooled", "ALL", "baseline_directional_accuracy")} ·
HKEX {_m("exchange", "HKEX", "directional_accuracy")} ·
KRX {_m("exchange", "KRX", "directional_accuracy")} ·
RMSE {_m("pooled", "ALL", "rmse", "{:.4f}")} vs naive {_m("pooled", "ALL", "naive_rmse", "{:.4f}")} ·
MAE {_m("pooled", "ALL", "mae", "{:.4f}")}.</p>
<img class="chart" src="data:image/png;base64,{chart_equity(strategy)}" alt="equity curves"/>
<img class="chart" src="data:image/png;base64,{chart_pred_vs_actual(preds)}" alt="pred vs actual"/>
<p>Long/short rule on out-of-sample predictions ({strategy.get('oos_start')}
→ {strategy.get('oos_end')}, {strategy.get('n_predictions')} predictions):
Sharpe <strong>{s.get('sharpe', 0):.2f}</strong> vs buy-and-hold
{b.get('sharpe', 0):.2f}; max drawdown {s.get('max_drawdown', 0):.1%} vs
{b.get('max_drawdown', 0):.1%}. No transaction costs modelled.</p>
<h2>How the n+1 results will be checked</h2>
<p>The predictions above are frozen in
<code>outputs/predictions_nplus1.csv</code> with their origin date. After
the next trading session, <code>tools/check_predictions.py</code> fetches
the realised closes, computes the actual next-day returns, and scores
direction accuracy and MAE against the frozen file — the same discipline
as the walk-forward evaluation, applied to the live forecast.</p>
<h2>Methods &amp; limitations (summary)</h2>
<p class="small">Features use price history and daily aggregated social/news
sentiment only; targets are next-trading-day returns. Validation is an
expanding-window walk-forward — no shuffling, no lookahead. Social
sentiment history is limited to the last 10 days by the assignment's
design, so most training rows carry the coverage-flagged neutral
treatment; lexicon scoring for Chinese/Korean forum text captures
directional market vocabulary but not sarcasm or negation. Full detail
in the repository README. This is a research prototype, not financial
advice.</p>
</body></html>"""
    out = config.REPORT_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"[report] visual report saved to {out}")
    return str(out)
