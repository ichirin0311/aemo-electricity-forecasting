"""
Step 2: does AEMO's PD7DAY outlook help as a model input for the "today" lead?
Walk-forward (2025-01..2026-08, monthly retrain) with vs without the AEMO features.
PD7DAY history starts 2024-04-24, so models for Jan-Apr 2025 are trained with the AEMO
features missing for part of their window; results are also shown from 2025-05 on.

Result (adopted as model v2.2), today lead, 2025-01..2026-08:
  normal MAE 27.92 -> 21.55 (-23%), normal R2 0.531 -> 0.674, median abs. error 20.3 -> 14.6,
  better in 20/20 months. Q90: spike MAE 1917 -> 1820, spikes flagged 51% -> 61%,
  flags that were spikes 14% -> 18%, coverage 87.8% -> 88.5%.

Usage: python experiments/pd7day_as_input.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_backtest import load_backtest_frame, metrics, walk_forward  # noqa: E402
from src.day_ahead import SPIKE, load_aemo_asof  # noqa: E402

if __name__ == "__main__":
    first, last = pd.Period("2025-01", "M"), pd.Period("2026-08", "M")
    df = load_backtest_frame(first, last)
    months = pd.period_range(first, last, freq="M")
    aemo = load_aemo_asof()
    leads = {"today": 1}

    runs = {}
    for name, a in [("without AEMO", None), ("with AEMO", aemo)]:
        monthly, preds = walk_forward(df, months, leads=leads, verbose=False, aemo=a)
        runs[name] = (monthly, preds)
        print(f"done: {name}", flush=True)

    pd.set_option("display.width", 200)
    for label, since in [("2025-01..2026-08", "2025-01"), ("2025-05..2026-08 (full AEMO history in training)", "2025-05")]:
        rows = {}
        for name, (monthly, preds) in runs.items():
            p = preds[preds["month"] >= since]
            m = metrics(p)
            n = p["rrp"] < SPIKE
            s = p["rrp"] >= SPIKE
            flag = p["rrp_risk_ceiling"] >= SPIKE
            rows[name] = {**{k: m[k] for k in ["normal_MAE", "normal_R2", "spike_MAE_q90", "q90_coverage"]},
                          "normal_medAE": (p.loc[n, "rrp_base_prediction"] - p.loc[n, "rrp"]).abs().median(),
                          "normal_bias": (p.loc[n, "rrp_base_prediction"] - p.loc[n, "rrp"]).mean(),
                          "spikes_flagged": (flag & s).sum() / s.sum(),
                          "flags_that_were_spikes": (flag & s).sum() / max(flag.sum(), 1)}
        print(f"\n===== Today lead, {label} =====")
        print(pd.DataFrame(rows).T.round(3).to_string())

    mw = runs["without AEMO"][0].set_index("month")["normal_MAE"]
    ma = runs["with AEMO"][0].set_index("month")["normal_MAE"]
    print("\n===== Monthly normal MAE =====")
    print(pd.DataFrame({"without": mw, "with": ma, "change": ma - mw}).round(2).to_string())
    print(f"\nMonths improved: {(ma < mw).sum()}/{len(ma)}")
