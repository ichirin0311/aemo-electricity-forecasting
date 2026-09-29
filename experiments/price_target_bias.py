"""
Compare price-target formulations for the central model on the 20-month walk-forward
(run_backtest.walk_forward). Motivation: with asinh(rrp) + L2 loss, the central model
under-predicted in every month (bias -3 to -36 $/MWh), because negative prices pull the
mean in asinh space far down. Candidates:
  - asinh(rrp / c) with a scale c, so the normal price range is closer to linear
  - L1 objective (predicts the median, which survives the inverse transform unbiased)

Usage: python experiments/price_target_bias.py round1|round2 [lead ...]   (default leads: today)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_backtest import load_backtest_frame, metrics, walk_forward  # noqa: E402
from src.day_ahead import LEADS, PARAMS_BASE, SPIKE  # noqa: E402

PARAMS_L1 = {**PARAMS_BASE, "objective": "regression_l1", "metric": "l1"}
PARAMS_L2 = {**PARAMS_BASE, "objective": "regression", "metric": "rmse"}
OLD = {"base_scale": 1.0, "q90_scale": 1.0, "base_params": PARAMS_L2}  # config before this experiment

# Round 1 (today lead) scaled both the central and Q90 targets. Result: asinh(x/100)+L1 was
# best for the central model (normal MAE 31.87 -> 27.92, bias -16.4 -> -0.2, beat naive 20/20
# months), but scaling the Q90 target worsened its spike MAE (1917 -> 2039), so the Q90 model
# keeps asinh(x). Round 2 confirmed that split on both leads (tomorrow: normal MAE 34.99 -> 30.86,
# bias -16.9 -> +1.4, beat naive 18/20 -> 20/20 months; Q90 unchanged) and it was adopted.
ROUND1 = {
    "asinh(x), L2 [old]": OLD,
    "asinh(x/30), L2": {**OLD, "base_scale": 30.0, "q90_scale": 30.0},
    "asinh(x/100), L2": {**OLD, "base_scale": 100.0, "q90_scale": 100.0},
    "asinh(x), L1": {**OLD, "base_params": PARAMS_L1},
    "asinh(x/100), L1": {"base_scale": 100.0, "q90_scale": 100.0, "base_params": PARAMS_L1},
}
ROUND2 = {
    "asinh(x), L2 [old]": OLD,
    "central asinh(x/100)+L1, Q90 asinh(x) [adopted]": {"base_scale": 100.0, "q90_scale": 1.0,
                                                        "base_params": PARAMS_L1},
}

if __name__ == "__main__":
    variants = {"round1": ROUND1, "round2": ROUND2}[sys.argv[1] if len(sys.argv) > 1 else "round1"]
    leads = {k: LEADS[k] for k in (sys.argv[2:] or ["today"])}
    first, last = pd.Period("2025-01", "M"), pd.Period("2026-08", "M")
    df = load_backtest_frame(first, last)
    months = pd.period_range(first, last, freq="M")

    summary, monthly_bias = [], {}
    for name, kwargs in variants.items():
        monthly, preds = walk_forward(df, months, leads=leads, verbose=False, **kwargs)
        for lead, g in preds.groupby("lead", sort=False):
            n = g["rrp"] < SPIKE
            err = (g["rrp_base_prediction"] - g["rrp"])[n]
            m = metrics(g)
            summary.append({
                "variant": name, "lead": lead,
                "normal_MAE": m["normal_MAE"], "normal_R2": m["normal_R2"], "normal_bias": err.mean(),
                "months_beating_naive": int((monthly.query("lead == @lead")["normal_MAE"]
                                             < monthly.query("lead == @lead")["normal_MAE_naive"]).sum()),
                "spike_MAE_central": m["spike_MAE_central"],
                "spike_MAE_q90": m["spike_MAE_q90"], "q90_coverage": m["q90_coverage"],
            })
            monthly_bias[(name, lead)] = g[n].assign(e=err).groupby("month")["e"].mean()
        print(f"done: {name}", flush=True)

    pd.set_option("display.width", 200)
    print("\n===== Pooled (2025-01..2026-08) =====")
    print(pd.DataFrame(summary).set_index(["lead", "variant"]).round(3).to_string())
    print("\n===== Monthly normal-condition bias (pred - actual, $/MWh) =====")
    print(pd.DataFrame(monthly_bias).round(1).to_string())
