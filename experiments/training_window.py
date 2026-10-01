"""
Issue #5: how should the training window treat older market regimes?
Spikes became far rarer in 2026, even at the same demand level, so older data may
teach the models a spikier market than the current one. But spikes still happen on
extreme-demand days, so dropping or down-weighting old data could understate tail risk.

Walk-forward (2025-01..2026-08, monthly retrain, AEMO input for "today") comparing:
  - 365 days (current), 180 days, 730 days
  - 365 days with recency weights (half-life 180 and 90 days) on the price models

Reported pooled, split by 2025 / 2026, and on high-demand intervals (actual >= 10 GW),
where the risk ceiling must not get complacent.

Result (adopted: 730 days for "today", 365 for "tomorrow" = model v2.3):
- 180 days: worse normal MAE and much weaker on high-demand days (tomorrow: spikes flagged
  on >= 10 GW intervals 62% -> 23%; none at all in 2026). Dropping old regimes is unsafe.
- Recency weights (half-life 180 / 90 days): no consistent gain; beat 365d in only 4-9 of 20 months.
- 730 days, today: normal MAE unchanged (21.55 -> 21.56), every spike metric better (flagged 60% ->
  65%, high-demand flagged 68% -> 75%, coverage 88.3% -> 89.3%).
- 730 days, tomorrow: normal MAE 30.86 -> 30.43, but spike flags weaker (48% -> 42%), so it stays 365.

Usage: python experiments/training_window.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from run_backtest import load_backtest_frame, walk_forward  # noqa: E402
from src.day_ahead import SPIKE, load_aemo_asof  # noqa: E402

HIGH_DEMAND = 10_000

VARIANTS = {
    "365d (current)": {"train_days": 365},
    "180d": {"train_days": 180},
    "730d": {"train_days": 730},
    "365d, half-life 180d": {"train_days": 365, "half_life_days": 180},
    "365d, half-life 90d": {"train_days": 365, "half_life_days": 90},
}


def summarise(p: pd.DataFrame) -> dict:
    y = p["rrp"]
    n, s = y < SPIKE, y >= SPIKE
    flag = p["rrp_risk_ceiling"] >= SPIKE
    hd = p["totaldemand"] >= HIGH_DEMAND
    return {
        "normal_MAE": (p.loc[n, "rrp_base_prediction"] - y[n]).abs().mean(),
        "normal_bias": (p.loc[n, "rrp_base_prediction"] - y[n]).mean(),
        "q90_coverage": (y <= p["rrp_risk_ceiling"]).mean(),
        "spikes_flagged": (flag & s).sum() / max(s.sum(), 1),
        "flags_real": (flag & s).sum() / max(flag.sum(), 1),
        "hours_flagged": flag.sum() * 5 / 60,
        "HD_coverage": (y[hd] <= p.loc[hd, "rrp_risk_ceiling"]).mean(),
        "HD_spikes_flagged": (flag & s & hd).sum() / max((s & hd).sum(), 1),
    }


if __name__ == "__main__":
    first, last = pd.Period("2025-01", "M"), pd.Period("2026-08", "M")
    df = load_backtest_frame(first, last, train_days=730)
    months = pd.period_range(first, last, freq="M")
    aemo = load_aemo_asof()

    rows, monthly = [], {}
    for name, kw in VARIANTS.items():
        m, preds = walk_forward(df, months, verbose=False, aemo=aemo, **kw)
        preds["year"] = preds["month"].str[:4]
        for lead, g in preds.groupby("lead", sort=False):
            rows.append({"lead": lead, "variant": name, "period": "all", **summarise(g)})
            for year, gy in g.groupby("year"):
                rows.append({"lead": lead, "variant": name, "period": year, **summarise(gy)})
        monthly[name] = m.set_index(["lead", "month"])["normal_MAE"]
        print(f"done: {name}", flush=True)

    res = pd.DataFrame(rows)
    hd = (df["totaldemand"] >= HIGH_DEMAND) & (df["settlementdate"] >= first.start_time)
    print(f"\nHigh-demand intervals (>= {HIGH_DEMAND} MW) in the evaluation period: {hd.sum()}, "
          f"of which spikes: {(hd & (df['rrp'] >= SPIKE)).sum()}")
    pd.set_option("display.width", 220)
    for lead in ["today", "tomorrow"]:
        print(f"\n===== {lead} =====")
        t = res[res["lead"] == lead].drop(columns="lead").set_index(["period", "variant"])
        print(t.round(3).to_string())

    mm = pd.DataFrame(monthly)
    base = mm["365d (current)"]
    print("\n===== Months where the variant beats 365d (normal MAE) =====")
    for lead in ["today", "tomorrow"]:
        print(lead, {c: f"{(mm.loc[lead, c] < base.loc[lead]).sum()}/{len(base.loc[lead])}"
                     for c in mm.columns if c != "365d (current)"})
