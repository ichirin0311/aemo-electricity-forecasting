"""
Benchmark: our walk-forward forecasts vs AEMO's own PD7DAY price forecast, as it stood
when our forecast would have been issued (latest run published before 06:00 on the issue
day, i.e. the previous evening's 18:00 run). Compared at 30-minute resolution, since
PD7DAY is 30-minute: actuals and our 5-minute forecasts are averaged per 30 minutes.

Findings (2025-01..2026-08):
- Only the "today" lead has a meaningful AEMO benchmark. Before the day-ahead bid deadline
  (~12:30), PD7DAY prices for the following days sit at the market price cap in ~10% of
  30-minute intervals (07:30 run: 10.4% one day ahead; 13:00 run: 1.0%). At our 06:00 issue
  time the latest run is D-1 18:00: after the deadline for day D, but not for D+1.
- Today: AEMO is better on typical intervals (median abs. error $15.6 vs $19.4) and flags
  more spikes (61% vs 52%, both ~17% precision), but occasional high false alarms make its
  mean error much worse (normal MAE $106.7 vs $26.2). A plain average of the two when
  AEMO < $300 gives normal MAE $20.8, which motivates using PD7DAY as a model input.

Usage: python experiments/pd7day_benchmark.py   (needs data/backtest/walk_forward_predictions.parquet)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.aemo_pd7day import as_of_issue, load_pd7day_history  # noqa: E402

SPIKE = 300
LEAD_DAYS = {"today": 1, "tomorrow": 2}


def to_30min(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds.assign(interval_datetime=preds["settlementdate"].dt.ceil("30min"))
    cols = ["rrp", "rrp_base_prediction", "rrp_risk_ceiling"]
    out = p.groupby(["lead", "interval_datetime"])[cols].mean().astype("float64").reset_index()
    out["trading_day"] = (out["interval_datetime"] - pd.Timedelta(minutes=1)).dt.normalize()
    return out


def attach_aemo(ours: pd.DataFrame, pd7: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for lead, lead_days in LEAD_DAYS.items():
        o = ours[ours["lead"] == lead]
        for day in o["trading_day"].unique():
            issue_day = day - pd.Timedelta(days=lead_days - 1)
            parts.append(as_of_issue(pd7, issue_day, lead_days).assign(lead=lead))
    aemo = pd.concat(parts, ignore_index=True)
    return ours.merge(aemo, on=["lead", "interval_datetime"], how="inner")


def summary(df: pd.DataFrame) -> dict:
    y = df["rrp"]
    n, s = y < SPIKE, y >= SPIKE
    mae = lambda col, m: (df.loc[m, col] - y[m]).abs().mean()
    flag = lambda col: (df[col] >= SPIKE)
    hits = lambda col: (flag(col) & s).sum()
    ae = lambda col: (df.loc[n, col] - y[n]).abs()
    return {
        "intervals_30min": len(df), "spikes": int(s.sum()),
        "normal_medAE_ours": ae("rrp_base_prediction").median(), "normal_medAE_aemo": ae("aemo_rrp").median(),
        "normal_MAE_ours": mae("rrp_base_prediction", n), "normal_MAE_aemo": mae("aemo_rrp", n),
        "normal_bias_ours": (df.loc[n, "rrp_base_prediction"] - y[n]).mean(),
        "normal_bias_aemo": (df.loc[n, "aemo_rrp"] - y[n]).mean(),
        "spike_MAE_aemo": mae("aemo_rrp", s), "spike_MAE_q90": mae("rrp_risk_ceiling", s),
        "spike_recall_aemo": hits("aemo_rrp") / s.sum(), "spike_recall_q90": hits("rrp_risk_ceiling") / s.sum(),
        "flag_precision_aemo": hits("aemo_rrp") / max(flag("aemo_rrp").sum(), 1),
        "flag_precision_q90": hits("rrp_risk_ceiling") / max(flag("rrp_risk_ceiling").sum(), 1),
    }


if __name__ == "__main__":
    preds = pd.read_parquet("data/backtest/walk_forward_predictions.parquet")
    ours = to_30min(preds)
    first_issue = ours["trading_day"].min() - pd.Timedelta(days=1)
    last_issue = ours["trading_day"].max()
    pd7 = load_pd7day_history(first_issue, last_issue + pd.Timedelta(hours=6))
    print(f"PD7DAY runs loaded: {pd7['run_datetime'].nunique()} "
          f"({pd7['run_datetime'].min()} .. {pd7['run_datetime'].max()})")

    df = attach_aemo(ours, pd7)
    days = df.groupby("lead")["trading_day"].nunique()
    print(f"Matched trading days: {days.to_dict()} (of {ours.groupby('lead')['trading_day'].nunique().to_dict()})")
    used_hours = df["run_datetime"].dt.strftime("%H:%M").value_counts().to_dict()
    print(f"AEMO runs used, by run time: {used_hours}")

    cap_share = df.groupby("lead")["aemo_rrp"].apply(lambda s: (s >= 10000).mean())
    print(f"Share of AEMO prices >= $10,000: {cap_share.round(4).to_dict()} "
          "(tomorrow's run is before the bid deadline, so only 'today' is compared below)")
    df = df[df["lead"] == "today"].copy()

    pd.set_option("display.width", 200)
    print("\n===== Today lead, pooled =====")
    print(pd.Series(summary(df)).round(3).to_string())
    n = df["rrp"] < SPIKE
    blend = np.where(df["aemo_rrp"] < SPIKE, (df["rrp_base_prediction"] + df["aemo_rrp"]) / 2,
                     df["rrp_base_prediction"])
    print(f"Plain average (when AEMO < ${SPIKE}), normal MAE: {np.abs(blend[n] - df.loc[n, 'rrp']).mean():.2f}")

    df["month"] = df["trading_day"].dt.to_period("M")
    monthly = df.groupby("month").apply(
        lambda g: pd.Series({k: summary(g)[k] for k in
                             ["normal_MAE_ours", "normal_MAE_aemo", "normal_medAE_ours", "normal_medAE_aemo", "spikes"]}),
        include_groups=False)
    print("\n===== Today lead, monthly (ours vs AEMO PD7DAY) =====")
    print(monthly.round(1).to_string())
    print("\nMonths where ours beats AEMO: mean abs. error "
          f"{(monthly['normal_MAE_ours'] < monthly['normal_MAE_aemo']).sum()}/{len(monthly)}, median abs. error "
          f"{(monthly['normal_medAE_ours'] < monthly['normal_medAE_aemo']).sum()}/{len(monthly)}")

    out = df[["interval_datetime", "rrp", "rrp_base_prediction", "rrp_risk_ceiling", "aemo_rrp"]]
    out.astype({c: "float32" for c in ["rrp", "rrp_base_prediction", "rrp_risk_ceiling", "aemo_rrp"]}) \
       .to_parquet("data/backtest/pd7day_benchmark_30min.parquet", index=False, compression="zstd")
