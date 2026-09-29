# run_backtest.py
# v2 stage 3: walk-forward evaluation of the day-ahead models (src/day_ahead.py).
# For each month, models are trained on the 365 days before the month starts and
# then forecast every day of that month (both leads). Retraining monthly rather
# than daily (as in production) makes this slightly conservative.
# Temperature uses archive actuals (a forecast isn't available historically).
#
# Usage: python run_backtest.py [first_month] [last_month]   e.g. 2025-01 2026-08
# Outputs: data/backtest/walk_forward_monthly.csv, data/backtest/walk_forward_predictions.parquet
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.day_ahead import (LEADS, SPIKE, assemble_frame, build_features, load_aemo_actuals,
                           predict_lead, train_lead_models)

REGION, LAT, LON = "NSW", -33.86, 151.20
TRAIN_DAYS = 365
OUT_DIR = os.path.join("data", "backtest")


def metrics(df: pd.DataFrame) -> dict:
    y, base, q90, naive = df["rrp"], df["rrp_base_prediction"], df["rrp_risk_ceiling"], df["naive_rrp"]
    n, s = y < SPIKE, y >= SPIKE
    spike = lambda pred: mean_absolute_error(y[s], pred[s]) if s.any() else np.nan
    return {
        "intervals": len(df),
        "spikes": int(s.sum()),
        "demand_R2": r2_score(df["totaldemand"], df["demand_forecast"]),
        "demand_R2_naive": r2_score(df["totaldemand"], df["naive_demand"]),
        "normal_MAE": mean_absolute_error(y[n], base[n]),
        "normal_MAE_naive": mean_absolute_error(y[n], naive[n]),
        "normal_R2": r2_score(y[n], base[n]),
        "spike_MAE_central": spike(base),
        "spike_MAE_q90": spike(q90),
        "spike_MAE_naive": spike(naive),
        "q90_coverage": (y <= q90).mean(),
    }


def load_backtest_frame(first: pd.Period, last: pd.Period) -> pd.DataFrame:
    start = first.start_time - pd.Timedelta(days=TRAIN_DAYS + 7 + max(LEADS.values()) + 1)
    end = (last + 1).start_time  # last interval of the last month is 00:00 of the next month
    actuals = load_aemo_actuals(f"{REGION}1", start, end)
    return assemble_frame(actuals, REGION, cutoff=end, horizon_end=end, lat=LAT, lon=LON)


def walk_forward(df: pd.DataFrame, months, leads=LEADS, verbose=True, **train_kwargs):
    """Monthly-retrained walk-forward. Returns (monthly metrics, all predictions)."""
    preds, rows = [], []
    for lead_name, lead_days in leads.items():
        feat = build_features(df, lead_days)
        for month in months:
            cutoff = month.start_time
            models, _ = train_lead_models(feat, cutoff, train_days=TRAIN_DAYS, **train_kwargs)
            target = feat[(feat["settlementdate"] > cutoff)
                          & (feat["settlementdate"] <= (month + 1).start_time)].dropna()
            p = predict_lead(models, target)
            p["rrp"], p["totaldemand"] = target["rrp"].values, target["totaldemand"].values
            p["naive_rrp"] = target["rrp_same_time_latest"].values
            p["naive_demand"] = target["totaldemand_same_time_latest"].values
            p.insert(0, "lead", lead_name)
            p.insert(0, "month", str(month))
            preds.append(p)
            rows.append({"month": str(month), "lead": lead_name, **metrics(p)})
            if verbose:
                print(f"{lead_name:8s} {month}: normal MAE={rows[-1]['normal_MAE']:.2f} "
                      f"(naive {rows[-1]['normal_MAE_naive']:.2f}), coverage={rows[-1]['q90_coverage']:.3f}, "
                      f"spikes={rows[-1]['spikes']}")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


if __name__ == "__main__":
    first = pd.Period(sys.argv[1] if len(sys.argv) > 1 else "2025-01", freq="M")
    last = pd.Period(sys.argv[2] if len(sys.argv) > 2 else "2026-08", freq="M")
    df = load_backtest_frame(first, last)
    monthly, all_preds = walk_forward(df, pd.period_range(first, last, freq="M"))

    os.makedirs(OUT_DIR, exist_ok=True)
    monthly.to_csv(os.path.join(OUT_DIR, "walk_forward_monthly.csv"), index=False, float_format="%.6f")
    all_preds.to_parquet(os.path.join(OUT_DIR, "walk_forward_predictions.parquet"), index=False)

    pd.set_option("display.width", 160)
    print("\n===== Pooled over all months =====")
    pooled = pd.DataFrame({lead: metrics(g) for lead, g in all_preds.groupby("lead", sort=False)}).T
    print(pooled.round(4).to_string())
