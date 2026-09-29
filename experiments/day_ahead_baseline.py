"""
Day-ahead feasibility experiment (v2, stage 2).

The production models (src/train.py) use same-interval actuals (totaldemand,
availablegeneration, reserve_margin, temperature) and lags as short as 5 minutes,
which are not available when a forecast is issued in advance. This script
re-measures accuracy using only information available at issue time, on the
same 2025 split as train.py (Jan-Oct train / Nov val / Dec test) so the numbers
are directly comparable.

Issue-time assumption: the forecast is issued on the morning of day D, and AEMO's
PRICE_AND_DEMAND CSV is complete up to D 00:00 (confirmed: the file is refreshed
at 00:00 market time). Two targets are evaluated:
  - lead_days=1: forecast day D     (6-24h ahead, latest complete day = D-1)
  - lead_days=2: forecast day D+1   (24-48h ahead, latest complete day = D-1)
i.e. for target day T, the latest complete day of actuals is T - lead_days.

Temperature uses actuals as a stand-in for a weather forecast (optimistic, but
1-2 day temperature forecasts are fairly accurate).

Usage: python experiments/day_ahead_baseline.py
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.day_ahead import (  # noqa: E402  (features/params shared with production)
    DEMAND_FEATURES, PARAMS_BASE, PARAMS_DEMAND, PARAMS_Q90, SPIKE,
    add_calendar, build_features, price_inputs,
)

PARQUET = "data/processed/cleansed_aemo_NSW_2025.parquet"
# This experiment predates the L1 / scaled-asinh central model; keep its original L2 setup
PARAMS_BASE = {**PARAMS_BASE, "objective": "regression", "metric": "rmse"}


def fit_predict(params, X_tr, y_tr, X_va, y_va, X_te):
    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
                      valid_sets=[lgb.Dataset(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, model.predict(X_te, num_iteration=model.best_iteration)


def price_report(name, y, pred):
    n, s = y < SPIKE, y >= SPIKE
    return {
        "model": name,
        "normal_MAE": mean_absolute_error(y[n], pred[n]),
        "normal_R2": r2_score(y[n], pred[n]),
        "spike_MAE": mean_absolute_error(y[s], pred[s]),
    }


def run(df_raw: pd.DataFrame, lead_days: int):
    df = df_raw[["settlementdate", "rrp", "totaldemand", "temperature"]].copy()
    df = build_features(add_calendar(df, "NSW"), lead_days).dropna().reset_index(drop=True)
    tr, va, te = df[df.month <= 10], df[df.month == 11], df[df.month == 12]

    # --- Demand ---
    _, pred_d = fit_predict(PARAMS_DEMAND, tr[DEMAND_FEATURES], tr.totaldemand,
                            va[DEMAND_FEATURES], va.totaldemand, te[DEMAND_FEATURES])
    demand = {
        "model_R2": r2_score(te.totaldemand, pred_d),
        "naive7d_R2": r2_score(te.totaldemand, te.totaldemand_same_time_7d),
        "naive_latest_R2": r2_score(te.totaldemand, te.totaldemand_same_time_latest),
    }

    # --- Price ---
    X = price_inputs
    y = te.rrp.values
    m_base, pred_b = fit_predict(PARAMS_BASE, X(tr), np.arcsinh(tr.rrp), X(va), np.arcsinh(va.rrp), X(te))
    _, pred_q = fit_predict(PARAMS_Q90, X(tr), np.arcsinh(tr.rrp), X(va), np.arcsinh(va.rrp), X(te))
    pred_b, pred_q = np.sinh(pred_b), np.sinh(pred_q)

    rows = [
        price_report("central (asinh)", y, pred_b),
        price_report("Q90 risk ceiling", y, pred_q),
        price_report("naive: same time 7d ago", y, te.rrp_same_time_7d.values),
        price_report("naive: same time latest day", y, te.rrp_same_time_latest.values),
    ]
    price = pd.DataFrame(rows).set_index("model")
    coverage = (y <= pred_q).mean()

    importance = pd.Series(m_base.feature_importance("gain"), index=m_base.feature_name())
    return demand, price, coverage, importance.sort_values(ascending=False), (y >= SPIKE).sum()


if __name__ == "__main__":
    df_raw = pd.read_parquet(PARQUET)
    pd.set_option("display.width", 120)
    for lead_days, label in [(1, "Target = today (D), 6-24h ahead"),
                             (2, "Target = tomorrow (D+1), 24-48h ahead")]:
        demand, price, coverage, importance, n_spike = run(df_raw, lead_days)
        print(f"\n===== {label} =====")
        print("Demand R2: " + ", ".join(f"{k}={v:.4f}" for k, v in demand.items()))
        print(f"Price (test=Dec 2025, spikes={n_spike}):")
        print(price.round(4).to_string())
        print(f"Q90 coverage: {coverage:.4f}")
        print("Top-8 features (central model, gain):")
        print((importance.head(8) / importance.sum()).round(3).to_string())
