"""
Data loading and analysis for the v2 dashboard (src/app.py). No Streamlit here, so
everything can be checked without launching the app. Reads only files committed to
the repo (the daily GitHub Actions run refreshes them), so Streamlit Community Cloud
never calls AEMO or Open-Meteo itself.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

RAW_DIR = os.path.join("data", "raw", "aemo_data_1year")
FORECAST_DIR = os.path.join("data", "forecasts")
BACKTEST_DIR = os.path.join("data", "backtest")

LEAD_DAYS = {"today": 1, "tomorrow": 2}
INTERVAL_HOURS = 5 / 60  # 5-minute interval: MW -> MWh

# First issue date produced by the current central model (L1 on asinh(rrp/100)).
# Earlier live forecasts came from the L2 setup that under-predicted (see CLAUDE.md, v2).
CURRENT_MODEL_SINCE = pd.Timestamp("2026-09-30")


# ---------------------------------------------------------------- loading

def load_actuals(raw_dir: str = RAW_DIR, region_id: str = "NSW1") -> pd.DataFrame:
    """All committed AEMO 5-min actuals: settlementdate, rrp, totaldemand."""
    files = sorted(glob.glob(os.path.join(raw_dir, f"PRICE_AND_DEMAND_*_{region_id}.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.lower()
    df["settlementdate"] = pd.to_datetime(df["settlementdate"])
    return (df[["settlementdate", "rrp", "totaldemand"]]
            .drop_duplicates("settlementdate").sort_values("settlementdate").reset_index(drop=True))


def load_forecasts(forecast_dir: str = FORECAST_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(latest forecast, full forecast log)."""
    read = lambda name: pd.read_csv(os.path.join(forecast_dir, name),
                                    parse_dates=["issue_date", "settlementdate"])
    return read("latest_forecast.csv"), read("forecast_log.csv")


def load_backtest(backtest_dir: str = BACKTEST_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly walk-forward metrics, interval-level walk-forward predictions)."""
    monthly = pd.read_csv(os.path.join(backtest_dir, "walk_forward_monthly.csv"))
    preds = pd.read_parquet(os.path.join(backtest_dir, "walk_forward_predictions.parquet"))
    return monthly, preds.rename(columns={"rrp": "rrp_actual"})


# ---------------------------------------------------------------- forecast track record

def score_forecast_log(log: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast log joined with actuals (only intervals that have happened), plus a naive
    benchmark: the actual price at the same time on the latest complete day at issue
    (i.e. lead_days earlier), which is what the model has to beat.
    """
    a = actuals.set_index("settlementdate")
    scored = log.merge(actuals.rename(columns={"rrp": "rrp_actual", "totaldemand": "demand_actual"}),
                       on="settlementdate", how="inner")
    lag = scored["lead"].map(LEAD_DAYS).map(lambda d: pd.Timedelta(days=d))
    scored["naive_rrp"] = (scored["settlementdate"] - lag).map(a["rrp"])
    scored["current_model"] = scored["issue_date"] >= CURRENT_MODEL_SINCE
    return scored


def price_metrics(df: pd.DataFrame, spike_threshold: float, actual: str = "rrp_actual") -> dict:
    """Two-axis accuracy: central model on normal intervals, Q90 coverage on all intervals."""
    y = df[actual]
    normal, spike = y < spike_threshold, y >= spike_threshold
    mae = lambda col, m: mean_absolute_error(y[m], df.loc[m, col]) if m.any() else np.nan
    out = {
        "intervals": len(df),
        "n_spike": int(spike.sum()),
        "normal_mae": mae("rrp_base_prediction", normal),
        "normal_bias": (df.loc[normal, "rrp_base_prediction"] - y[normal]).mean() if normal.any() else np.nan,
        "spike_mae_q90": mae("rrp_risk_ceiling", spike),
        "coverage": (y <= df["rrp_risk_ceiling"]).mean() if len(df) else np.nan,
    }
    if "naive_rrp" in df:
        ok = normal & df["naive_rrp"].notna()
        out["normal_mae_naive"] = mae("naive_rrp", ok)
    return out


# ---------------------------------------------------------------- market regime

def monthly_regime(actuals: pd.DataFrame, spike_threshold: float) -> pd.DataFrame:
    """Per-month price behaviour. Interval-ending convention: 00:00 belongs to the previous day."""
    df = actuals.assign(month=(actuals["settlementdate"] - pd.Timedelta(minutes=5)).dt.to_period("M"))
    g = df.groupby("month")["rrp"]
    out = pd.DataFrame({
        "intervals": g.size(),
        "mean_price": g.mean(),
        "median_price": g.median(),
        "max_price": g.max(),
        "spikes": g.apply(lambda s: int((s >= spike_threshold).sum())),
        "spike_share": g.apply(lambda s: (s >= spike_threshold).mean()),
        "negative_share": g.apply(lambda s: (s < 0).mean()),
    })
    out.index = out.index.to_timestamp()
    return out


def spike_drift(actuals: pd.DataFrame, spike_threshold: float,
                recent_days: int = 30, baseline_days: int = 365) -> dict:
    """
    Compare the spike rate of the last recent_days with the baseline_days before them
    (roughly the models' training window). A large ratio in either direction means the
    market no longer looks like what the models learned from.
    """
    end = actuals["settlementdate"].max()
    recent_start = end - pd.Timedelta(days=recent_days)
    base_start = recent_start - pd.Timedelta(days=baseline_days)
    t = actuals["settlementdate"]
    recent = actuals.loc[t > recent_start, "rrp"]
    base = actuals.loc[(t > base_start) & (t <= recent_start), "rrp"]
    per_day = lambda s, days: (s >= spike_threshold).sum() / days
    recent_rate, base_rate = per_day(recent, recent_days), per_day(base, baseline_days)
    ratio = recent_rate / base_rate if base_rate > 0 else np.nan
    if np.isnan(ratio):
        level = "unknown"
    elif ratio < 0.5:
        level = "calmer"
    elif ratio > 2:
        level = "spikier"
    else:
        level = "similar"
    return {"end": end, "recent_days": recent_days, "baseline_days": baseline_days,
            "recent_per_day": recent_rate, "baseline_per_day": base_rate,
            "ratio": ratio, "level": level,
            "recent_negative_share": (recent < 0).mean(), "baseline_negative_share": (base < 0).mean()}


def period_profile(actuals: pd.DataFrame, start, end, spike_threshold: float) -> dict:
    """Summary of one period for side-by-side comparison: hourly profile and price duration curve."""
    t = actuals["settlementdate"] - pd.Timedelta(minutes=5)  # trading-day convention
    df = actuals[(t.dt.date >= start) & (t.dt.date <= end)]
    y = df["rrp"]
    hourly = df.groupby(df["settlementdate"].dt.hour)["rrp"].median()
    duration = np.sort(y.values)[::-1]
    return {
        "days": (end - start).days + 1,
        "intervals": len(df),
        "mean_price": y.mean(),
        "median_price": y.median(),
        "spikes": int((y >= spike_threshold).sum()),
        "spikes_per_day": (y >= spike_threshold).sum() / max((end - start).days + 1, 1),
        "negative_share": (y < 0).mean(),
        "max_price": y.max(),
        "hourly_median": hourly,
        "duration_pct": np.linspace(0, 100, len(duration)),
        "duration_price": duration,
    }


# ---------------------------------------------------------------- risk strategy simulation

def simulate_hedging_strategy(df: pd.DataFrame, risk_threshold: float, hedge_ratio: float) -> pd.DataFrame:
    """
    Proof of concept: curtail hedge_ratio of demand whenever the Q90 risk ceiling forecast
    is at or above risk_threshold. Needs rrp_actual, totaldemand, rrp_risk_ceiling.
    """
    df = df.sort_values("settlementdate").copy()
    # Backtest predictions are stored as float32; cumulative sums reach ~1e10, so use float64
    cols = ["rrp_actual", "totaldemand", "rrp_risk_ceiling"]
    df[cols] = df[cols].astype("float64")
    df["cost_naive"] = df["rrp_actual"] * df["totaldemand"] * INTERVAL_HOURS
    df["is_high_risk"] = df["rrp_risk_ceiling"] >= risk_threshold
    hedged_demand = np.where(df["is_high_risk"], df["totaldemand"] * (1 - hedge_ratio), df["totaldemand"])
    df["cost_hedged"] = df["rrp_actual"] * hedged_demand * INTERVAL_HOURS
    df["cumulative_cost_naive"] = df["cost_naive"].cumsum()
    df["cumulative_cost_hedged"] = df["cost_hedged"].cumsum()
    return df


def alert_quality(sim: pd.DataFrame, spike_threshold: float) -> dict:
    """How well the warning (Q90 >= warning line) lined up with actual spikes."""
    alert, spike = sim["is_high_risk"], sim["rrp_actual"] >= spike_threshold
    hits = int((alert & spike).sum())
    return {
        "alerts": int(alert.sum()),
        "spikes": int(spike.sum()),
        "spikes_caught": hits / spike.sum() if spike.any() else np.nan,        # recall
        "alerts_on_spikes": hits / alert.sum() if alert.any() else np.nan,     # precision
    }
