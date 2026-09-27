"""
v2: day-ahead forecasting on a rolling window.

Every morning (market time) the forecast is issued for:
  - lead "today"    (lead_days=1): trading day D     -> 6-24h ahead at a 06:00 run
  - lead "tomorrow" (lead_days=2): trading day D+1   -> 24-48h ahead
using only information available at issue time: AEMO actuals up to D 00:00
(the PRICE_AND_DEMAND CSV is refreshed at 00:00 market time), calendar, and a
temperature forecast. Same-interval demand/capacity actuals used by the v1
models (src/train.py) are deliberately NOT used, since they are unknown in
advance. See experiments/day_ahead_baseline.py for the feasibility numbers.
"""
import os
from datetime import datetime

import holidays
import lightgbm as lgb
import numpy as np
import openmeteo_requests
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, r2_score

from src.aemo_downloader import AEMO_TZ, download_price_and_demand

SPIKE = 300
INTERVALS_PER_DAY = 288
LEADS = {"today": 1, "tomorrow": 2}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CALENDAR = ["month", "hour", "minute_of_day", "day_of_week", "is_holiday", "season"]
WEATHER = ["temperature", "temp_day_max", "temp_day_min"]
DEMAND_FEATURES = CALENDAR + WEATHER + [
    "totaldemand_same_time_latest", "totaldemand_same_time_7d", "demand_day_mean", "demand_day_max",
]
RRP_FEATURES = DEMAND_FEATURES + [
    "rrp_same_time_latest", "rrp_same_time_7d", "rrp_day_mean", "rrp_day_max", "rrp_day_std",
    "rrp_day_spike_share", "rrp_day_mean_7d", "rrp_day_max_7d",
]
# Price-valued inputs are fed in asinh space, like the target
RRP_ASINH_COLS = ["rrp_same_time_latest", "rrp_same_time_7d", "rrp_day_mean", "rrp_day_max",
                  "rrp_day_mean_7d", "rrp_day_max_7d"]

# Same hyperparameters as the v1 models in src/train.py
PARAMS_DEMAND = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05,
                 "num_leaves": 31, "seed": 42, "verbose": -1}
PARAMS_BASE = {"objective": "regression", "metric": "rmse", "learning_rate": 0.03,
               "num_leaves": 15, "min_data_in_leaf": 100, "seed": 42, "verbose": -1}
PARAMS_Q90 = {"objective": "quantile", "alpha": 0.9, "metric": "quantile", "learning_rate": 0.05,
              "num_leaves": 31, "seed": 42, "verbose": -1}


# ---------------------------------------------------------------- data

def load_aemo_actuals(region_id: str, start: pd.Timestamp, end: pd.Timestamp,
                      raw_dir: str = os.path.join("data", "raw", "aemo_data_1year")) -> pd.DataFrame:
    """AEMO 5-min price/demand actuals for start < settlementdate <= end (market time)."""
    months = pd.period_range(start, end, freq="M").strftime("%Y%m").tolist()
    files = download_price_and_demand(region_id, months, raw_dir)
    if not files:
        raise FileNotFoundError(f"No AEMO CSVs available for {months[0]}-{months[-1]}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.lower()
    df = df[df["region"].str.upper() == region_id]
    df["settlementdate"] = pd.to_datetime(df["settlementdate"])
    df = df[(df["settlementdate"] > start) & (df["settlementdate"] <= end)]
    return (df[["settlementdate", "totaldemand", "rrp"]]
            .drop_duplicates("settlementdate").sort_values("settlementdate").reset_index(drop=True))


def _hourly_temperature(client, url: str, params: dict) -> pd.Series:
    resp = client.weather_api(url, params={**params, "hourly": "temperature_2m"})[0]
    hourly = resp.Hourly()
    index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    # AEMO market time is fixed UTC+10 (no DST) -> Etc/GMT-10, not Australia/Sydney
    index = index.tz_convert("Etc/GMT-10").tz_localize(None)
    return pd.Series(hourly.Variables(0).ValuesAsNumpy(), index=index, name="temperature")


def load_temperature(start: pd.Timestamp, end: pd.Timestamp, cutoff: pd.Timestamp,
                     lat: float, lon: float) -> pd.Series:
    """
    5-min temperature for start..end. Up to the cutoff, archive values (actuals);
    after the cutoff, the forecast API (what is actually known at issue time).
    For a historical cutoff (backtest) the forecast API can't reach back, so
    archive actuals are used throughout (slightly optimistic).
    """
    client = openmeteo_requests.Client(session=requests.Session())
    # The archive rejects end_date beyond today (UTC); later hours come from the forecast API
    archive_end = min(end + pd.Timedelta(days=1), pd.Timestamp.now(tz="UTC").tz_localize(None))
    temp = _hourly_temperature(client, ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": archive_end.strftime("%Y-%m-%d"),
    })

    now = pd.Timestamp(datetime.now(AEMO_TZ).replace(tzinfo=None))
    if now - cutoff < pd.Timedelta(days=7):
        forecast = _hourly_temperature(client, FORECAST_URL, {
            "latitude": lat, "longitude": lon, "past_days": 7, "forecast_days": 3,
        })
        temp = temp[temp.index <= cutoff]
        forecast = forecast[forecast.index > cutoff]
        temp = pd.concat([temp, forecast])
        # Fill any archive gaps near the cutoff with the forecast run's values
        temp = temp.groupby(level=0).first()

    temp = temp.sort_index().resample("5min").interpolate(method="linear")
    return temp[(temp.index > start) & (temp.index <= end)]


# ---------------------------------------------------------------- features

def add_calendar(df: pd.DataFrame, region: str) -> pd.DataFrame:
    ts = df["settlementdate"]
    years = range(ts.dt.year.min(), ts.dt.year.max() + 1)
    au_holidays = holidays.Australia(subdiv=region, years=years)
    df["month"] = ts.dt.month
    df["hour"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["is_holiday"] = ts.dt.date.map(lambda d: int(d in au_holidays))
    df["season"] = df["month"].map(lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8] else 3)))
    return df


def build_features(df: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    """
    df: settlementdate, rrp, totaldemand, temperature + calendar columns. Future
    rows (to be forecast) have NaN rrp/totaldemand. For a row in trading day T,
    every feature only uses actuals up to the end of day T - lead_days, so the
    same features can be computed for training rows and for forecast rows.
    """
    df = df.set_index("settlementdate").copy()

    lag = pd.Timedelta(days=lead_days)
    for col in ["rrp", "totaldemand"]:
        df[f"{col}_same_time_latest"] = df[col].shift(freq=lag).reindex(df.index)
        df[f"{col}_same_time_7d"] = df[col].shift(freq="7D").reindex(df.index)

    # Interval-ending convention: 00:00 belongs to the previous trading day
    df["trading_day"] = (df.index - pd.Timedelta(minutes=5)).normalize()

    actual = df[df["rrp"].notna()]
    daily = actual.groupby("trading_day").agg(
        rrp_day_mean=("rrp", "mean"),
        rrp_day_max=("rrp", "max"),
        rrp_day_std=("rrp", "std"),
        rrp_day_spike_share=("rrp", lambda s: (s >= SPIKE).mean()),
        demand_day_mean=("totaldemand", "mean"),
        demand_day_max=("totaldemand", "max"),
        n=("rrp", "size"),
    )
    daily = daily[daily["n"] == INTERVALS_PER_DAY].drop(columns="n")  # complete days only
    daily = daily.join(daily[["rrp_day_mean", "rrp_day_max"]].rolling(7).mean().add_suffix("_7d"))
    daily = daily.shift(lead_days, freq="D")  # stats of day T - lead_days, keyed by target day T
    df = df.join(daily, on="trading_day")

    # Target-day temperature range (a forecast at issue time)
    df["temp_day_max"] = df.groupby("trading_day")["temperature"].transform("max")
    df["temp_day_min"] = df.groupby("trading_day")["temperature"].transform("min")
    return df.reset_index()


def price_inputs(df: pd.DataFrame) -> pd.DataFrame:
    x = df[RRP_FEATURES].copy()
    for c in RRP_ASINH_COLS:
        x[c] = np.arcsinh(x[c])
    return x


# ---------------------------------------------------------------- models

def _fit(params, X_tr, y_tr, X_va, y_va):
    return lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=1000,
                     valid_sets=[lgb.Dataset(X_va, y_va)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def train_lead_models(df_feat: pd.DataFrame, cutoff: pd.Timestamp,
                      train_days: int = 365, val_days: int = 28) -> tuple[dict, dict]:
    """
    Train demand / central price / Q90 models on the train_days before the cutoff;
    the last val_days are held out for early stopping and reported as metrics.
    """
    hist = df_feat[(df_feat["settlementdate"] <= cutoff)
                   & (df_feat["settlementdate"] > cutoff - pd.Timedelta(days=train_days))]
    hist = hist.dropna(subset=RRP_FEATURES + ["rrp", "totaldemand"])
    val_start = cutoff - pd.Timedelta(days=val_days)
    tr, va = hist[hist["settlementdate"] <= val_start], hist[hist["settlementdate"] > val_start]

    models = {
        "demand": _fit(PARAMS_DEMAND, tr[DEMAND_FEATURES], tr["totaldemand"],
                       va[DEMAND_FEATURES], va["totaldemand"]),
        "base": _fit(PARAMS_BASE, price_inputs(tr), np.arcsinh(tr["rrp"]),
                     price_inputs(va), np.arcsinh(va["rrp"])),
        "q90": _fit(PARAMS_Q90, price_inputs(tr), np.arcsinh(tr["rrp"]),
                    price_inputs(va), np.arcsinh(va["rrp"])),
    }

    # Validation metrics (last val_days) for monitoring
    y = va["rrp"].values
    base = np.sinh(models["base"].predict(price_inputs(va), num_iteration=models["base"].best_iteration))
    q90 = np.sinh(models["q90"].predict(price_inputs(va), num_iteration=models["q90"].best_iteration))
    normal = y < SPIKE
    metrics = {
        "train_rows": len(tr), "val_rows": len(va),
        "demand_R2": r2_score(va["totaldemand"], models["demand"].predict(
            va[DEMAND_FEATURES], num_iteration=models["demand"].best_iteration)),
        "normal_MAE": mean_absolute_error(y[normal], base[normal]),
        "normal_R2": r2_score(y[normal], base[normal]) if normal.sum() > 1 else np.nan,
        "q90_coverage": (y <= q90).mean(),
        "val_spikes": int((~normal).sum()),
    }
    return models, metrics


def predict_lead(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[["settlementdate"]].copy()
    out["demand_forecast"] = models["demand"].predict(
        rows[DEMAND_FEATURES], num_iteration=models["demand"].best_iteration)
    out["rrp_base_prediction"] = np.sinh(models["base"].predict(
        price_inputs(rows), num_iteration=models["base"].best_iteration))
    out["rrp_risk_ceiling"] = np.sinh(models["q90"].predict(
        price_inputs(rows), num_iteration=models["q90"].best_iteration))
    return out


# ---------------------------------------------------------------- orchestration

def latest_cutoff(actuals: pd.DataFrame) -> pd.Timestamp:
    """The end of the latest complete trading day in the data (a 00:00 timestamp)."""
    last = actuals["settlementdate"].max()
    return last if last == last.normalize() else last.normalize()


def run_day_ahead(region: str = "NSW", lat: float = -33.86, lon: float = 151.20,
                  cutoff: pd.Timestamp | None = None, train_days: int = 365) -> tuple[pd.DataFrame, dict]:
    """
    Issue the "today" + "tomorrow" forecast. cutoff = D 00:00 (market time): actuals
    up to and including the cutoff are used. None -> the latest complete day
    available from AEMO (normally today's 00:00). Pass a past cutoff to backtest.
    """
    region_id = f"{region.upper()}1"
    now = pd.Timestamp(datetime.now(AEMO_TZ).replace(tzinfo=None))
    history_days = train_days + 7 + max(LEADS.values()) + 1  # window + 7d lag + lead lag
    requested = cutoff if cutoff is not None else now.normalize()

    actuals = load_aemo_actuals(region_id, requested - pd.Timedelta(days=history_days), requested)
    if cutoff is None:
        cutoff = latest_cutoff(actuals)
        if cutoff != now.normalize():
            print(f"WARNING: AEMO data only complete up to {cutoff}; forecasting relative to that day")
    actuals = actuals[actuals["settlementdate"] <= cutoff]

    horizon_end = cutoff + pd.Timedelta(days=max(LEADS.values()))
    future = pd.DataFrame({"settlementdate": pd.date_range(
        cutoff + pd.Timedelta(minutes=5), horizon_end, freq="5min")})
    df = pd.concat([actuals, future], ignore_index=True)

    temp = load_temperature(df["settlementdate"].min() - pd.Timedelta(minutes=5), horizon_end, cutoff, lat, lon)
    df = df.merge(temp.rename_axis("settlementdate").reset_index(), on="settlementdate", how="left")
    df = add_calendar(df, region)

    issue_day = cutoff.normalize()
    forecasts, report = [], {"cutoff": cutoff}
    for lead_name, lead_days in LEADS.items():
        feat = build_features(df, lead_days)
        models, metrics = train_lead_models(feat, cutoff, train_days=train_days)
        report[lead_name] = metrics

        day_start = issue_day + pd.Timedelta(days=lead_days - 1)
        rows = feat[(feat["settlementdate"] > day_start)
                    & (feat["settlementdate"] <= day_start + pd.Timedelta(days=1))]
        missing = rows[RRP_FEATURES].isna().any(axis=1)
        if missing.any():
            raise ValueError(f"{lead_name}: {missing.sum()} forecast rows have missing features "
                             f"(e.g. {rows.loc[missing, 'settlementdate'].iloc[0]})")
        pred = predict_lead(models, rows)
        pred.insert(0, "lead", lead_name)
        forecasts.append(pred)

    forecast = pd.concat(forecasts, ignore_index=True)
    forecast.insert(0, "issue_date", issue_day.date())
    return forecast, report


def save_forecast(forecast: pd.DataFrame, out_dir: str = os.path.join("data", "forecasts")) -> None:
    """Write the latest forecast and append it to the forecast log (one issue per date)."""
    os.makedirs(out_dir, exist_ok=True)
    fmt = {"index": False, "float_format": "%.6f"}  # rounding avoids runner-CPU float noise
    forecast.to_csv(os.path.join(out_dir, "latest_forecast.csv"), **fmt)

    log_path = os.path.join(out_dir, "forecast_log.csv")
    log = forecast
    if os.path.exists(log_path):
        prev = pd.read_csv(log_path, parse_dates=["settlementdate"])
        prev["issue_date"] = pd.to_datetime(prev["issue_date"]).dt.date
        prev = prev[prev["issue_date"] != forecast["issue_date"].iloc[0]]  # re-runs replace the same issue
        log = pd.concat([prev, forecast], ignore_index=True)
    log.sort_values(["issue_date", "lead", "settlementdate"]).to_csv(log_path, **fmt)
