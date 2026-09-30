# run_day_ahead.py
# v2: issue the rolling-window "today" + "tomorrow" forecast (see src/day_ahead.py).
# Usage:
#   python run_day_ahead.py                 # live: forecast from the latest complete AEMO day
#   python run_day_ahead.py 2025-12-01      # backtest: as if issued on the morning of that date
#                                           #   (prints only; does not touch data/forecasts/)
import sys

import pandas as pd

from src.day_ahead import run_day_ahead, save_forecast

if __name__ == "__main__":
    cutoff = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else None
    forecast, report = run_day_ahead(region="NSW", lat=-33.86, lon=151.20, cutoff=cutoff)

    print(f"\nCutoff (actuals up to): {report['cutoff']}")
    if "aemo_run" in report:
        print(f"AEMO PD7DAY outlook logged: run {report['aemo_run']}")
    print(f"AEMO outlook used for today's price models: {report.get('aemo_today', False)}")
    for lead in ["today", "tomorrow"]:
        m = report[lead]
        print(f"[{lead}] validation (last 28 days, {m['val_spikes']} spikes): "
              f"demand R2={m['demand_R2']:.4f}, normal MAE={m['normal_MAE']:.2f}, "
              f"normal R2={m['normal_R2']:.4f}, Q90 coverage={m['q90_coverage']:.4f}")

    summary = forecast.groupby("lead", sort=False).agg(
        start=("settlementdate", "min"), end=("settlementdate", "max"),
        base_mean=("rrp_base_prediction", "mean"), ceiling_max=("rrp_risk_ceiling", "max"),
        demand_peak=("demand_forecast", "max"),
    )
    print(summary.round({"base_mean": 2, "ceiling_max": 2, "demand_peak": 2}).to_string())

    if cutoff is None:
        save_forecast(forecast)
        print("Saved to data/forecasts/latest_forecast.csv and appended to forecast_log.csv")
