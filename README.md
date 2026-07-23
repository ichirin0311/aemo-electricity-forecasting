# AEMO Electricity Price Forecasting

A machine learning pipeline that predicts NSW electricity prices (RRP) and
demand using AEMO market data, Bureau of Meteorology-sourced weather data
(via Open-Meteo), and generation capacity data. Includes an interactive
Streamlit dashboard that translates model predictions into a business
risk-hedging scenario.

**[Live Dashboard](https://aemo-electricity-forecasting-xhtf7sj4xqjesgknueoqza.streamlit.app/)** 

## What this project does

Electricity prices in Australia's National Electricity Market (NEM) can
swing from ~$50/MWh to over $20,000/MWh within minutes when supply is
tight. This project builds a forecasting system that:

1. Predicts a **central price estimate** for planning purposes
2. Predicts a **90th-percentile risk ceiling** for worst-case scenario planning
3. Simulates the cost impact of a simple risk-hedging strategy based on
   these predictions


## Current limitations & roadmap

This pipeline currently retrains on a fixed 2025 dataset daily via GitHub
Actions (functioning as an automated regression test to catch API/schema
changes early). A planned v2 will shift to a rolling window with automated
AEMO data ingestion and true next-day price forecasting.


## Pipeline overview

```
AEMO price/demand data ──┐
Open-Meteo weather data ─┼──> feature engineering ──> LightGBM models ──> Streamlit dashboard
AEMO capacity data ──────┘
```

- **`main.py`**: runs the full data pipeline (fetch, merge, feature engineering)
- **`src/pipeline.py`**: `AEMODataPipeline` class — data collection & merging
- **`src/train.py`**: trains the demand model and the dual price models
  (central estimate + 90th-percentile risk ceiling)
- **`src/app.py`**: Streamlit dashboard
- **`src/visualize.py`**: static HTML report generator

## Key modeling decisions

- **arcsinh transform** instead of log transform for the price target,
  to avoid compressing normal-range price variation given the market's
  price floor (-$1,000/MWh)
- **Two separate models** (central estimate tuned for normal conditions,
  quantile regression for tail risk) rather than a single blended model —
  several blending/routing approaches were tested and underperformed a
  simple dual-model setup
- **Supply-side features** (available generation capacity, reserve margin)
  sourced via [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS), which
  turned out to be the most important predictors for price spikes

## Results (NSW1, trained on 2025 data, tested on December 2025)

| Metric | Value |
|---|---|
| Demand forecast R² | 0.9665 |
| Price forecast R² (normal conditions, <$300/MWh) | 0.5667 |
| Price forecast MAE (normal conditions) | $19.22 |
| Risk ceiling model MAE (spike conditions) | $1,790.42 |

## Setup

### Option 1: Just try the dashboard (no data setup required)

The dashboard reads pre-computed predictions already included in this repo
(`data/processed/rrp_dual_prediction_output.csv`).

​```bash
pip install -r requirements.txt
python -m streamlit run src/app.py
​```

### Option 2: Run the full pipeline (fetch fresh data, retrain models)

Requires AEMO Price and Demand CSVs placed in `data/raw/aemo_data_1year/`
(download from [AEMO's data portal](https://aemo.com.au)). First-time
NEMOSIS capacity data collection may take significant time.

​```bash
pip install -r requirements.txt
python main.py
python -m streamlit run src/app.py
​```

## Data sources

- [AEMO Price and Demand data](https://aemo.com.au)
- [Open-Meteo Historical Weather API](https://open-meteo.com)
- [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS) for AEMO MMSDM tables (generation capacity)

## Disclaimer

The hedging cost simulation in the dashboard is a simplified
proof-of-concept. It assumes avoided demand has no replacement cost and
does not constitute financial or trading advice.
