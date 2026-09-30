# AEMO Electricity Price Forecasting

A daily forecasting system for NSW electricity prices (RRP) and demand in
Australia's National Electricity Market (NEM). Every morning it forecasts
**today and tomorrow**, keeps a record of how past forecasts turned out, and
flags when the market has stopped behaving like the data the models learned
from. Runs end-to-end on GitHub Actions and a Streamlit dashboard.

**[Live Dashboard](https://aemo-electricity-forecasting-xhtf7sj4xqjesgknueoqza.streamlit.app/)**

## What this project does

NEM prices can swing from ~$50/MWh to over $20,000/MWh within minutes when
supply is tight, and can fall below zero when renewables flood the market.
A single "expected price" hides that risk, so every forecast has two parts:

1. **Central forecast**: the most likely price, for planning
2. **Risk ceiling**: a 90th-percentile upper bound that actual prices should
   stay below about 90% of the time

Every morning (market time), GitHub Actions downloads the latest AEMO data,
retrains the models on the most recent 365 days, issues a forecast for today
(6-24h ahead) and tomorrow (24-48h ahead), and commits it to the repo. The
dashboard redeploys automatically.

## Dashboard

| Tab | What it answers |
|---|---|
| 🔮 **Outlook** | What will prices look like today and tomorrow, and when is the risk highest? |
| 🎯 **Track record** | How did the forecasts saved before each day compare with what actually happened? |
| 🌡️ **Market regime** | Is the market calmer or spikier than the models' training period? Compare any two periods side by side |
| 💰 **Risk strategy backtest** | What would acting on the risk ceiling have saved, and how many warnings were real spikes? |

## Results

Walk-forward backtest, January 2025 to August 2026 (20 months). For each
month, the models were trained only on the 365 days before it, then forecast
every day of that month. "Normal" means prices below $300/MWh. The naive
benchmark repeats the price at the same time on the latest day known at issue.

| Horizon | Normal MAE | Naive MAE | Normal R² | Months beating naive | Risk ceiling coverage (target 90%) |
|---|---|---|---|---|---|
| Today (6-24h ahead) | **$21.55** | $42.70 | 0.674 | 20 / 20 | 88.5% |
| Tomorrow (24-48h ahead) | **$30.86** | $51.44 | 0.449 | 20 / 20 | 89.4% |

Demand forecast R²: 0.869 (today), 0.853 (tomorrow).

**Against AEMO's own outlook** (today's forecast vs AEMO's PD7DAY pre-dispatch
price published before 06:00, compared per half hour): typical error $13.4 vs
$15.6/MWh (better in 20 / 20 months), and the risk ceiling flagged 62% of
spikes with 21% of flags being real, vs 61% and 17% for AEMO's outlook.

What the models **cannot** do: predict how large a spike will be one or two
days out. Spike-period errors (~$1,900/MWh) are close to the naive benchmark.
That is why the risk ceiling exists as a separate output rather than trying
to fold spikes into the central forecast.

## Things I found along the way

**1. My first model's headline numbers relied on information a real forecast
wouldn't have.** The v1 price model (normal R² 0.57) used same-interval
actual demand, available generation, reserve margin and prices up to 5
minutes before. Its three most important features were all of this kind.
Rebuilding it with only what is known when the forecast is issued gave a
lower but honest baseline, which the results above are based on.

**2. A walk-forward backtest exposed a systematic bias.** The central model
(L2 loss on an arcsinh-transformed price) under-predicted in **every one of
20 months**, by about $16/MWh on average. In arcsinh space, negative prices
sit very far from normal ones, so they drag the fitted mean down. Switching
to an L1 (median) objective on arcsinh(price / 100) removed the bias
(-$16.4 to -$0.2) and cut normal MAE by 12%. A single test month had not
revealed this.

**3. The market changed under the models.** Spikes (≥ $300/MWh) fell from
40-389 per month in 2025 to 0-15 per month from March 2026, and negative
prices became rare. Public reporting links this to growing battery storage
([issue #5](https://github.com/ichirin0311/aemo-electricity-forecasting/issues/5)).
A model trained on the past year can quietly go stale, so the dashboard
compares recent spike rates with the training window and tracks risk ceiling
coverage month by month.

**4. The "blocked" AEMO download was a User-Agent problem.** Automated
downloads had failed with HTTP 403, which looked like AEMO blocking scripts.
The actual cause was Cloudflare rejecting Python `urllib`'s default
User-Agent. With `requests`, the monthly CSVs download fine, including from
GitHub Actions runners.

**5. AEMO's own outlook is a strong input, but only up to the bid deadline.**
On its own, AEMO's 7-day pre-dispatch price was closer than my model on
typical half-hours, yet its average error was four times worse because it
occasionally projects very high prices that never happen. For days whose
generator bids aren't in yet (due around 12:30 the day before), it sits at
the market price cap in ~10% of half-hours, so it is only usable for today's
forecast. Feeding it into today's model as an input, and letting the model
learn when to trust it, cut normal MAE by 23% (better in 20 / 20 months) and
made the risk ceiling catch more spikes with fewer false alarms.

## How it works

```
AEMO monthly price/demand CSVs ──┐
Open-Meteo temperature ──────────┼──> features known at issue time ──> LightGBM (per horizon) ──> data/forecasts/
AEMO PD7DAY price outlook ───────┘    (outlook: today's price only)     demand / central / Q90

GitHub Actions (daily): download → retrain → forecast → commit → Streamlit Cloud redeploys
```

## Key modeling decisions

- **Only information available at issue time**: prices and demand up to the
  previous midnight, the same time on the latest day and a week earlier,
  calendar features, a temperature forecast, and (for today) AEMO's price
  outlook as published before 06:00. AEMO's CSV is refreshed at 00:00 market
  time, so a morning forecast always has yesterday complete.
- **Separate central and risk-ceiling models** rather than one blended
  prediction. Four blending/routing approaches were tried in v1 (classifier
  routing, dollar-scale and arcsinh-space blending, self-routing on the
  ceiling). All underperformed, because the spike classifier's precision was
  only ~18%.
- **arcsinh, not log**, for the price target: the -$1,000/MWh price floor
  forces a large log shift that flattens normal-range variation. The central
  model uses a scaled arcsinh with a median objective (see finding 2); the
  risk ceiling is LightGBM quantile regression at α = 0.9.
- **Rolling 365-day window, retrained daily**, with separate models for each
  horizon.
- **AEMO market time is fixed UTC+10** (no daylight saving), so weather data
  is converted with `Etc/GMT-10`, not `Australia/Sydney`.

## Project structure

```
run_day_ahead.py           daily forecast (today + tomorrow); run by GitHub Actions
run_backtest.py            walk-forward evaluation (monthly retrain)
src/day_ahead.py           data assembly, features, training, forecasting
src/aemo_downloader.py     AEMO monthly CSV download with caching and retries
src/aemo_pd7day.py         AEMO PD7DAY outlook: archive + live download, as-of-issue selection
src/dashboard_data.py      data loading and analysis for the dashboard
src/app.py                 Streamlit dashboard
experiments/               feasibility and target-transform experiments
.github/workflows/         daily pipeline
data/forecasts/            latest forecast + forecast log (updated daily)
data/backtest/             walk-forward results

main.py, src/pipeline.py,  v1: fixed-2025 pipeline, still run daily as a
src/train.py               regression test of the data sources (see below)
```

## Setup

```bash
pip install -r requirements.txt
```

Open the dashboard (uses the forecasts and backtest results in the repo, no downloads):

```bash
python -m streamlit run src/app.py
```

Issue a fresh forecast for today and tomorrow (downloads AEMO and Open-Meteo data):

```bash
python run_day_ahead.py
```

Backtest as if the forecast had been issued on a past morning (prints only):

```bash
python run_day_ahead.py 2025-12-01
```

Re-run the 20-month walk-forward evaluation (about 5 minutes):

```bash
python run_backtest.py
```

## v1 (archived results)

The first version trained on January-October 2025 and tested on December
2025, using supply-side features from AEMO's DISPATCHREGIONSUM table via
[NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS). It still runs daily
(`main.py`) as a check that the data sources work.

| Metric | Value |
|---|---|
| Demand forecast R² | 0.9665 |
| Price forecast R² (normal conditions, < $300/MWh) | 0.5667 |
| Price forecast MAE (normal conditions) | $19.22 |
| Risk ceiling MAE (spike conditions) | $1,790.42 |

These numbers use same-interval actuals as inputs (see finding 1), so they
are not comparable with the v2 forecast results above.

## Data sources

- [AEMO aggregated price and demand data](https://www.aemo.com.au) (5-minute, NSW1)
- AEMO PD7DAY pre-dispatch price outlook ([NEMweb](https://www.nemweb.com.au) MMSDM archive and current reports)
- [Open-Meteo](https://open-meteo.com): Historical Weather API and Forecast API
- [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS) for AEMO MMSDM tables (v1 only)

## Disclaimer

The risk strategy backtest is a simplified proof of concept. It assumes
flagged demand could be curtailed at no cost and does not constitute
financial or trading advice.
