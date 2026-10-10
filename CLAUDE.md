# RRP (Electricity Price) Spike Forecasting — Accuracy Improvement Notes

Target: AEMO NEM (NSW1) electricity price forecasting pipeline
Task: RRP regression forecasting via LightGBM (Task B)
Period: Train = Jan-Oct, Validation = Nov, Test = Dec (2025 data)

## Background / Problem

Demand forecasting (Task A) performed well with R2=0.9665, but price forecasting
(Task B) was extremely low at R2=0.0085. Several approaches were tried to isolate
the cause.

RRP has an extremely skewed distribution, which turned out to be the root cause
of the poor accuracy.
- Normal conditions: roughly $0-150/MWh
- Spike conditions: up to $20,300/MWh (within the training data)
- Price floor: -$1000/MWh (negative values occur)

## What Was Tried (in chronological order)

### 1. Adding a supply-side feature (Capacity)
- Fetched `availablegeneration` (available generation capacity) from AEMO's
  DISPATCHREGIONSUM table via NEMOSIS
- `reserve_margin = availablegeneration - totaldemand` (spare capacity)
- `reserve_margin_ratio = reserve_margin / totaldemand` (spare capacity ratio)
- -> Confirmed to rank near the top in feature importance and to be useful for
  price forecasting

### 2. Target transform: log -> asinh
- Initially used `log(rrp + shift_val)` for the log transform, but correcting
  for the price floor (-1000) required a large constant shift (about +1000),
  which nearly flattened normal-condition price variation in log space
  (R2 in the transformed scale was a poor 0.10)
- Switching to an `arcsinh transform` (`log(x + sqrt(x^2+1))`) substantially
  improved R2 in the transformed scale from 0.10 to **0.7662**
  - Normal-conditions R2: -3.46 -> **0.2519**
  - Normal-conditions MAE: 40.29 -> **22.30**
  - Spike-conditions MAE: 2004.61

### 3. Tried a two-stage approach (classification + segment-specific regression) -> Rejected
Tried classifying "spike or not" and then training/combining separate regression
models for normal vs. spike conditions, but **it could not beat the single
asinh model and was rejected**.

| Approach | Normal MAE | Normal R2 | Spike MAE |
|---|---|---|---|
| Single asinh model (adopted) | **22.30** | **0.2519** | 2004.61 |
| Two-stage (hard routing) | 35.48 | -2.7873 | 1837.66 |
| Two-stage (soft blend in dollar scale) | 35.49 | -1.9322 | 1731.88 |
| Two-stage (soft blend in asinh space) | 30.28 | -1.1148 | 1821.23 |

Reason for rejection: the classification model had a strong AUC of 0.9748, but
Precision was only 0.1815 (82% of cases predicted as spikes were false
positives), so a large amount of normal-condition data was incorrectly routed
to the spike side, degrading normal-condition accuracy. Three blending methods
were tried (hard -> dollar blend -> asinh-space blend), and while each showed
some improvement, none ever matched the accuracy of the single model.

### 4. Additional features (for classification / quantile use)
Added the following to `generate_features` in `pipeline.py` and used them in
subsequent experiments:
- `reserve_margin_slope_1h`: rate of change (slope) of reserve_margin over the
  last hour
- `rrp_volatility_1h`: rolling std dev of RRP over the last hour (computed
  after shift(1), excluding the current point)

### 5. Standalone evaluation of quantile regression (objective='quantile')
Tried a model that directly predicts the "90th percentile value (an estimate
of upside risk)" without going through a classification-based switch.
Comparison of alpha values:

| alpha | Spike MAE |
|---|---|
| 0.9 | **1790.42** (best) |
| 0.95 | 1971.04 |
| 0.99 | 5307.62 (much worse — an extreme quantile sticks to high values and
  misses badly during normal-to-mild-spike conditions) |

**With alpha=0.9, the standalone Q90 model improved spike-conditions MAE from
2004.61 (single asinh model) to 1790.42 (about a 10% reduction).** This was the
only result across the whole series of experiments that partially beat the
single model.

### 6. Integration via self-routing from the Q90 model -> Failed
Rather than routing through the low-precision classification model, tried a
self-judged routing scheme: "if the Q90 model's prediction exceeds $300, use
the Q90 prediction; otherwise use the single asinh model's prediction."

| Metric | Single asinh model | Q90 standalone | Integrated (Q90 routing) |
|---|---|---|---|
| Normal MAE | 22.30 | 22.30 (shared) | 25.56 (worse) |
| Normal R2 | 0.2519 | - | -0.9216 (worse) |
| Spike MAE | 2004.61 | 1790.42 | 1850.01 (worse than Q90 standalone) |

**Result: integrating the two made both normal- and spike-condition accuracy
worse than the standalone Q90 model.** The cause mirrors the earlier
classification-based two-stage approach: because the Q90 model is
"conservatively biased toward the upside" by nature, it sometimes predicts
above $300 even for genuinely normal price cases ($50-150), and each such
false alarm discards the more accurate single model's prediction. Trying to
use the Q90 model itself as the router — to sidestep the classification
model's precision problem — just reproduced the same false-positive pattern.

## Current Conclusion

Four integration methods have now been tried (hard classification routing,
dollar-scale blending, asinh-space blending, and Q90 self-routing), and **all
of them underperformed the single asinh model's normal-conditions accuracy**.
This does not look like coincidence — it suggests that "clearly determining
the boundary between normal and spike conditions" is inherently difficult with
the current data and features.

Rather than forcing everything into a single combined prediction, the proposal
is to **run two independent models in parallel**:

- **Central prediction model** (single asinh model): "the most likely price"
  -> Normal-conditions R2=0.2519, normal-conditions MAE=22.30
- **Risk ceiling model** (Q90 quantile model, alpha=0.9): "the price that
  actuals should stay below 90% of the time (a tail-risk estimate)"
  -> Spike-conditions MAE=1790.42 (when used standalone)

For a heavy-tailed distribution like electricity prices, presenting an
"expected value" separately from a "tail risk" estimate is likely more
practically useful than a single point prediction.

## Not Yet Started / Future Candidates

1. **Establish an implementation and evaluation approach for running the two
   models in parallel**
   - Design a dashboard-style layout that shows the "central prediction" and
     the "Q90 risk ceiling" side by side
   - Move from a single combined score to monitoring accuracy metrics for each
     of the two axes separately
2. **Further improve normal-conditions R2 (currently 0.25)**
   - Tune `num_leaves`, `min_data_in_leaf`
   - Further develop capacity-related features (e.g. reserve margin
     acceleration)
3. **Further reduce spike-conditions MAE (currently above 1790)**
   - Spikes themselves are frequently extreme outliers, so it's worth
     reconsidering whether MAE is even the right metric
   - Fine-tuning around alpha=0.9 (e.g. 0.92, 0.93) is unlikely to yield large
     gains, so it's a lower priority
4. **Extend the training period**
   - Currently limited to one year (2025). Multiple years of data could make
     learning of seasonality and spike patterns more stable
5. **Classification-based integration is on hold for now**
   - Improving precision (adjusting scale_pos_weight, raising the threshold)
     is theoretically worth trying, but since the Q90 self-routing approach
     reproduced the same kind of problem, this is deprioritized for now

## Reply Comments on Gemini's Roadmap v2

The direction of the "two models in parallel" roadmap Gemini proposed is
sound. The phasing (features/parameters -> multi-year data expansion ->
deployment) is also a natural sequence. However, the following three points
should be checked/considered before starting.

### 1. Some candidate new features may overlap with existing features

`totaldemand_ratio_to_generation` (ratio of demand to available generation
capacity) expresses essentially the same information, in a different form, as
the existing `reserve_margin_ratio` (= (availablegeneration - totaldemand) /
totaldemand). LightGBM is reasonably robust to nonlinear transforms of this
kind, so the practical harm is likely small, but it should be adopted with the
understanding that it's "rephrasing existing information" rather than "adding
new information." It also risks making feature-importance interpretation more
confusing.

`reserve_margin_acceleration_1h` (rate of change of the slope) would require
an additional 2 hours of lag information on top of `reserve_margin_slope_1h`,
which already involves a shift(12) computation. Note that this will increase
the number of NaNs near the start of the data (i.e., more rows dropped by
dropna).

### 2. Extending to multiple years of data is harder than the roadmap suggests

Looking back at experience so far, even a single year (2025) required repeated
troubleshooting for both `fetch_aemo_data` (manually placed CSVs) and
`fetch_capacity_data` (via NEMOSIS) — column-name casing, date formats, NEMWeb
403 errors, and so on. Extending this to three years (2023-2025) would mean:
- Needing to assemble 36 months of AEMO PRICE_AND_DEMAND CSVs (manual
  downloads)
- Tripling the volume of DISPATCHREGIONSUM data pulled via NEMOSIS, making the
  initial download considerably heavier
- Possibly being affected, even for historical data, by the April 2026 NEMWeb
  URL migration

This is described only lightly as "Phase 2," but it's likely to actually be
the most time-consuming step. Before committing to it, it's recommended to
first verify at small scale that 2023-2024 AEMO CSVs can be fetched without
issue in the same URL format.

> **Update (2026-09):** AEMO CSV ingestion is now automated (see "AEMO CSV
> Ingestion Automated" below), and 2023 CSVs were confirmed to be downloadable
> from the same URL format, so the manual-download concern above no longer
> applies. The NEMOSIS volume concern still does.

### 3. Be careful about the sample size behind the Phase 1 evaluation

Phase 1 plans to measure the effect of parameter changes
(`num_leaves=15, min_data_in_leaf=100`, etc.) using only 2025 data, but in the
most recent experiment the test period (December) contained only 62 spike
cases. With a sample that small, it's hard to tell whether an
improvement/regression from a parameter change is a "real improvement" or just
"random noise." It's recommended not to finalize conclusions from Phase 1
alone, and to plan on re-validating after Phase 2 (multi-year data expansion).

## Focused Implementation and Results (Finalized Dual-Axis Operation)

After discussion with Gemini, implementation was narrowed down to the
following three directions.

1. **Don't overreach on features**: adopted only the two finalized features,
   `reserve_margin_slope_1h` (reserve margin slope) and `rrp_volatility_1h`
   (price volatility). Held off on candidate new features that risked
   overlap (`totaldemand_ratio_to_generation`,
   `reserve_margin_acceleration_1h`).
2. **Tune the central prediction model specifically for normal conditions**:
   set `num_leaves=15`, `min_data_in_leaf=100`, `learning_rate=0.03` so it
   wouldn't get pulled around by spike noise.
3. **Implement dual-axis output**: train and run inference for the central
   prediction model (single asinh) and the risk ceiling model (Q90 quantile
   regression) at the same time, and changed the output to a CSV with two
   columns, `rrp_base_prediction` (central prediction) and
   `rrp_risk_ceiling` (risk ceiling)
   (`data/processed/rrp_dual_prediction_output.csv`).

### Results

| Metric | Before (generic parameters) | After (normal-conditions-specific tuning) |
|---|---|---|
| Normal MAE | 22.30 | **19.22** (improved) |
| Normal R2 | 0.2519 | **0.5667** (substantially improved) |
| Spike MAE (central prediction model) | 2004.61 | 2023.85 (roughly the same, slightly worse) |
| Spike MAE (Q90 risk ceiling model) | 1790.42 | 1790.42 (unchanged) |

**Normal-conditions R2 improved substantially, from 0.25 to 0.57.** The
strategy of "tuning the central prediction model specifically for normal
conditions so it isn't thrown off by spikes" clearly paid off. Spike-
conditions MAE got slightly worse, but that was expected (the central
prediction model was never intended to capture spikes in the first place —
that role is by design left to the risk ceiling model).

Feature importance also confirmed things are working as intended:
`reserve_margin_ratio` ranked most important, followed by `rrp_lag_1h_t` and
`rrp_volatility_1h`, with the newly adopted finalized features ranking near
the top.

### New Issue: the Q90 Risk Ceiling Model's Coverage Rate Falls Short of Target

In theory, the Q90 model should have "90% of actuals fall below this
prediction," but the observed coverage rate was only **83.51%** (about 6.5pt
short of the 90% target). In other words, in more cases than expected (about
17%), the actual value exceeded the Q90 prediction — meaning it's somewhat
optimistic (prone to erring on the dangerous side) for use as a "risk
ceiling."

**Possible causes**:
- The test period (December) falls in a season with more spikes, so upside
  surprises may be happening "more often than expected" relative to the
  distribution seen in the training period (Jan-Oct) — i.e., a seasonal
  mismatch
- With only 62 spikes in the test period, the estimate of the 90th percentile
  from quantile regression is itself prone to being statistically unstable

**A trade-off to keep in mind**: raising alpha from 0.9 to 0.95 might improve
coverage, but in the earlier experiment (alpha comparison, see the table in
section 5), alpha=0.95 made spike-conditions MAE worse, up to 1971.04.
"Getting coverage closer to 90%" and "minimizing spike-conditions MAE" are in
tension with each other, and the right call depends on the use case (whether
it needs to function as a strict risk ceiling, or should prioritize staying
close to actual prices).

## AEMO CSV Ingestion Automated (2026-09)

`fetch_aemo_data` no longer relies on manually placed CSVs. `src/aemo_downloader.py`
downloads the monthly `PRICE_AND_DEMAND_{YYYYMM}_{REGION}.csv` files into
`data/raw/aemo_data_1year/`, and the GitHub Actions workflow commits any new CSVs
back to the repo.

Findings from the investigation:
- The URL `https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{YYYYMM}_NSW1.csv`
  still works and returns files byte-identical to the manually downloaded ones.
  The bare `aemo.com.au` host 301-redirects to `www`.
- **The earlier HTTP 403s were caused by Python's `urllib` default User-Agent
  (`Python-urllib`)**, which Cloudflare blocks. `requests` (and browser-like UAs)
  get 200. Confirmed working from GitHub Actions runners too.
- The current month is available as a partial file (up to the latest day);
  future months return 404.

Downloader behavior: closed months that already exist locally are not re-downloaded;
the current and previous month are always refreshed; on download failure an existing
local copy is used so a transient outage doesn't break the daily run.

The NEMOSIS cache (`data/raw/nemosis_cache/`, gitignored) is persisted between
Actions runs via `actions/cache`, and only the feather copies are kept
(`keep_csv=False`, ~120MB instead of ~388MB for 14 months).

## v2: Rolling Day-Ahead Forecast (2026-09)

v1 (`main.py`, `src/train.py`) stays as is. v2 runs alongside it:
`src/day_ahead.py` + `run_day_ahead.py` (daily in Actions, output `data/forecasts/`),
`run_backtest.py` (walk-forward evaluation).

### Why v1's metrics don't carry over
The v1 price model uses **same-interval actuals** (`totaldemand`,
`availablegeneration`, `reserve_margin*`, actual temperature) and lags as short
as 5 minutes (`rrp_volatility_1h`). None of these are known when a forecast is
issued in advance, so v1's normal R2=0.5667 is closer to a nowcast. Its top
3 features by gain are all of this kind.

### Forecast definition
- Issued each morning. The nominal issue time is 06:00 market time (it
  defines which AEMO PD7DAY run counts as published); Actions runs at UTC
  16:00 = 02:00 market time because scheduled runs are often hours late. AEMO's
  PRICE_AND_DEMAND CSV is refreshed at 00:00 market time, so actuals are
  complete up to D 00:00.
- Two leads, each with its own demand / central / Q90 models:
  `today` (day D, 6-24h ahead) and `tomorrow` (day D+1, 24-48h ahead).
- Features use only actuals up to the end of day T - lead: same time-of-day
  on the latest complete day and 7 days earlier, daily price/demand stats of
  the latest complete day (+7-day means), calendar, temperature (forecast API
  for future hours; archive actuals in backtests).
- **No demand/capacity inputs for the target interval** (deliberate decision).
  AEMO PREDISPATCH forecasts are the planned next addition. As a side effect
  v2 does not depend on NEMOSIS.
- Rolling window: trained on the 365 days before the cutoff, last 28 days for
  early stopping / monitoring.

### Walk-forward evaluation (2025-01..2026-08, monthly retrain)
Central model after the fix below, pooled over 20 months:

| Lead | Normal MAE | Normal R2 | Naive normal MAE | Months beating naive | Q90 coverage |
|---|---|---|---|---|---|
| today | 27.92 | 0.531 | 42.70 | 20/20 | 87.8% |
| tomorrow | 30.86 | 0.449 | 51.44 | 20/20 | 89.4% |

(Naive = same time on the latest complete day.) Spike magnitude is not
predictable at these horizons: spike MAE (~1,900) is close to naive (~2,050-2,150).
Q90 coverage is close to 90% pooled but ranges 76-97% by month.

### Fix: asinh + L2 under-predicted systematically
With the v1 setup (L2 on asinh(rrp)) the central model under-predicted in
**every** month (pooled bias about -16 $/MWh, worst -36 in Jun 2026). In asinh
space +$80 and -$30 are far apart (5.1 vs -4.1), so negative prices (21-30% of
intervals in late 2025) pull the mean down hard, and 2026 had almost none, so
the gap widened. Compared in `experiments/price_target_bias.py`:
- `asinh(x/30)` / `asinh(x/100)` with L2, and L1 on `asinh(x)`, all helped
- **Adopted: central = L1 (median) on asinh(rrp/100)**. Median survives the
  inverse transform unbiased. Normal MAE 31.87 -> 27.92 (today), bias -16.4 -> -0.2.
- The Q90 model keeps asinh(rrp): scaling its target worsened spike MAE
  (1917 -> 2039).

Note this does not contradict the earlier "asinh, not log" decision; it is
still asinh, just with a scale and a median objective.

### AEMO PD7DAY benchmark (2026-10)
`src/aemo_pd7day.py`, `experiments/pd7day_benchmark.py`. NEMOSIS does not
support pre-dispatch tables, so this is a custom loader.
- **Which AEMO forecast to use**: PREDISPATCH is unusable here. The MMSDM
  archive keeps only the latest run per interval, and a 06:00 run only
  reaches 04:00 the next day. PD7DAY runs 3x/day (RUN_DATETIME 07:30 /
  13:00 / 18:00, published ~17 min earlier = LASTCHANGED) and the MMSDM
  archive keeps every run. Files up to 2026-07 are cumulative (all runs since
  2024-04); later files are monthly. The loader walks back until covered.
- **As-of rule**: the latest run with LASTCHANGED <= issue day 06:00, i.e.
  the D-1 18:00 run.
- **Only valid for the "today" lead.** Before the day-ahead bid deadline
  (~12:30), PD7DAY prices for the following days sit at the market price
  cap in ~10% of half-hours (07:30 run: 10.4% one day ahead; 13:00 run:
  1.0%). Using PD7DAY for "tomorrow" would require issuing after ~13:00.
- **Result (today, 30-min, 20 months)**: AEMO is better on typical intervals
  (median abs. error $15.6 vs ours $19.4) and flags more spikes (61% vs 52%,
  both ~17% precision), but occasional high false alarms give it normal MAE
  $106.7 (bias +$81) vs ours $26.2. A plain average when AEMO < $300 gives
  $20.8, so the two carry complementary information.
- Live: `run_day_ahead.py` logs AEMO's today price to
  `data/forecasts/aemo_pd7day_log.csv` (non-fatal if NEMweb fails); the
  dashboard shows it on Outlook and Track record.

### PD7DAY as a model input: model v2.2 (2026-10)
`experiments/pd7day_as_input.py`. The "today" central and Q90 models get
`aemo_rrp` (the 30-min outlook for the interval) plus the day's AEMO mean,
max and share >= $300. Not used for "tomorrow" (bid deadline, see above) or
for demand.
- `aemo_pd7day_log.csv` doubles as training data: it was backfilled with the
  as-of-06:00 outlook for every day since 2024-04-24 (MMSDM archive + NEMweb
  Current), and the daily run appends one day. So Actions never downloads
  the large archive. AEMO features may be NaN (early training rows, or a
  day NEMweb failed); LightGBM handles that and they are excluded from dropna.
- Walk-forward, today lead: normal MAE 27.92 -> **21.55** (-23%), R2
  0.531 -> 0.674, better in **20/20 months**. Q90: spike MAE 1917 -> 1820,
  spikes flagged 51% -> 61%, coverage 87.8% -> 88.5%.
- Live forecasts now carry `model_version` (v2.0 L2/ran low, v2.1 L1 fix,
  v2.2 + AEMO; `v2.2-noaemo` if the outlook was unavailable that morning).
  The dashboard's track record splits by it.

### Training window per lead: model v2.3 (2026-10, issue #5)
`experiments/training_window.py`. Question: spikes became far rarer in
2026 even at the same demand level, so should old regimes be dropped or
down-weighted? Compared 365 / 180 / 730 days and 365 days with recency
weights (half-life 180 / 90 days on the price models), judged pooled, by
year, and on high-demand intervals (actual >= 10 GW, 867 spike intervals).
- **180 days is unsafe**: worse normal MAE and far weaker on high-demand
  days (tomorrow: high-demand spikes flagged 62% -> 23%, none in 2026).
- **Recency weighting**: no consistent gain (beat 365d in 4-9/20 months).
- **730 days**: today, same normal MAE but every spike metric better
  (flagged 60% -> 65%, high-demand 68% -> 75%, coverage 88.3% -> 89.3%);
  tomorrow, normal MAE slightly better but spike flags weaker (48% -> 42%).
- **Adopted (v2.3)**: `TRAIN_DAYS = {"today": 730, "tomorrow": 365}` in
  `src/day_ahead.py`. Older experiments pin `train_days=365` so their
  recorded numbers still reproduce. Rows before 2024-04-24 have no AEMO
  features (NaN), which LightGBM handles.
- Takeaway for #5: don't drop old regimes. Calm months dominate recent
  data, but extreme-demand days still need the spiky history.

### Operations notes
- **Missed days are not backfilled.** The live record's value is that each
  forecast was saved before its day. Gaps so far: 2026-10-03 (the cron change
  skipped that day's run) and 2026-10-09 (below).
- **2026-10-09 failure**: Open-Meteo's archive API hung during the TLS
  handshake; with no timeout set, the run waited ~18 minutes and failed.
  Open-Meteo calls now use a 60 s timeout and 4 retries with backoff
  (`_openmeteo_client` in `src/day_ahead.py`), so an outage fails in ~5-6
  minutes. Temperature is still required: if Open-Meteo is down for longer,
  that day's forecast is skipped (AEMO calls already fall back gracefully).

### AI briefing on the dashboard (2026-10-10)
Motivated by a job requirement on integrating LLMs into data pipelines
programmatically. `src/ai_summary.py`, called at the end of the live
`run_day_ahead.py` run; output `data/forecasts/ai_summary_log.jsonl`, shown
at the top of the Outlook tab.
- One Claude API call per day (`claude-opus-5-5`, effort low, JSON-schema
  output, `fallbacks: "default"`). Input is only a dict of rounded facts
  built in code (`build_facts`); risk level is a fixed rule (`risk_level`).
- `unsupported_numbers` rejects any draft containing a number or HH:MM time
  not in the facts (within rounding); the rejected draft stays in the log.
- No key / API error / rejection -> deterministic `template_summary`,
  `source: "template"`. Never fails the run (same policy as PD7DAY).
- Requires the `ANTHROPIC_API_KEY` repository secret. Like forecasts,
  briefings are not backfilled. Bump `PROMPT_VERSION` when the prompt changes.

## Not Yet Started / Future Candidates (Updated)

Done since v2 started: dashboard switched to v2 (`src/app.py` +
`src/dashboard_data.py`: Outlook, Track record, Market regime, Risk strategy
backtest), README rewritten (v1 numbers labelled as using same-interval
actuals), PD7DAY benchmark, PD7DAY as input (v2.2), training-window
decision for issue #5 (v2.3).

- Step 3: STPASA (forecast demand / available capacity, ~220MB/month
  archive) as inputs, which would bring back a forecast reserve margin
- Using PD7DAY for "tomorrow" would require issuing after ~13:00
- Scenario comparison (issue #4) only if a clear use case appears


以下 For Secretary

## 進捗管理のルール

このプロジェクトの進捗は、共有の台帳で管理する。

- 台帳の場所: "G:\My Drive\secretary\tasks.md"
- このプロジェクトのセクション名: 「aemo-electricity-forecasting」

### 作業の前に
- 台帳を読み、このプロジェクトのセクションにある未完了タスクと「次の一手」を確認する
- 作業内容がタスクとずれているときは、先にそのことを伝える

### 作業の後に
- 完了したタスクにチェックを付ける（`- [x]`）
- 「次の一手」を最新の状態に更新する
- 新しく発生したタスクは、期限付きで追記する（期限が不明なら「期限: 未定」）
- 先頭の「最終更新」の日付を今日に更新する

### 守ること
- 編集してよいのは、このプロジェクトのセクションだけ。他のセクションは変更しない
- 既存のタスクを勝手に削除しない（不要なら、削除してよいか確認する）
- 台帳にアクセスできない場合は、その旨を伝えて、台帳なしで作業を続ける


 
