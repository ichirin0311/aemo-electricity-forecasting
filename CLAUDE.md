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

## Not Yet Started / Future Candidates (Updated)
