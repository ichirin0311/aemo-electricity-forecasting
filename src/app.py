# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="AEMO NSW1 Electricity Price Risk Management Dashboard", layout="wide")


@st.cache_data
def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["settlementdate"])
    return df


def compute_kpis(df: pd.DataFrame, spike_threshold: float) -> dict:
    normal_mask = df["rrp_actual"] < spike_threshold
    spike_mask = df["rrp_actual"] >= spike_threshold

    kpis = {
        "normal_mae": mean_absolute_error(df.loc[normal_mask, "rrp_actual"], df.loc[normal_mask, "rrp_base_prediction"]) if normal_mask.sum() > 0 else np.nan,
        "normal_r2": r2_score(df.loc[normal_mask, "rrp_actual"], df.loc[normal_mask, "rrp_base_prediction"]) if normal_mask.sum() > 0 else np.nan,
        "spike_mae_base": mean_absolute_error(df.loc[spike_mask, "rrp_actual"], df.loc[spike_mask, "rrp_base_prediction"]) if spike_mask.sum() > 0 else np.nan,
        "spike_mae_q90": mean_absolute_error(df.loc[spike_mask, "rrp_actual"], df.loc[spike_mask, "rrp_risk_ceiling"]) if spike_mask.sum() > 0 else np.nan,
        "coverage": (df["rrp_actual"] <= df["rrp_risk_ceiling"]).mean(),
        "n_normal": int(normal_mask.sum()),
        "n_spike": int(spike_mask.sum()),
    }
    return kpis


def simulate_hedging_strategy(df: pd.DataFrame, risk_threshold: float, hedge_ratio: float) -> pd.DataFrame:
    df = df.copy()
    INTERVAL_HOURS = 5 / 60  # 5分足なので、MW→MWh変換のため時間換算

    df["cost_naive"] = df["rrp_actual"] * df["totaldemand"] * INTERVAL_HOURS

    is_high_risk = df["rrp_risk_ceiling"] >= risk_threshold
    hedged_demand = np.where(is_high_risk, df["totaldemand"] * (1 - hedge_ratio), df["totaldemand"])
    df["cost_hedged"] = df["rrp_actual"] * hedged_demand * INTERVAL_HOURS

    df["cumulative_cost_naive"] = df["cost_naive"].cumsum()
    df["cumulative_cost_hedged"] = df["cost_hedged"].cumsum()
    df["is_high_risk"] = is_high_risk

    return df


# ==================== Sidebar (business levers only) ====================
st.sidebar.header("⚙️ Scenario Settings")
st.sidebar.caption("Adjust how you prepare for price spike risk using the two settings below.")

df_full = load_predictions("data/processed/rrp_dual_prediction_output.csv")

min_date = df_full["settlementdate"].min().date()
max_date = df_full["settlementdate"].max().date()

date_range = st.sidebar.date_input(
    "Display period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

risk_threshold = st.sidebar.slider(
    "⚠️ Warning line price ($/MWh)",
    min_value=100, max_value=2000, value=300, step=50,
    help="If the predicted price ceiling exceeds this value, it is treated as 'at risk'"
)

hedge_ratio = st.sidebar.slider(
    "Demand reduction rate when avoiding risk",
    min_value=0.0, max_value=0.5, value=0.1, step=0.05,
    help="How much electricity usage is assumed to be curtailed during periods predicted to exceed the warning line"
)

with st.sidebar.expander("🔧 Advanced settings (for experts)"):
    spike_threshold = st.slider(
        "Definition of 'price spike' ($/MWh)", min_value=100, max_value=1000, value=300, step=50,
        help="The threshold value used to separate 'normal' from 'spike' periods when evaluating model accuracy"
    )

# Period filter
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df_full["settlementdate"].dt.date >= start_date) & (df_full["settlementdate"].dt.date <= end_date)
    df = df_full.loc[mask].copy()
else:
    df = df_full.copy()

if df.empty:
    st.warning("There is no data for the selected period. Please adjust the period.")
    st.stop()

# ==================== Main screen ====================
st.title("⚡ AEMO NSW1 Electricity Price Risk Management Dashboard")
st.caption(
    "Predicts the likely near-term price movement and the risk of a potential price spike, "
    "and simulates the effect of a risk avoidance strategy."
)
st.caption(f"Display period: {df['settlementdate'].min().date()} - {df['settlementdate'].max().date()}")

st.markdown("---")

# --- (1) Business value section (most prominent position) ---
st.header("💰 Effect of the risk avoidance strategy")

df_sim = simulate_hedging_strategy(df, risk_threshold, hedge_ratio)

total_naive = df_sim["cumulative_cost_naive"].iloc[-1]
total_hedged = df_sim["cumulative_cost_hedged"].iloc[-1]
savings = total_naive - total_hedged
savings_pct = savings / total_naive * 100 if total_naive != 0 else 0

sim_col1, sim_col2, sim_col3 = st.columns(3)
sim_col1.metric("Total cost if nothing is done", f"${total_naive:,.0f}")
sim_col2.metric("Total cost with the risk avoidance strategy", f"${total_hedged:,.0f}")
sim_col3.metric("Estimated savings", f"${savings:,.0f}", f"{savings_pct:.2f}%")

st.caption(
    "* This simulation is a proof of concept based on the simplified assumption that electricity "
    "usage during periods exceeding the warning line could actually be reduced. It does not account "
    "for the cost of sourcing the curtailed electricity elsewhere."
)

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(
    x=df_sim["settlementdate"], y=df_sim["cumulative_cost_naive"],
    name="Cumulative cost if nothing is done", line=dict(color="firebrick"),
))
fig_sim.add_trace(go.Scatter(
    x=df_sim["settlementdate"], y=df_sim["cumulative_cost_hedged"],
    name="Cumulative cost with risk avoidance strategy", line=dict(color="seagreen"),
))
fig_sim.update_layout(height=400, xaxis_title="Date/time", yaxis_title="Cumulative cost ($)")
st.plotly_chart(fig_sim, use_container_width=True)

with st.expander("🔔 View the periods when risk avoidance was triggered"):
    hedge_events = df_sim[df_sim["is_high_risk"]][
        ["settlementdate", "rrp_actual", "rrp_risk_ceiling", "totaldemand"]
    ].rename(columns={
        "settlementdate": "Date/time",
        "rrp_actual": "Actual price ($)",
        "rrp_risk_ceiling": "Predicted risk ceiling ($)",
        "totaldemand": "Demand (MW)",
    })
    st.dataframe(hedge_events, use_container_width=True)
    st.caption(f"Number of matching records: {len(hedge_events):,}")

st.markdown("---")

# --- (2) Price outlook section ---
st.header("📈 Price outlook")
st.caption("The black line is the actual price, the blue dotted line is the model's prediction, "
           "and the red band is the risk ceiling representing a 'worst case' estimate.")

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=df["settlementdate"], y=df["rrp_risk_ceiling"],
    name="Risk ceiling (assumed maximum)", line=dict(color="rgba(255,99,71,0.5)", width=1),
))
fig_ts.add_trace(go.Scatter(
    x=df["settlementdate"], y=df["rrp_actual"],
    name="Actual price", line=dict(color="black", width=1.5),
))
fig_ts.add_trace(go.Scatter(
    x=df["settlementdate"], y=df["rrp_base_prediction"],
    name="Predicted price", line=dict(color="royalblue", width=1.5, dash="dot"),
))
fig_ts.add_hline(y=risk_threshold, line_dash="dash", line_color="grey",
                  annotation_text=f"Warning line (${risk_threshold})")
fig_ts.update_layout(height=450, xaxis_title="Date/time", yaxis_title="Price ($/MWh)")
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# --- (3) Technical details (supplementary, collapsible) ---
with st.expander("📊 About this model's accuracy (technical details for engineers)"):
    st.markdown(
        "The following are metrics evaluating the predictive accuracy of the machine learning model. "
        "They show how accurately the model predicted a period it had never actually seen (the test data)."
    )

    kpis = compute_kpis(df, spike_threshold)

    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)
    tcol1.metric("Normal-period MAE (mean error)", f"{kpis['normal_mae']:.2f}")
    tcol2.metric("Normal-period R2 (explanatory power)", f"{kpis['normal_r2']:.4f}")
    tcol3.metric("Spike-period MAE (base prediction)", f"{kpis['spike_mae_base']:.2f}")
    tcol4.metric("Spike-period MAE (risk ceiling model)", f"{kpis['spike_mae_q90']:.2f}")
    tcol5.metric("Risk ceiling coverage (target 90%)", f"{kpis['coverage']:.2%}")

    st.caption(
        f"Normal periods: {kpis['n_normal']:,} | Spike periods: {kpis['n_spike']:,} "
        f"(spike defined as price exceeding ${spike_threshold})"
    )

    st.markdown(
        "- **Base prediction model**: LightGBM regression. Tuned to accurately capture price movement during normal periods\n"
        "- **Risk ceiling model**: Quantile regression (90th percentile). Conservatively estimates the 'upside range' during spikes\n"
        "- Supply-side features (such as the reserve margin ratio) function as the most important predictors in the model"
    )