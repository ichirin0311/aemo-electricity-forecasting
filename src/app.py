# src/app.py
# v2 dashboard: day-ahead outlook, live track record, market regime monitoring and a
# walk-forward risk-strategy backtest. Data logic lives in src/dashboard_data.py.
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dashboard_data import (MODEL_LABELS, LEAD_DAYS, alert_quality, benchmark_metrics,  # noqa: E402
                                load_actuals, load_aemo_log, load_backtest, load_backtest_benchmark,
                                load_forecasts, monthly_regime, period_profile, price_metrics,
                                score_forecast_log, simulate_hedging_strategy, spike_drift, to_30min_vs_aemo)

st.set_page_config(page_title="AEMO NSW1 Price Outlook & Risk Dashboard", layout="wide")

# Colors that read on both light and dark themes
C_ACTUAL, C_CENTRAL, C_RISK = "#8c8c8c", "#4c78a8", "#f58518"
C_NAIVE, C_HEDGED, C_A, C_B = "#e45756", "#54a24b", "#9d755d", "#4c78a8"
C_AEMO = "#b279a2"
MARKET_TZ = timezone(timedelta(hours=10))  # AEMO market time: fixed UTC+10, no DST


# ==================== Cached data ====================
@st.cache_data
def cached_actuals():
    return load_actuals()


@st.cache_data
def cached_forecasts():
    return load_forecasts()


@st.cache_data
def cached_backtest():
    return load_backtest()


@st.cache_data
def cached_aemo():
    return load_aemo_log(), load_backtest_benchmark()


@st.cache_data
def cached_regime(spike_threshold):
    return monthly_regime(cached_actuals(), spike_threshold)


actuals = cached_actuals()
latest, log = cached_forecasts()
bt_monthly, bt_preds = cached_backtest()
aemo_log, aemo_bt = cached_aemo()


def aemo_trace(rows: pd.DataFrame, **kw) -> go.Scatter:
    """AEMO's 30-min interval-ending prices drawn as steps over each half hour."""
    rows = rows.sort_values("interval_datetime")
    return go.Scatter(x=rows["interval_datetime"] - pd.Timedelta(minutes=30), y=rows["aemo_rrp"],
                      line=dict(color=C_AEMO, width=1.5, dash="dash", shape="hv"), **kw)


AEMO_NOTE = ("AEMO PD7DAY = AEMO's own 7-day pre-dispatch price outlook, taken from the last run published before "
             "06:00 on the issue day. Shown for today only: generators submit next-day bids around 12:30, so at "
             "06:00 AEMO's outlook for tomorrow is not yet a real price forecast (it sits at the price cap in "
             "about 10% of half-hours).")


# ==================== Sidebar (business levers) ====================
st.sidebar.header("⚙️ Settings")
risk_threshold = st.sidebar.slider(
    "⚠️ Warning line price ($/MWh)", min_value=100, max_value=2000, value=300, step=50,
    help="A period is flagged 'at risk' when the forecast risk ceiling reaches this price",
)
hedge_ratio = st.sidebar.slider(
    "Demand reduction when at risk", min_value=0.0, max_value=0.5, value=0.1, step=0.05,
    help="Share of electricity use assumed to be curtailed in flagged periods (risk strategy backtest)",
)
with st.sidebar.expander("🔧 Advanced settings"):
    spike_threshold = st.slider(
        "Definition of a 'price spike' ($/MWh)", min_value=100, max_value=1000, value=300, step=50,
        help="Separates 'normal' from 'spike' intervals in accuracy metrics and market-regime counts",
    )


# ==================== Header ====================
issue_date = latest["issue_date"].max()
cutoff = issue_date  # actuals are complete up to 00:00 of the issue date
st.title("⚡ AEMO NSW1 Electricity Price Outlook & Risk Dashboard")
st.caption(
    "Every morning a fresh forecast is issued for today and tomorrow: the most likely price "
    "(central forecast) and a risk ceiling that actual prices should stay below about 90% of the time. "
    "Data and models are refreshed daily by GitHub Actions."
)
market_today = pd.Timestamp(datetime.now(MARKET_TZ).date())
if (market_today - issue_date).days > 1:
    st.warning(f"The latest forecast was issued on {issue_date:%d %b %Y}; today's run may not have completed yet.")

tab_outlook, tab_track, tab_regime, tab_strategy = st.tabs(
    ["🔮 Outlook", "🎯 Track record", "🌡️ Market regime", "💰 Risk strategy backtest"]
)


# ==================== (1) Outlook ====================
with tab_outlook:
    st.subheader(f"Forecast issued {issue_date:%a %d %b %Y}")
    st.caption(f"Uses actual prices up to {cutoff:%d %b %H:%M} (market time) plus a temperature forecast.")

    at_risk = latest[latest["rrp_risk_ceiling"] >= risk_threshold]
    peak = latest.loc[latest["rrp_risk_ceiling"].idxmax()]
    by_lead = latest.groupby("lead")["rrp_base_prediction"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today: expected average price", f"${by_lead.get('today', np.nan):,.0f}/MWh")
    c2.metric("Tomorrow: expected average price", f"${by_lead.get('tomorrow', np.nan):,.0f}/MWh")
    c3.metric("Highest risk ceiling (next 2 days)", f"${peak['rrp_risk_ceiling']:,.0f}/MWh",
              f"at {peak['settlementdate']:%a %H:%M}", delta_color="off", delta_arrow="off")
    c4.metric("Time at or above the warning line", f"{len(at_risk) * 5 / 60:.1f} h",
              f"warning line ${risk_threshold}", delta_color="off", delta_arrow="off")

    recent = actuals[actuals["settlementdate"] > cutoff - pd.Timedelta(days=2)]
    fc = latest.sort_values("settlementdate")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["settlementdate"], y=recent["rrp"], name="Actual (last 2 days)",
                             line=dict(color=C_ACTUAL, width=1.5)))
    fig.add_trace(go.Scatter(x=fc["settlementdate"], y=fc["rrp_base_prediction"], name="Central forecast",
                             line=dict(color=C_CENTRAL, width=2)))
    fig.add_trace(go.Scatter(x=fc["settlementdate"], y=fc["rrp_risk_ceiling"], name="Risk ceiling (90%)",
                             line=dict(color=C_RISK, width=1.5), fill="tonexty",
                             fillcolor="rgba(245,133,24,0.15)"))
    aemo_today = aemo_log[aemo_log["issue_date"] == issue_date]
    if not aemo_today.empty:
        fig.add_trace(aemo_trace(aemo_today, name="AEMO outlook"))
    fig.add_hline(y=risk_threshold, line_dash="dash", line_color="grey",
                  annotation_text=f"Warning line (${risk_threshold})", annotation_position="top left")
    fig.add_vline(x=cutoff, line_dash="dot", line_color="grey")
    fig.add_vrect(x0=cutoff + pd.Timedelta(days=1), x1=cutoff + pd.Timedelta(days=2),
                  fillcolor="grey", opacity=0.06, line_width=0,
                  annotation_text="Tomorrow", annotation_position="top left")
    fig.add_annotation(x=cutoff, y=1, yref="paper", text="Forecast from here", showarrow=False,
                       xanchor="left", yanchor="bottom")
    # Short names + normal order so the horizontal legend fits when the chart is exported as an image
    fig.update_layout(height=450, xaxis_title="Date/time (market time)", yaxis_title="Price ($/MWh)",
                      legend=dict(orientation="h", y=-0.2, traceorder="normal"), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    with st.expander("📉 Demand outlook"):
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=recent["settlementdate"], y=recent["totaldemand"], name="Actual demand",
                                   line=dict(color=C_ACTUAL, width=1.5)))
        fig_d.add_trace(go.Scatter(x=fc["settlementdate"], y=fc["demand_forecast"], name="Demand forecast",
                                   line=dict(color=C_CENTRAL, width=2)))
        fig_d.add_vline(x=cutoff, line_dash="dot", line_color="grey")
        fig_d.update_layout(height=320, xaxis_title="Date/time (market time)", yaxis_title="Demand (MW)",
                            legend=dict(orientation="h", y=-0.3), margin=dict(t=20))
        st.plotly_chart(fig_d, width="stretch")

    st.caption(
        "Today's forecast looks 6-24 hours ahead and tomorrow's 24-48 hours ahead, so tomorrow is less certain. "
        "The central forecast targets typical conditions; it does not try to predict the size of price spikes. "
        "That is the risk ceiling's job."
    )
    if not aemo_today.empty:
        st.caption(AEMO_NOTE)


# ==================== (2) Track record ====================
with tab_track:
    st.subheader("How past forecasts turned out")
    st.caption(
        "Each forecast is saved before the day it covers, then compared with what actually happened. "
        "Unlike the backtest, this record cannot be tuned after the fact."
    )

    scored = score_forecast_log(log, actuals)
    lead = st.radio("Forecast horizon", list(LEAD_DAYS), horizontal=True, key="track_lead",
                    format_func=lambda x: {"today": "Today (6-24h ahead)", "tomorrow": "Tomorrow (24-48h ahead)"}[x])
    s = scored[scored["lead"] == lead]

    if s.empty:
        st.info("No forecast for this horizon has been scored yet: actual prices arrive the day after.")
    else:
        rows = []
        for version, g in s.groupby("model_version"):
            m = price_metrics(g, spike_threshold)
            rows.append({
                "Model": MODEL_LABELS.get(version, version),
                "Days scored": g["settlementdate"].sub(pd.Timedelta(minutes=5)).dt.date.nunique(),
                "Normal MAE ($/MWh)": m["normal_mae"],
                "Naive MAE ($/MWh)": m.get("normal_mae_naive"),
                "Average error ($/MWh)": m["normal_bias"],
                "Risk ceiling coverage": m["coverage"],
                "Spikes": m["n_spike"],
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "Normal MAE ($/MWh)": "{:.1f}", "Naive MAE ($/MWh)": "{:.1f}", "Average error ($/MWh)": "{:+.1f}",
            "Risk ceiling coverage": "{:.0%}"}), hide_index=True, width="stretch")
        latest_version = log["model_version"].dropna().max() if "model_version" in log else None
        if latest_version and latest_version not in set(s["model_version"]):
            st.info(f"The current model ({MODEL_LABELS.get(latest_version, latest_version)}) will appear here "
                    "once its first forecast day's actual prices arrive.")
        st.caption(
            "Naive = the price at the same time on the latest day known when the forecast was issued. "
            "Each row is one model version; the v2.0 model ran systematically low (negative average error) and "
            "was fixed after the backtest exposed it. A few days is too short to judge accuracy; see the "
            "backtest for 20 months of results."
        )

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=s["settlementdate"], y=s["rrp_actual"], name="Actual price",
                                   line=dict(color=C_ACTUAL, width=1.5)))
        fig_t.add_trace(go.Scatter(x=s["settlementdate"], y=s["rrp_base_prediction"], name="Central forecast",
                                   line=dict(color=C_CENTRAL, width=2)))
        fig_t.add_trace(go.Scatter(x=s["settlementdate"], y=s["rrp_risk_ceiling"], name="Risk ceiling (90%)",
                                   line=dict(color=C_RISK, width=1.5, dash="dot")))
        if lead == "today":
            aemo_scored = aemo_log[aemo_log["issue_date"].isin(s["issue_date"].unique())]
            if not aemo_scored.empty:
                fig_t.add_trace(aemo_trace(aemo_scored, name="AEMO PD7DAY"))
        starts = s.groupby("model_version")["settlementdate"].min().sort_values()
        for version, start in starts.iloc[1:].items():
            fig_t.add_vline(x=start, line_dash="dot", line_color="grey", annotation_text=version)
        fig_t.update_layout(height=420, xaxis_title="Date/time (market time)", yaxis_title="Price ($/MWh)",
                            legend=dict(orientation="h", y=-0.2), margin=dict(t=30))
        st.plotly_chart(fig_t, width="stretch")

        st.markdown("##### Compared with AEMO's own outlook")
        if lead == "today":
            vs = to_30min_vs_aemo(s, aemo_log)
            if vs.empty:
                st.info("No day with both a scored forecast and an AEMO outlook yet.")
            else:
                st.dataframe(benchmark_metrics(vs, spike_threshold).style.format({
                    "Typical error (median, $/MWh)": "{:.1f}", "Average error (MAE, $/MWh)": "{:.1f}",
                    "Bias ($/MWh)": "{:+.1f}", "Spikes flagged": "{:.0%}", "Flags that were spikes": "{:.0%}"},
                    na_rep="no spikes"), width="stretch")
                st.caption(f"{vs['issue_date'].nunique()} day(s), 30-minute averages, normal = below "
                           f"\\${spike_threshold}. Our spike flags use the risk ceiling. {AEMO_NOTE}")
        else:
            st.caption(AEMO_NOTE)


# ==================== (3) Market regime ====================
with tab_regime:
    st.subheader("Is the market behaving like the data the models learned from?")

    drift = spike_drift(actuals, spike_threshold)
    msg = (f"Last {drift['recent_days']} days: **{drift['recent_per_day']:.2f} spikes/day** "
           f"vs **{drift['baseline_per_day']:.2f}/day** over the {drift['baseline_days']} days before "
           f"(roughly the models' training window; spike = price ≥ \\${spike_threshold}).")
    if drift["level"] == "calmer":
        st.info(f"🟦 **Calmer than the training period.** {msg} "
                "Models trained on the past year may overstate spike risk, so the risk ceiling is likely "
                "conservative right now.")
    elif drift["level"] == "spikier":
        st.warning(f"🟧 **Spikier than the training period.** {msg} "
                   "The risk ceiling may understate current risk until the models catch up.")
    else:
        st.success(f"🟩 **Similar to the training period.** {msg}")

    regime = cached_regime(spike_threshold)
    col1, col2 = st.columns(2)
    fig_s = go.Figure(go.Bar(x=regime.index, y=regime["spikes"], marker_color=C_RISK, name="Spike intervals"))
    fig_s.update_layout(height=320, title="Spike intervals per month", yaxis_title="Intervals",
                        margin=dict(t=40))
    col1.plotly_chart(fig_s, width="stretch")
    fig_n = go.Figure(go.Bar(x=regime.index, y=regime["negative_share"] * 100, marker_color=C_CENTRAL,
                             name="Negative-price share"))
    fig_n.update_layout(height=320, title="Share of intervals with negative prices", yaxis_title="%",
                        margin=dict(t=40))
    col2.plotly_chart(fig_n, width="stretch")

    fig_c = go.Figure()
    for lead_name, color in [("today", C_CENTRAL), ("tomorrow", C_RISK)]:
        m = bt_monthly[bt_monthly["lead"] == lead_name]
        fig_c.add_trace(go.Scatter(x=pd.to_datetime(m["month"]), y=m["q90_coverage"] * 100, mode="lines+markers",
                                   name=lead_name.capitalize(), line=dict(color=color)))
    fig_c.add_hline(y=90, line_dash="dash", line_color="grey", annotation_text="Target 90%")
    fig_c.update_layout(height=320, title="Risk ceiling coverage by month (walk-forward backtest)",
                        yaxis_title="% of intervals below the ceiling", legend=dict(orientation="h", y=-0.25),
                        margin=dict(t=40))
    st.plotly_chart(fig_c, width="stretch")
    st.caption("Coverage swings month to month even when the long-run average is close to 90%, "
               "which is why it is tracked over time rather than as a single number.")

    st.markdown("#### Compare two periods")
    last_day = (actuals["settlementdate"].max() - pd.Timedelta(minutes=5)).date()
    first_day = (actuals["settlementdate"].min()).date()
    default_b = (last_day - timedelta(days=89), last_day)
    default_a = (default_b[0] - timedelta(days=365), default_b[1] - timedelta(days=365))
    pc1, pc2 = st.columns(2)
    period_a = pc1.date_input("Period A", value=default_a, min_value=first_day, max_value=last_day, key="period_a")
    period_b = pc2.date_input("Period B", value=default_b, min_value=first_day, max_value=last_day, key="period_b")

    if not (isinstance(period_a, tuple) and len(period_a) == 2 and isinstance(period_b, tuple) and len(period_b) == 2):
        st.info("Select a start and end date for both periods.")
    else:
        pa = period_profile(actuals, *period_a, spike_threshold)
        pb = period_profile(actuals, *period_b, spike_threshold)
        label_a = f"A: {period_a[0]:%d %b %y} - {period_a[1]:%d %b %y}"
        label_b = f"B: {period_b[0]:%d %b %y} - {period_b[1]:%d %b %y}"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average price (B)", f"${pb['mean_price']:,.0f}", f"{pb['mean_price'] - pa['mean_price']:+,.0f} vs A",
                  delta_color="inverse")
        m2.metric("Median price (B)", f"${pb['median_price']:,.0f}", f"{pb['median_price'] - pa['median_price']:+,.0f} vs A",
                  delta_color="inverse")
        m3.metric("Spikes per day (B)", f"{pb['spikes_per_day']:.2f}", f"{pb['spikes_per_day'] - pa['spikes_per_day']:+.2f} vs A",
                  delta_color="inverse")
        m4.metric("Negative-price share (B)", f"{pb['negative_share']:.1%}",
                  f"{(pb['negative_share'] - pa['negative_share']) * 100:+.1f} pt vs A", delta_color="off")

        col1, col2 = st.columns(2)
        fig_h = go.Figure()
        for p, label, color in [(pa, label_a, C_A), (pb, label_b, C_B)]:
            fig_h.add_trace(go.Scatter(x=p["hourly_median"].index, y=p["hourly_median"].values, name=label,
                                       line=dict(color=color, width=2)))
        fig_h.update_layout(height=360, title="Median price by hour of day", xaxis_title="Hour",
                            yaxis_title="$/MWh", legend=dict(orientation="h", y=-0.3), margin=dict(t=40))
        col1.plotly_chart(fig_h, width="stretch")

        fig_dc = go.Figure()
        for p, label, color in [(pa, label_a, C_A), (pb, label_b, C_B)]:
            step = max(len(p["duration_price"]) // 2000, 1)  # thin out for plotting
            fig_dc.add_trace(go.Scatter(x=p["duration_pct"][::step], y=p["duration_price"][::step], name=label,
                                        line=dict(color=color, width=2)))
        top = max(np.percentile(pa["duration_price"], 99), np.percentile(pb["duration_price"], 99))
        fig_dc.update_layout(height=360, title="Price duration curve", xaxis_title="% of time price is at or above",
                             yaxis_title="$/MWh", yaxis_range=[min(pa["duration_price"].min(), pb["duration_price"].min(), 0) - 20,
                                                              top * 1.2],
                             legend=dict(orientation="h", y=-0.3), margin=dict(t=40))
        col2.plotly_chart(fig_dc, width="stretch")
        st.caption("The duration curve's top 1% is cut off so the typical range stays readable; "
                   f"max price was \\${pa['max_price']:,.0f} in A and \\${pb['max_price']:,.0f} in B.")


# ==================== (4) Risk strategy backtest ====================
with tab_strategy:
    st.subheader("💰 What if you had acted on the risk ceiling?")
    st.caption(
        "Walk-forward backtest: for each month, the models were trained only on data before it "
        "(730 days for today's forecast, 365 for tomorrow's), "
        "then forecast every day of that month. Nothing after the forecast date was used."
    )

    sc1, sc2 = st.columns([1, 2])
    bt_lead = sc1.radio("Act on", list(LEAD_DAYS), horizontal=True, key="bt_lead",
                        format_func=lambda x: {"today": "Today's forecast", "tomorrow": "Tomorrow's forecast"}[x])
    bt = bt_preds[bt_preds["lead"] == bt_lead]
    bt_min = (bt["settlementdate"].min() - pd.Timedelta(minutes=5)).date()
    bt_max = (bt["settlementdate"].max() - pd.Timedelta(minutes=5)).date()
    bt_range = sc2.date_input("Period", value=(bt_min, bt_max), min_value=bt_min, max_value=bt_max, key="bt_range")
    if isinstance(bt_range, tuple) and len(bt_range) == 2:
        d = (bt["settlementdate"] - pd.Timedelta(minutes=5)).dt.date
        bt = bt[(d >= bt_range[0]) & (d <= bt_range[1])]
    if bt.empty:
        st.warning("There is no data for the selected period.")
        st.stop()

    sim = simulate_hedging_strategy(bt, risk_threshold, hedge_ratio)
    total_naive, total_hedged = sim["cost_naive"].sum(), sim["cost_hedged"].sum()
    savings = total_naive - total_hedged
    q = alert_quality(sim, spike_threshold)

    k1, k2, k3 = st.columns(3)
    k1.metric("Wholesale cost if nothing is done", f"${total_naive / 1e6:,.1f}M")
    k2.metric("Cost with the strategy", f"${total_hedged / 1e6:,.1f}M")
    k3.metric("Estimated savings", f"${savings / 1e6:,.1f}M",
              f"{savings / total_naive * 100:.2f}%" if total_naive else None)
    k4, k5, k6 = st.columns(3)
    k4.metric("Time flagged at risk", f"{q['alerts'] * 5 / 60:,.0f} h")
    k5.metric("Spikes flagged in advance", f"{q['spikes_caught']:.0%}" if q["spikes"] else "no spikes",
              help=f"Share of intervals priced ≥ \\${spike_threshold} that were flagged (recall)")
    k6.metric("Flags that were real spikes", f"{q['alerts_on_spikes']:.0%}" if q["alerts"] else "no flags",
              help=f"Share of flagged intervals that actually reached \\${spike_threshold} (precision). "
                   "Most flags are precautionary: the ceiling is a 90% upper bound, not a spike prediction.")

    sim["month"] = (sim["settlementdate"] - pd.Timedelta(minutes=5)).dt.to_period("M").dt.to_timestamp()
    monthly_sav = sim.groupby("month").apply(lambda g: (g["cost_naive"] - g["cost_hedged"]).sum() / 1e6,
                                             include_groups=False)
    col1, col2 = st.columns(2)
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=sim["settlementdate"], y=sim["cumulative_cost_naive"] / 1e6,
                                 name="If nothing is done", line=dict(color=C_NAIVE)))
    fig_sim.add_trace(go.Scatter(x=sim["settlementdate"], y=sim["cumulative_cost_hedged"] / 1e6,
                                 name="With the strategy", line=dict(color=C_HEDGED)))
    fig_sim.update_layout(height=360, title="Cumulative wholesale cost", yaxis_title="$ million",
                          legend=dict(orientation="h", y=-0.25), margin=dict(t=40))
    col1.plotly_chart(fig_sim, width="stretch")
    fig_ms = go.Figure(go.Bar(x=monthly_sav.index, y=monthly_sav.values, marker_color=C_HEDGED))
    fig_ms.update_layout(height=360, title="Savings by month", yaxis_title="$ million", margin=dict(t=40))
    col2.plotly_chart(fig_ms, width="stretch")
    st.caption(
        "* Proof of concept. Assumes the flagged share of NSW demand could simply be curtailed, and ignores "
        "the cost of doing so or of sourcing the power elsewhere. Savings concentrate in spiky months; "
        "in calm months (see Market regime) there is little to save."
    )

    with st.expander("🔔 Periods flagged at risk"):
        events = sim[sim["is_high_risk"]][["settlementdate", "rrp_actual", "rrp_risk_ceiling", "totaldemand"]]
        st.dataframe(events.rename(columns={
            "settlementdate": "Date/time", "rrp_actual": "Actual price ($)",
            "rrp_risk_ceiling": "Risk ceiling forecast ($)", "totaldemand": "Demand (MW)",
        }).round({"Actual price ($)": 1, "Risk ceiling forecast ($)": 1, "Demand (MW)": 1}),
            width="stretch", hide_index=True)
        st.caption(f"{len(events):,} flagged 5-minute intervals")

    with st.expander("📊 Model accuracy (technical details)"):
        tech = []
        for lead_name, g in bt_preds.groupby("lead", sort=False):
            m = price_metrics(g, spike_threshold)
            mb = bt_monthly[bt_monthly["lead"] == lead_name]
            tech.append({
                "Horizon": lead_name, "Normal MAE ($/MWh)": m["normal_mae"],
                "Naive MAE ($/MWh)": np.average(mb["normal_MAE_naive"], weights=mb["intervals"] - mb["spikes"]),
                "Average error ($/MWh)": m["normal_bias"], "Spike MAE, risk ceiling": m["spike_mae_q90"],
                "Risk ceiling coverage": m["coverage"],
                "Months beating naive": f"{(mb['normal_MAE'] < mb['normal_MAE_naive']).sum()}/{len(mb)}",
            })
        st.dataframe(pd.DataFrame(tech).style.format({
            "Normal MAE ($/MWh)": "{:.2f}", "Naive MAE ($/MWh)": "{:.2f}", "Average error ($/MWh)": "{:+.2f}",
            "Spike MAE, risk ceiling": "{:,.0f}", "Risk ceiling coverage": "{:.1%}"}),
            hide_index=True, width="stretch")
        st.caption(f"Walk-forward, {bt_monthly['month'].min()} to {bt_monthly['month'].max()}, "
                   f"all intervals (not affected by the period filter above); normal = price below \\${spike_threshold}. "
                   "Naive MAE uses the backtest's fixed \\$300 spike definition.")
        st.markdown(
            "- **Central forecast**: LightGBM, L1 (median) objective on asinh(price / 100); tuned for typical conditions\n"
            "- **Risk ceiling**: LightGBM quantile regression at the 90th percentile\n"
            "- **Inputs**: only information available at issue time: prices and demand up to the previous "
            "midnight, same time on the latest day and a week earlier, calendar, temperature forecast; "
            "today's price models also use AEMO's PD7DAY outlook published before 06:00\n"
            "- Retrained every morning on the latest 730 days (today) / 365 days (tomorrow); "
            "separate models for each horizon"
        )

        st.markdown("##### Today's forecast vs AEMO PD7DAY (walk-forward, 30-minute)")
        st.dataframe(benchmark_metrics(aemo_bt, spike_threshold).style.format({
            "Typical error (median, $/MWh)": "{:.1f}", "Average error (MAE, $/MWh)": "{:.1f}",
            "Bias ($/MWh)": "{:+.1f}", "Spikes flagged": "{:.0%}", "Flags that were spikes": "{:.0%}"}),
            width="stretch")
        st.caption(
            "AEMO's outlook is closer on typical half-hours and flags more spikes, but occasionally projects very "
            "high prices that don't happen, which inflates its average error. The two carry different "
            "information: a plain average of them beats either alone on normal-price error. "
            f"{AEMO_NOTE}"
        )
