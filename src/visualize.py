# src/visualize.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score


def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["settlementdate"])
    return df


def compute_kpis(df: pd.DataFrame, spike_threshold: float = 300) -> dict:
    normal_mask = df["rrp_actual"] < spike_threshold
    spike_mask = df["rrp_actual"] >= spike_threshold

    kpis = {
        "normal_mae": mean_absolute_error(df.loc[normal_mask, "rrp_actual"], df.loc[normal_mask, "rrp_base_prediction"]),
        "normal_r2": r2_score(df.loc[normal_mask, "rrp_actual"], df.loc[normal_mask, "rrp_base_prediction"]),
        "spike_mae_base": mean_absolute_error(df.loc[spike_mask, "rrp_actual"], df.loc[spike_mask, "rrp_base_prediction"]),
        "spike_mae_q90": mean_absolute_error(df.loc[spike_mask, "rrp_actual"], df.loc[spike_mask, "rrp_risk_ceiling"]),
        "coverage": (df["rrp_actual"] <= df["rrp_risk_ceiling"]).mean(),
        "n_normal": int(normal_mask.sum()),
        "n_spike": int(spike_mask.sum()),
    }
    return kpis


def simulate_hedging_strategy(
    df: pd.DataFrame,
    risk_threshold: float = 300,
    hedge_ratio: float = 0.1,
) -> pd.DataFrame:
    """
    Simulates the cost reduction effect of avoiding a fixed share of demand
    (hedge_ratio) during periods where the risk ceiling (Q90) exceeds the
    threshold (assumed shifted to another time period and simply excluded
    from cost).

    Note: this is a simplified simulation that does not account for the
    price at the time demand is shifted to (a conservative assumption that
    "avoided demand counts as savings").
    """
    df = df.copy()
    df["cost_naive"] = df["rrp_actual"] * df["totaldemand"]

    is_high_risk = df["rrp_risk_ceiling"] >= risk_threshold
    hedged_demand = np.where(is_high_risk, df["totaldemand"] * (1 - hedge_ratio), df["totaldemand"])
    df["cost_hedged"] = df["rrp_actual"] * hedged_demand

    df["cumulative_cost_naive"] = df["cost_naive"].cumsum()
    df["cumulative_cost_hedged"] = df["cost_hedged"].cumsum()

    return df


def build_dashboard(
    df: pd.DataFrame,
    kpis: dict,
    df_sim: pd.DataFrame,
    output_html: str = "data/processed/rrp_dashboard.html",
):
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.45, 0.15, 0.4],
        vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "table"}], [{"type": "xy"}]],
        subplot_titles=(
            "RRP: Actual vs Base Prediction vs Risk Ceiling (Q90)",
            "Accuracy Summary (KPI)",
            "Bidding Hedge Strategy Simulation: Cumulative Cost Comparison",
        ),
    )

    # --- Section 1: Time series chart ---
    fig.add_trace(go.Scatter(
        x=df["settlementdate"], y=df["rrp_risk_ceiling"],
        name="Risk Ceiling (Q90)", line=dict(color="rgba(255,99,71,0.4)", width=1),
        fill=None,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["settlementdate"], y=df["rrp_actual"],
        name="Actual RRP", line=dict(color="black", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["settlementdate"], y=df["rrp_base_prediction"],
        name="Base Prediction", line=dict(color="royalblue", width=1.5, dash="dot"),
    ), row=1, col=1)

    # --- Section 2: KPI table ---
    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], fill_color="lightgrey", align="left"),
        cells=dict(values=[
            ["Normal MAE", "Normal R2", "Spike MAE (Base Prediction)", "Spike MAE (Q90)", "Coverage (target 90%)", "Normal count", "Spike count"],
            [
                f"{kpis['normal_mae']:.2f}",
                f"{kpis['normal_r2']:.4f}",
                f"{kpis['spike_mae_base']:.2f}",
                f"{kpis['spike_mae_q90']:.2f}",
                f"{kpis['coverage']:.4f}",
                kpis['n_normal'],
                kpis['n_spike'],
            ]
        ], align="left"),
    ), row=2, col=1)

    # --- Section 3: Cumulative cost comparison ---
    fig.add_trace(go.Scatter(
        x=df_sim["settlementdate"], y=df_sim["cumulative_cost_naive"],
        name="Cumulative Cost (No Hedge)", line=dict(color="firebrick"),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df_sim["settlementdate"], y=df_sim["cumulative_cost_hedged"],
        name="Cumulative Cost (Q90 Hedge Strategy)", line=dict(color="seagreen"),
    ), row=3, col=1)

    total_savings = df_sim["cumulative_cost_naive"].iloc[-1] - df_sim["cumulative_cost_hedged"].iloc[-1]
    savings_pct = total_savings / df_sim["cumulative_cost_naive"].iloc[-1] * 100

    fig.update_layout(
        height=1100,
        title_text=f"AEMO NSW1 Electricity Price Forecast Dashboard (Dec 2025 Test Period)"
                    f" | Estimated Cost Savings from Hedge Strategy: ${total_savings:,.0f} ({savings_pct:.2f}%)",
        showlegend=True,
    )

    fig.write_html(output_html)
    print(f"🎉 Dashboard saved: {output_html}")
    print(f"[Diagnostics] Estimated cost savings from hedge strategy: ${total_savings:,.0f} ({savings_pct:.2f}%)")


if __name__ == "__main__":
    df = load_predictions("data/processed/rrp_dual_prediction_output.csv")
    kpis = compute_kpis(df)
    df_sim = simulate_hedging_strategy(df, risk_threshold=300, hedge_ratio=0.1)
    build_dashboard(df, kpis, df_sim)