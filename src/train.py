# src/train.py
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_and_evaluate_models(parquet_path: str, output_dir: str = "data/processed"):
    print(f"--- [Step 5] Loading data from the Parquet file ---")
    df = pd.read_parquet(parquet_path)

    # Check that required columns exist (Capacity + the 2 finalized features)
    required_cols = [
        "availablegeneration", "reserve_margin",
        "reserve_margin_slope_1h", "rrp_volatility_1h"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Required columns not found: {missing_cols}. "
                        f"Check that pipeline.py's generate_features is up to date.")

    df["reserve_margin_ratio"] = df["reserve_margin"] / (df["totaldemand"] + 1e-5)

    # 1. Data split (time-series based: Jan-Oct Train, Nov Val, Dec Test)
    train_df = df[df["month"] <= 10]
    val_df = df[df["month"] == 11]
    test_df = df[df["month"] == 12]

    # --- Task A: Demand Forecasting ---
    print("\n--- Task A: Starting training of the demand forecasting model ---")
    features_demand = [
        "temperature", "year", "month", "day", "hour", "day_of_week",
        "is_holiday", "season", "demand_lag_1h", "demand_lag_24h",
        "demand_roll_mean_6h", "Temp_Roll_Mean_1H"
    ]

    X_train_d, y_train_d = train_df[features_demand], train_df["totaldemand"]
    X_val_d, y_val_d = val_df[features_demand], val_df["totaldemand"]
    X_test_d, y_test_d = test_df[features_demand], test_df["totaldemand"]

    train_data_d = lgb.Dataset(X_train_d, label=y_train_d)
    val_data_d = lgb.Dataset(X_val_d, label=y_val_d, reference=train_data_d)

    params_demand = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 31, "seed": 42, "verbose": -1
    }

    model_demand = lgb.train(
        params_demand, train_data_d, num_boost_round=1000,
        valid_sets=[train_data_d, val_data_d],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    pred_d = model_demand.predict(X_test_d, num_iteration=model_demand.best_iteration)
    r2_d = r2_score(y_test_d, pred_d)
    print(f"✅ Demand forecasting model complete! Final test data (Dec) R2 score: {r2_d:.4f}")

    # --- Task B: Price Forecasting (dual-axis operation: central prediction + risk ceiling) ---
    print("\n--- Task B: Starting training of the price forecasting model (dual-axis operation) ---")

    def to_asinh(x):
        return np.log(x + np.sqrt(x**2 + 1))

    def from_asinh(x):
        return np.sinh(x)

    # Formally adopt the 2 finalized features (reserve_margin_slope_1h, rrp_volatility_1h)
    features_rrp_t = [
        "temperature", "year", "month", "day", "hour", "day_of_week", "is_holiday", "season",
        "totaldemand",
        "availablegeneration", "reserve_margin", "reserve_margin_ratio",
        "reserve_margin_slope_1h", "rrp_volatility_1h"
    ]

    y_train_r_t = to_asinh(train_df["rrp"])
    y_val_r_t = to_asinh(val_df["rrp"])

    X_train_r = train_df[features_rrp_t].copy()
    X_val_r = val_df[features_rrp_t].copy()
    X_test_r = test_df[features_rrp_t].copy()

    for _df, _X in [(train_df, X_train_r), (val_df, X_val_r), (test_df, X_test_r)]:
        _X["rrp_lag_1h_t"] = to_asinh(_df["rrp_lag_1h"])
        _X["rrp_lag_24h_t"] = to_asinh(_df["rrp_lag_24h"])
        _X["rrp_roll_max_6h_t"] = to_asinh(_df["rrp_roll_max_6h"])

    y_test_r_original = test_df["rrp"].values
    normal_mask = y_test_r_original < 300
    spike_mask = y_test_r_original >= 300

    # --- (1) Central prediction model: tuned specifically for normal conditions ---
    print("--- Training the central prediction model (tuned specifically for normal conditions) ---")

    train_data_r = lgb.Dataset(X_train_r, label=y_train_r_t)
    val_data_r = lgb.Dataset(X_val_r, label=y_val_r_t, reference=train_data_r)

    params_base_model = {
        "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 15,          # Keep leaves shallow rather than trying to capture complex spikes
        "min_data_in_leaf": 100,   # Avoid being pulled around by a small number of spike-noise points
        "seed": 42, "verbose": -1
    }

    model_rrp = lgb.train(
        params_base_model, train_data_r, num_boost_round=1000,
        valid_sets=[train_data_r, val_data_r],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    pred_r_t = model_rrp.predict(X_test_r, num_iteration=model_rrp.best_iteration)
    pred_r_original = from_asinh(pred_r_t)

    r2_r = r2_score(y_test_r_original, pred_r_original)
    mae_r = mean_absolute_error(y_test_r_original, pred_r_original)
    print(f"✅ Central prediction model complete! Overall R2: {r2_r:.4f}, overall MAE: {mae_r:.2f}")

    r2_normal = r2_score(y_test_r_original[normal_mask], pred_r_original[normal_mask])
    mae_normal = mean_absolute_error(y_test_r_original[normal_mask], pred_r_original[normal_mask])
    mae_spike = mean_absolute_error(y_test_r_original[spike_mask], pred_r_original[spike_mask])
    print(f"[Diagnostics] Normal-conditions MAE: {mae_normal:.2f}, normal-conditions R2: {r2_normal:.4f} (count: {normal_mask.sum()}/{len(y_test_r_original)})")
    print(f"[Diagnostics] Spike-conditions MAE: {mae_spike:.2f} (count: {spike_mask.sum()}/{len(y_test_r_original)})")

    importance = pd.DataFrame({
        'feature': X_train_r.columns,
        'importance': model_rrp.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    print(f"[Diagnostics] Central prediction model feature importance (gain):\n{importance}")

    # --- (2) Risk ceiling model: Quantile regression (alpha=0.9) ---
    print("\n--- Training the risk ceiling model (Q90 quantile regression) ---")

    params_q90 = {
        "objective": "quantile", "alpha": 0.9,
        "metric": "quantile", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 31, "seed": 42, "verbose": -1
    }

    train_data_q = lgb.Dataset(X_train_r, label=y_train_r_t)
    val_data_q = lgb.Dataset(X_val_r, label=y_val_r_t, reference=train_data_q)

    model_rrp_q90 = lgb.train(
        params_q90, train_data_q, num_boost_round=1000,
        valid_sets=[train_data_q, val_data_q],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    pred_q90_t = model_rrp_q90.predict(X_test_r, num_iteration=model_rrp_q90.best_iteration)
    pred_q90_original = from_asinh(pred_q90_t)

    mae_spike_q90 = mean_absolute_error(y_test_r_original[spike_mask], pred_q90_original[spike_mask])
    coverage = (y_test_r_original <= pred_q90_original).mean()
    print(f"✅ Risk ceiling model complete! Spike-conditions MAE: {mae_spike_q90:.2f}")
    print(f"[Diagnostics] Coverage rate (share of actuals below the Q90 prediction; target ~=90%): {coverage:.4f}")

    # --- (3) Building the dual-axis output ---
    print("\n--- Building the dual-axis output (central prediction + risk ceiling) ---")

    df_output = test_df[["settlementdate", "totaldemand"]].copy()
    df_output["rrp_actual"] = y_test_r_original
    df_output["rrp_base_prediction"] = pred_r_original
    df_output["rrp_risk_ceiling"] = pred_q90_original

    output_path = f"{output_dir}/rrp_dual_prediction_output.csv"
    df_output.to_csv(output_path, index=False)
    print(f"🎉 Dual-axis prediction results saved to: {output_path}")
    print(df_output.head())

    return model_demand, model_rrp, model_rrp_q90