# main.py
import os
from src.pipeline import AEMODataPipeline
from src.train import train_and_evaluate_models

if __name__ == "__main__":
    # 1. Configure and run the pipeline (data engineering layer)
    pipeline = AEMODataPipeline(region="NSW", year="2025", download_dir="data/processed")
    df_cleansed = pipeline.run_pipeline(lat=-33.86, lon=151.20)

    # 2. Run model training against the generated Parquet path (data science layer)
    parquet_path = os.path.join("data", "processed", "cleansed_aemo_NSW_2025.parquet")
    train_and_evaluate_models(parquet_path)

    print("\n🎉 The full system pipeline completed successfully!")