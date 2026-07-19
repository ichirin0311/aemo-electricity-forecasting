import os
import glob
import urllib.request
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import holidays
import openmeteo_requests
import requests_cache
from retry_requests import retry
from nemosis import dynamic_data_compiler


class AEMODataPipeline:
    def __init__(self, region: str, year: str, download_dir: str = "data_pipeline_output"):
        """
        Initialize the pipeline
        """
        self.region = region.upper()  # e.g. 'NSW'
        self.year = str(year)         # e.g. '2025'
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

        # Cache location for NEMOSIS
        self.nemosis_cache = os.path.join("data", "raw", "nemosis_cache")
        os.makedirs(self.nemosis_cache, exist_ok=True)

        # Open-Meteo API configuration (caching and retries)
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)


    def fetch_aemo_data(self) -> pd.DataFrame:
        print(f"--- [Step 1] Starting collection of {self.year} AEMO data ---")
        csv_files = glob.glob(os.path.join("data","raw","aemo_data_1year", "*PRICE_AND_DEMAND*.csv"))

        if not csv_files:
            raise FileNotFoundError("No AEMO CSV files found in the specified folder.")

        dfs = []
        for file in csv_files:
            df_month = pd.read_csv(file)
            df_month.columns = df_month.columns.str.lower()
            df_month['region'] = df_month['region'].str.lower()  # <- added: lowercase the values too
            df_month = df_month[df_month['region'] == f"{self.region.lower()}1"]
            dfs.append(df_month)

        df_aemo = pd.concat(dfs, ignore_index=True)
        df_aemo['settlementdate'] = pd.to_datetime(df_aemo['settlementdate'])
        df_aemo = df_aemo.sort_values('settlementdate').reset_index(drop=True)

        print(f"AEMO data collection complete: {len(df_aemo)} rows")
        return df_aemo


    def fetch_weather_data(self, start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
        """
        Fetch temperature data for the specified period from the Open-Meteo Historical API
        """
        print(f"--- [Step 2] Starting collection of weather data from the Open-Meteo API ---")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m"
        }

        responses = self.openmeteo.weather_api(url, params=params)
        response = responses[0]

        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        df_weather = pd.DataFrame(data={
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature": hourly_temperature_2m
        })

        # AEMO's market clock is a fixed UTC+10 with no daylight saving, so convert fixed to Etc/GMT-10
        df_weather['date'] = df_weather['date'].dt.tz_convert('Etc/GMT-10').dt.tz_localize(None)

        print(f"Weather data collection complete: {len(df_weather)} rows")
        return df_weather


    def fetch_capacity_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Capacity data from DISPATCHREGIONSUM via NEMOSIS
        """
        print(f"--- [Step 2.5] Starting collection of AEMO Capacity data ---")
        region_id = f"{self.region}1"  # e.g. 'NSW1'

        # NEMOSIS requires the "YYYY/MM/DD HH:MM:SS" format, so convert to it
        start_time = pd.to_datetime(start_date).strftime('%Y/%m/%d 00:00:00')
        end_time = pd.to_datetime(end_date).strftime('%Y/%m/%d 23:59:59')

        df_capacity = dynamic_data_compiler(
            start_time, end_time,
            table_name="DISPATCHREGIONSUM",
            raw_data_location=self.nemosis_cache,
            filter_cols=['REGIONID'],
            filter_values=([region_id],),
            select_columns=['SETTLEMENTDATE', 'REGIONID', 'AVAILABLEGENERATION', 'DISPATCHABLEGENERATION']
        )

        df_capacity = df_capacity.rename(columns={'SETTLEMENTDATE': 'settlementdate'})
        df_capacity['settlementdate'] = pd.to_datetime(df_capacity['settlementdate'])
        df_capacity = df_capacity.sort_values('settlementdate').reset_index(drop=True)
        df_capacity.columns = df_capacity.columns.str.lower()

        print(f"Capacity data collection complete: {len(df_capacity)} rows")
        return df_capacity
    def transform_and_merge(self, df_aemo, df_weather, df_capacity=None) -> pd.DataFrame:
        print(f"--- [Step 3] Running time-series merge and linear interpolation ---")
        df_weather_5min = (
            df_weather.set_index('date')
            .resample('5min')
            .interpolate(method='linear')
            .reset_index()
        )
        merged = pd.merge(
            df_aemo, df_weather_5min,
            left_on='settlementdate', right_on='date',
            how='left'
        )

        if df_capacity is not None:
            df_capacity.columns = df_capacity.columns.str.lower()
            merged = pd.merge(
                merged, df_capacity,
                on='settlementdate',
                how='left'
            )
            # Also create the reserve margin (spare capacity) feature
            merged['reserve_margin'] = merged['availablegeneration'] - merged['totaldemand']

        return merged

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bulk generation of feature engineering (calendar, lag, rolling)
        """
        print(f"--- [Step 4] Bulk generation of feature engineering ---")

        # Calendar features
        df['year'] = df['settlementdate'].dt.year
        df['month'] = df['settlementdate'].dt.month
        df['day'] = df['settlementdate'].dt.day
        df['hour'] = df['settlementdate'].dt.hour
        df['day_of_week'] = df['settlementdate'].dt.dayofweek

        # Holiday flag
        au_holidays = holidays.Australia(subdiv=self.region, years=int(self.year))
        df['is_holiday'] = df['settlementdate'].dt.date.apply(lambda x: 1 if x in au_holidays else 0)

        # Southern hemisphere season flag
        df['season'] = df['month'].apply(
            lambda m: 0 if m in [12, 1, 2] else (1 if m in [3, 4, 5] else (2 if m in [6, 7, 8] else 3))
        )

        # Demand lag/rolling features
        df['demand_lag_1h'] = df['totaldemand'].shift(12)
        df['demand_lag_24h'] = df['totaldemand'].shift(288)
        df['demand_roll_mean_6h'] = df['totaldemand'].shift(1).rolling(window=72).mean()

        # Price (RRP) lag/rolling features
        df['rrp_lag_1h'] = df['rrp'].shift(12)
        df['rrp_lag_24h'] = df['rrp'].shift(288)
        df['rrp_roll_max_6h'] = df['rrp'].shift(1).rolling(window=72).max()

        # Temperature rolling
        df['Temp_Roll_Mean_1H'] = df['temperature'].rolling(window=12).mean()

        # Added after the existing lag/rolling features within generate_features in src/pipeline.py
        # Rate of change of reserve margin (slope over 1 hour = 12 steps)
        df['reserve_margin_slope_1h'] = (df['reserve_margin'] - df['reserve_margin'].shift(12)) / 12

        # Price volatility (std dev over the last hour, excluding the current point)
        df['rrp_volatility_1h'] = df['rrp'].shift(1).rolling(window=12).std()

        # Drop missing values and clean up
        df = df.dropna().reset_index(drop=True)
        return df

    def run_pipeline(self, lat: float, lon: float) -> pd.DataFrame:
        df_aemo = self.fetch_aemo_data()

        start_date = df_aemo['settlementdate'].min().strftime('%Y-%m-%d')
        end_date = df_aemo['settlementdate'].max().strftime('%Y-%m-%d')

        df_weather = self.fetch_weather_data(start_date, end_date, lat, lon)
        df_capacity = self.fetch_capacity_data(start_date, end_date)

        df_merged = self.transform_and_merge(df_aemo, df_weather, df_capacity)
        df_final = self.generate_features(df_merged)

        output_path = os.path.join(self.download_dir, f"cleansed_aemo_{self.region}_{self.year}.parquet")
        df_final.to_parquet(output_path, index=False)

        print(f"🎉 All pipeline processing completed successfully!")
        print(f"Final output saved to: {output_path} (shape: {df_final.shape})")

        return df_final