"""
AEMO PD7DAY (7-day pre-dispatch) price forecasts, used as a benchmark: "how does our
forecast compare with the price AEMO itself was projecting at the time?"

PD7DAY runs three times a day (RUN_DATETIME 07:30 / 13:00 / 18:00 market time) and is
published ~17 minutes before its RUN_DATETIME (LASTCHANGED). Prices are 30-minute,
interval-ending, out to ~7 days. For a forecast issued at D 06:00, the latest published
run is D-1 18:00, which covers both day D and D+1.

Sources:
- History: MMSDM monthly archive, table PD7DAY_PRICESOLUTION. Files up to 2026-07 are
  cumulative (each holds all runs from 2024-04 on); later files hold one month. The
  loader therefore walks back from the end month until the requested range is covered.
- Live: NEMweb Reports/Current/PD7Day (about 60 days of individual runs).
"""
import io
import os
import re
import zipfile

import pandas as pd
import requests

from src.aemo_downloader import USER_AGENT

MMSDM_URL = ("https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{y}/MMSDM_{y}_{m:02d}/"
             "MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_ARCHIVE%23PD7DAY_PRICESOLUTION%23FILE01%23{y}{m:02d}010000.zip")
CURRENT_URL = "https://www.nemweb.com.au/Reports/Current/PD7Day/"
CACHE_DIR = os.path.join("data", "raw", "pd7day")
COLUMNS = ["RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID", "RRP", "LASTCHANGED"]
ISSUE_HOUR = 6  # our forecasts are issued at 06:00 market time


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _tidy(df: pd.DataFrame, region_id: str) -> pd.DataFrame:
    df = df[df["REGIONID"] == region_id]
    out = pd.DataFrame({
        "run_datetime": pd.to_datetime(df["RUN_DATETIME"]),
        "interval_datetime": pd.to_datetime(df["INTERVAL_DATETIME"]),
        "aemo_rrp": df["RRP"].astype(float),
        "published": pd.to_datetime(df["LASTCHANGED"]),
    })
    return out.drop_duplicates(["run_datetime", "interval_datetime"]).reset_index(drop=True)


def _load_archive_file(year: int, month: int, region_id: str, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """One MMSDM PD7DAY_PRICESOLUTION file, region-filtered and cached as parquet."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"archive_{year}{month:02d}_{region_id}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    url = MMSDM_URL.format(y=year, m=month)
    print(f"  PD7DAY archive {year}-{month:02d}: downloading")
    resp = _session().get(url, timeout=300)
    if resp.status_code == 404:
        return pd.DataFrame(columns=["run_datetime", "interval_datetime", "aemo_rrp", "published"])
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf, zf.open(zf.namelist()[0]) as f:
        # Line 1 is the 'C' header; the trailing 'C,END OF REPORT' row drops out in the region filter
        chunks = pd.read_csv(f, skiprows=1, usecols=COLUMNS, chunksize=500_000, low_memory=False)
        df = _tidy(pd.concat([c[c["REGIONID"] == region_id] for c in chunks]), region_id)
    df.to_parquet(cache, index=False)
    return df


def load_pd7day_history(start: pd.Timestamp, end: pd.Timestamp, region_id: str = "NSW1") -> pd.DataFrame:
    """All archived runs published between start and end (walks back month by month)."""
    frames, month = [], pd.Period(end, freq="M")
    while True:
        df = _load_archive_file(month.year, month.month, region_id)
        frames.append(df)
        if (not df.empty and df["published"].min() <= start) or month < pd.Period(start, freq="M") - 1:
            break
        month -= 1
    df = pd.concat(frames).drop_duplicates(["run_datetime", "interval_datetime"])
    return df[(df["published"] >= start - pd.Timedelta(days=1)) & (df["published"] <= end)].reset_index(drop=True)


def _parse_current_file(content: bytes, region_id: str) -> pd.DataFrame:
    """A NEMweb PD7DAY report holds several tables; keep the PRICESOLUTION section."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf, zf.open(zf.namelist()[0]) as f:
        lines = [ln for ln in io.TextIOWrapper(f, encoding="utf-8")
                 if ln.startswith(("I,PD7DAY,PRICESOLUTION", "D,PD7DAY,PRICESOLUTION"))]
    df = pd.read_csv(io.StringIO("".join(lines)))
    return _tidy(df, region_id)


def load_pd7day_latest(issue_time: pd.Timestamp, region_id: str = "NSW1") -> pd.DataFrame:
    """The latest run published at or before issue_time, from NEMweb's Current folder."""
    s = _session()
    listing = s.get(CURRENT_URL, timeout=60)
    listing.raise_for_status()
    names = sorted(set(re.findall(r"PUBLIC_PD7DAY_(\d{14})_\d+\.zip", listing.text, flags=re.I)))
    names = [n for n in names if pd.Timestamp(n[:8] + " " + n[8:]) <= issue_time]
    if not names:
        raise FileNotFoundError(f"No PD7DAY run published before {issue_time} in {CURRENT_URL}")
    href = re.search(rf'href="([^"]*PUBLIC_PD7DAY_{names[-1]}_\d+\.zip)"', listing.text, flags=re.I).group(1)
    resp = s.get(requests.compat.urljoin(CURRENT_URL, href), timeout=300)
    resp.raise_for_status()
    return _parse_current_file(resp.content, region_id)


def as_of_issue(pd7: pd.DataFrame, issue_day: pd.Timestamp, lead_days: int) -> pd.DataFrame:
    """
    AEMO's price for trading day issue_day + lead_days - 1, from the latest run published
    before issue_day 06:00. Returns interval_datetime (30-min, interval-ending) and aemo_rrp.
    """
    issue_time = issue_day + pd.Timedelta(hours=ISSUE_HOUR)
    available = pd7[pd7["published"] <= issue_time]
    if available.empty:
        return available[["interval_datetime", "aemo_rrp", "run_datetime"]]
    run = available["run_datetime"].max()
    day_start = issue_day + pd.Timedelta(days=lead_days - 1)
    rows = available[(available["run_datetime"] == run)
                     & (available["interval_datetime"] > day_start)
                     & (available["interval_datetime"] <= day_start + pd.Timedelta(days=1))]
    return rows[["interval_datetime", "aemo_rrp", "run_datetime"]].reset_index(drop=True)


def save_benchmark(issue_day: pd.Timestamp, region_id: str = "NSW1",
                   out_dir: str = os.path.join("data", "forecasts")) -> pd.DataFrame:
    """
    Log AEMO's PD7DAY price for today's trading day (lead 1 only: for tomorrow, the run
    available at 06:00 predates the bid deadline and often sits at the price cap), from
    the latest run published before issue_day 06:00. Re-runs replace the same issue date.
    """
    issue_day = pd.Timestamp(issue_day).normalize()
    pd7 = load_pd7day_latest(issue_day + pd.Timedelta(hours=ISSUE_HOUR), region_id)
    bench = as_of_issue(pd7, issue_day, lead_days=1)
    bench.insert(0, "issue_date", issue_day.date())

    path = os.path.join(out_dir, "aemo_pd7day_log.csv")
    log = bench
    if os.path.exists(path):
        prev = pd.read_csv(path, parse_dates=["interval_datetime", "run_datetime"])
        prev["issue_date"] = pd.to_datetime(prev["issue_date"]).dt.date
        log = pd.concat([prev[prev["issue_date"] != issue_day.date()], bench], ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    log.sort_values(["issue_date", "interval_datetime"]).to_csv(path, index=False, float_format="%.4f")
    return bench
