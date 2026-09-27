import os
import time
from datetime import datetime, timedelta, timezone

import requests

# AEMO "Aggregated price and demand data" monthly CSVs.
# Note: the bare aemo.com.au host 301-redirects to www, and Cloudflare in front of it
# returns 403 for the default Python-urllib User-Agent (requests' default UA passes),
# so always use requests with an explicit browser-like UA.
AEMO_PRICE_DEMAND_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
EXPECTED_HEADER = "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE"

# AEMO's market clock is a fixed UTC+10 (no DST)
AEMO_TZ = timezone(timedelta(hours=10))


def price_demand_filename(yyyymm: str, region_id: str) -> str:
    return f"PRICE_AND_DEMAND_{yyyymm}_{region_id}.csv"


def _needs_refresh(yyyymm: str, path: str, now: datetime) -> bool:
    """
    A month is re-downloaded if the file is missing, or if the month is still open
    (current month) or only recently closed (previous month; AEMO may still append
    the final interval / revise values shortly after month end).
    """
    if not os.path.exists(path):
        return True
    current = now.strftime("%Y%m")
    previous = (now.replace(day=1) - timedelta(days=1)).strftime("%Y%m")
    return yyyymm in (current, previous)


def download_price_and_demand(region_id: str, months: list[str], dest_dir: str,
                              retries: int = 3, timeout: int = 60) -> list[str]:
    """
    Download AEMO PRICE_AND_DEMAND monthly CSVs for the given region (e.g. 'NSW1')
    and months (['YYYYMM', ...]) into dest_dir.

    - Months in the future (relative to AEMO market time) are skipped.
    - Already-present closed months are not re-downloaded.
    - If a download fails but a previous copy exists locally, the local copy is kept
      (so a transient AEMO/Cloudflare outage doesn't break the daily pipeline).

    Returns the list of local file paths that are available for the requested months.
    """
    os.makedirs(dest_dir, exist_ok=True)
    now = datetime.now(AEMO_TZ)
    current = now.strftime("%Y%m")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    available = []
    for yyyymm in months:
        if yyyymm > current:
            continue

        path = os.path.join(dest_dir, price_demand_filename(yyyymm, region_id))
        if not _needs_refresh(yyyymm, path, now):
            available.append(path)
            continue

        url = AEMO_PRICE_DEMAND_URL.format(yyyymm=yyyymm, region=region_id)
        content = None
        for attempt in range(1, retries + 1):
            try:
                resp = session.get(url, timeout=timeout)
                if resp.status_code == 404:
                    # Current month may not be published yet on the 1st
                    print(f"  {yyyymm}: not published yet (404)")
                    break
                resp.raise_for_status()
                if not resp.content.decode("utf-8", errors="replace").startswith(EXPECTED_HEADER):
                    raise ValueError("unexpected response body (not a PRICE_AND_DEMAND CSV)")
                content = resp.content
                break
            except (requests.RequestException, ValueError) as e:
                print(f"  {yyyymm}: attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)

        if content is not None:
            # Write atomically so an interrupted run never leaves a truncated CSV
            tmp_path = path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(content)
            os.replace(tmp_path, path)
            print(f"  {yyyymm}: downloaded ({len(content):,} bytes)")
            available.append(path)
        elif os.path.exists(path):
            print(f"  {yyyymm}: download failed, using existing local copy")
            available.append(path)

    return available
