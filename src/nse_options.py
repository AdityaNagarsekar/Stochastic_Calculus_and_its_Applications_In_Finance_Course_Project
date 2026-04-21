"""
NSE Bhav Copy downloader for historical Nifty options data.

NSE publishes daily F&O Bhav copy files (CSV) at:
  https://archives.nseindia.com/content/historical/DERIVATIVES/{YYYY}/{MMM}/fo{DD}{MMM}{YYYY}bhav.csv.zip

Each file has one row per contract traded that day. We filter for:
  - NIFTY index options (OPTIDX, symbol=NIFTY)
  - ATM call (CE) with the nearest weekly/monthly expiry
  - Strike nearest to the closing spot price

Usage:
    from nse_options import build_options_dataset
    df_opts = build_options_dataset(spot_df, start='2022-10-18', end='2024-12-26')
"""

import io
import time
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
OPT_CACHE = DATA_DIR / "raw" / "nse_options"
OPT_CACHE.mkdir(parents=True, exist_ok=True)

# NSE blocks default User-Agents; mimic a browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com",
}

MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


def _bhav_url(dt: datetime) -> str:
    d = dt.strftime("%d")
    m = MONTH_MAP[dt.month]
    y = dt.strftime("%Y")
    return (
        f"https://archives.nseindia.com/content/historical/DERIVATIVES"
        f"/{y}/{m}/fo{d}{m}{y}bhav.csv.zip"
    )


def _fetch_bhav(dt: datetime, session: requests.Session) -> pd.DataFrame | None:
    """Download and parse one day's Bhav copy. Returns None on failure."""
    cache_file = OPT_CACHE / f"bhav_{dt.strftime('%Y%m%d')}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    url = _bhav_url(dt)
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        # Normalise column names
        df.columns = df.columns.str.strip().str.upper()
        df.to_parquet(cache_file)
        return df
    except Exception:
        return None


def _atm_call_price(bhav: pd.DataFrame, spot: float) -> float | None:
    """
    From a single day's Bhav copy, extract the closing price of the
    nearest-expiry ATM call on NIFTY.
    """
    # Filter: NIFTY index calls
    mask = (
        (bhav["SYMBOL"] == "NIFTY") &
        (bhav["INSTRUMENT"].str.strip() == "OPTIDX") &
        (bhav["OPTION_TYP"].str.strip() == "CE")
    )
    calls = bhav[mask].copy()
    if calls.empty:
        return None

    # Parse expiry dates
    calls["EXPIRY_DT"] = pd.to_datetime(
        calls["EXPIRY_DT"].str.strip(), format="%d-%b-%Y", errors="coerce"
    )
    calls = calls.dropna(subset=["EXPIRY_DT"])
    if calls.empty:
        return None

    # Take nearest expiry
    nearest_expiry = calls["EXPIRY_DT"].min()
    calls = calls[calls["EXPIRY_DT"] == nearest_expiry]

    # ATM: strike closest to spot
    calls["STRIKE_PR"] = pd.to_numeric(calls["STRIKE_PR"], errors="coerce")
    calls = calls.dropna(subset=["STRIKE_PR"])
    if calls.empty:
        return None

    atm_strike = calls.iloc[(calls["STRIKE_PR"] - spot).abs().argsort()[:1]]["STRIKE_PR"].values[0]
    row = calls[calls["STRIKE_PR"] == atm_strike].iloc[0]

    close_col = next((c for c in ["CLOSE", "CLOSE_PRICE", "SETTL_PRICE"] if c in calls.columns), None)
    if close_col is None:
        return None

    price = pd.to_numeric(row[close_col], errors="coerce")
    return float(price) if not np.isnan(price) else None


def build_options_dataset(
    spot_df: pd.DataFrame,
    start: str,
    end: str,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    """
    For each trading date in [start, end], download the NSE Bhav copy and
    extract the ATM Nifty call price.

    Parameters
    ----------
    spot_df  : DataFrame with DatetimeIndex and 'Close' column (Nifty spot)
    start    : ISO date string
    end      : ISO date string
    sleep_sec: polite delay between requests (default 0.3s)

    Returns
    -------
    DataFrame with columns: date, spot, atm_strike, market_price
    Only rows where market_price was successfully fetched are included.
    """
    session = requests.Session()
    # Prime NSE session cookie (sometimes required)
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
    except Exception:
        pass

    dates = spot_df.loc[start:end].index
    rows = []
    n = len(dates)

    FAST_FAIL_AFTER = 20   # give up if first N attempts all fail

    for i, dt in enumerate(dates):
        spot = float(spot_df.loc[dt, "Close"])
        dt_plain = pd.Timestamp(dt).to_pydatetime().replace(tzinfo=None)

        bhav = _fetch_bhav(dt_plain, session)
        price = _atm_call_price(bhav, spot) if bhav is not None else None

        if price is not None:
            rows.append({
                "date":         dt,
                "spot":         spot,
                "atm_strike":   round(spot / 50) * 50,
                "market_price": price,
            })

        if (i + 1) % 20 == 0:
            fetched = len(rows)
            print(f"  {i+1}/{n} dates processed  |  {fetched} prices fetched  ({100*(i+1)/n:.0f}%)")
            # Fast-fail: NSE is blocking us
            if i + 1 >= FAST_FAIL_AFTER and fetched == 0:
                print("  NSE is blocking requests — stopping early.")
                break

        time.sleep(sleep_sec)

    df_opts = pd.DataFrame(rows)
    if not df_opts.empty:
        df_opts["date"] = pd.to_datetime(df_opts["date"])
        # Cache the final dataset
        df_opts.to_csv(DATA_DIR / "raw" / "nse_atm_calls.csv", index=False)
        print(f"Saved {len(df_opts)} option prices → data/raw/nse_atm_calls.csv")
    else:
        print("Warning: no option prices fetched. NSE may be blocking requests.")

    return df_opts


def load_options_dataset() -> pd.DataFrame | None:
    """Load cached options dataset if available."""
    p = DATA_DIR / "raw" / "nse_atm_calls.csv"
    if p.exists() and p.stat().st_size > 200:
        df = pd.read_csv(p, parse_dates=["date"])
        print(f"Loaded {len(df)} cached option prices from nse_atm_calls.csv")
        return df
    return None
