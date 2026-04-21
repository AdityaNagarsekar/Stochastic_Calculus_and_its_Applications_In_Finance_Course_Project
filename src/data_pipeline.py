"""
Project 3 — Data pipeline.

Downloads Nifty 50 daily closes from Yahoo Finance (^NSEI, 2010-2024),
computes log-returns and rolling realised vols, and builds
fixed-length sequences for LSTM training.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Download helpers (multiple fallback methods for ^NSEI reliability)
# ---------------------------------------------------------------------------

def _flatten(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that newer yfinance versions return."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _download_nifty(start: str, end: str) -> pd.DataFrame:
    """
    Try several methods/tickers to get Nifty 50 daily Close data.
    Returns a DataFrame with at least a 'Close' column and DatetimeIndex.
    Raises RuntimeError with clear instructions if all methods fail.
    """
    attempts = [
        # (method_label, callable)
        ("yf.Ticker('^NSEI').history",
         lambda: _flatten(yf.Ticker("^NSEI").history(start=start, end=end, auto_adjust=True))),
        ("yf.download('^NSEI')",
         lambda: _flatten(yf.download("^NSEI", start=start, end=end,
                                       auto_adjust=True, progress=False))),
        ("yf.download('NIFTY50.NS')",
         lambda: _flatten(yf.download("NIFTY50.NS", start=start, end=end,
                                       auto_adjust=True, progress=False))),
        ("yf.Ticker('NIFTY50.NS').history",
         lambda: _flatten(yf.Ticker("NIFTY50.NS").history(start=start, end=end,
                                                            auto_adjust=True))),
    ]

    last_error = None
    for label, fn in attempts:
        try:
            raw = fn()
            if raw is not None and not raw.empty and "Close" in raw.columns:
                print(f"  Downloaded via {label}: {len(raw)} rows")
                return raw
            print(f"  {label}: returned empty — trying next")
        except Exception as e:
            print(f"  {label}: {e} — trying next")
            last_error = e

    raise RuntimeError(
        "All yfinance download attempts for Nifty 50 failed.\n"
        "Steps to fix:\n"
        "  1. Make sure you are in the scaf env:  conda activate scaf\n"
        "  2. Reinstall/upgrade yfinance:         pip install -U yfinance\n"
        "  3. Check your internet connection.\n"
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Download & feature engineering
# ---------------------------------------------------------------------------

def load_nifty(
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load Nifty 50 prices, compute features and the 21-day forward vol target.

    Columns returned
    ----------------
    Close, log_ret, ret_sq,
    rv5  : 5-day  rolling annualised vol
    rv21 : 21-day rolling annualised vol
    target_vol : forward 21-day realised vol (what we're forecasting)
    """
    cache_path = DATA_DIR / "raw" / "nifty_raw.csv"

    if cache and cache_path.exists() and cache_path.stat().st_size > 500:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    else:
        raw = _download_nifty(start, end)
        df = raw[["Close"]].dropna().copy()
        # Strip timezone: tz_convert(None) works whether or not index is tz-aware
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index.name = "Date"
        df.to_csv(cache_path)

    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ret_sq"]  = df["log_ret"] ** 2

    df["rv5"]  = df["log_ret"].rolling(5).std()  * np.sqrt(252)
    df["rv21"] = df["log_ret"].rolling(21).std() * np.sqrt(252)

    # Vol term structure: ratio of short-term to long-term realised vol.
    # > 1 means vol is spiking (short > long); < 1 means vol is subsiding.
    # Strong predictor of mean-reversion in volatility.
    df["rv_ratio"] = np.clip(df["rv5"] / df["rv21"].replace(0, np.nan), 0.1, 10.0)

    # 5-day price momentum: signed cumulative return over the past week.
    # Provides direction signal independent of vol magnitude.
    df["mom5"] = df["log_ret"].rolling(5).sum()

    # Target: forward-looking 21-day realised vol
    # shift(-21) so that on day t, the label is the vol over [t+1, t+21]
    df["target_vol"] = (
        df["log_ret"].shift(-1).rolling(21).std() * np.sqrt(252)
    )

    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def make_sequences(
    df: pd.DataFrame,
    seq_len: int = 60,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build overlapping windows for LSTM input.

    Returns
    -------
    X     : (n_samples, seq_len, n_features)  float32
    y     : (n_samples,)                      float32  — target_vol
    dates : DatetimeIndex of length n_samples (date of the *last* row in each window)
    """
    if feature_cols is None:
        feature_cols = ["log_ret", "ret_sq", "rv5", "rv21", "rv_ratio", "mom5"]

    arr = df[feature_cols].values.astype(np.float32)
    tgt = df["target_vol"].values.astype(np.float32)
    idx = df.index

    n = len(arr) - seq_len
    X     = np.stack([arr[i : i + seq_len] for i in range(n)])
    y     = tgt[seq_len:]
    dates = idx[seq_len:]

    return X, y, dates


# ---------------------------------------------------------------------------
# Normalisation  (fit on train only, apply to all splits)
# ---------------------------------------------------------------------------

def normalize_splits(splits: dict) -> tuple[dict, StandardScaler]:
    """
    StandardScale features (X) using statistics from the training split only.
    Target (y) is left in original annualised-vol units for interpretability.

    Returns updated splits dict and the fitted scaler (needed at inference time).
    """
    n_train, seq_len, n_feat = splits["train"]["X"].shape

    # Flatten to (n_samples * seq_len, n_features), fit, reshape back
    scaler = StandardScaler()
    train_flat = splits["train"]["X"].reshape(-1, n_feat)
    scaler.fit(train_flat)

    out = {}
    for key, data in splits.items():
        flat = data["X"].reshape(-1, n_feat)
        scaled = scaler.transform(flat).astype(np.float32)
        out[key] = {**data, "X": scaled.reshape(data["X"].shape)}

    return out, scaler


# ---------------------------------------------------------------------------
# Train / val / test split (70 / 15 / 15 chronological)
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
):
    n    = len(X)
    i1   = int(n * train_frac)
    i2   = int(n * (train_frac + val_frac))

    splits = {}
    for key, sl in [("train", slice(None, i1)),
                    ("val",   slice(i1, i2)),
                    ("test",  slice(i2, None))]:
        splits[key] = {"X": X[sl], "y": y[sl], "dates": dates[sl]}

    return splits


# ---------------------------------------------------------------------------
# Convenience: save processed data
# ---------------------------------------------------------------------------

def save_processed(splits: dict, scaler: StandardScaler | None = None):
    proc = DATA_DIR / "processed"
    proc.mkdir(exist_ok=True)
    for split, data in splits.items():
        np.save(proc / f"X_{split}.npy", data["X"])
        np.save(proc / f"y_{split}.npy", data["y"])
        pd.Series(data["dates"]).to_csv(
            proc / f"dates_{split}.csv", index=False
        )
    if scaler is not None:
        with open(proc / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    print(f"Saved to {proc}")


def load_processed():
    proc = DATA_DIR / "processed"
    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = {
            "X":     np.load(proc / f"X_{split}.npy"),
            "y":     np.load(proc / f"y_{split}.npy"),
            "dates": pd.read_csv(proc / f"dates_{split}.csv",
                                 parse_dates=[0]).iloc[:, 0],
        }
    scaler = None
    scaler_path = proc / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return splits, scaler
