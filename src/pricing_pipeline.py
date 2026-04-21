"""
Project 3 — Pricing pipeline.

Prices options under three vol sources (historical, LSTM, implied) and
runs a horse-race backtest comparing pricing accuracy across methods.
"""

import numpy as np
import pandas as pd
from bsm import bsm_price, bsm_vega, crr_binomial, implied_vol


# ---------------------------------------------------------------------------
# Single-option pricing under multiple vol sources
# ---------------------------------------------------------------------------

def price_option_three_ways(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma_hist: float,
    sigma_lstm: float,
    sigma_impl: float | None = None,
    option: str = "call",
    N: int = 100,
) -> dict:
    """
    Price one option under historical, LSTM, and implied vol.

    Returns a nested dict keyed by method name, each containing
    'bsm_price', 'binomial_price', and 'sigma'.
    """
    sources = {"historical": sigma_hist, "lstm": sigma_lstm}
    if sigma_impl is not None and not np.isnan(sigma_impl):
        sources["implied"] = sigma_impl

    results = {}
    for name, sig in sources.items():
        results[name] = {
            "bsm_price":      bsm_price(S, K, T, r, sig, option),
            "binomial_price": crr_binomial(S, K, T, r, sig, N, option),
            "sigma":          sig,
        }
    return results


# ---------------------------------------------------------------------------
# Vega sensitivity analysis
# ---------------------------------------------------------------------------

def vega_sensitivity_analysis(
    S: float,
    K: float,
    r: float,
    sigma_range: np.ndarray,
    maturities: list[float] | None = None,
) -> pd.DataFrame:
    """
    For each (T, sigma) combination, compute:
      - BSM price
      - Vega  (∂V/∂σ)
      - Price error from a 1 percentage-point vol error (Vega × 0.01)

    This quantifies how much an LSTM forecasting error translates to a
    pricing error — the core theoretical link in the project.
    """
    if maturities is None:
        maturities = [7 / 252, 21 / 252, 63 / 252]  # 1w, 1m, 3m

    rows = []
    for T in maturities:
        for sig in sigma_range:
            p = bsm_price(S, K, T, r, sig, "call")
            v = bsm_vega(S, K, T, r, sig)
            rows.append({
                "T_days":          round(T * 252),
                "sigma":           sig,
                "bsm_price":       p,
                "vega":            v,
                "price_err_1pct":  v * 0.01,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Horse-race backtest
# ---------------------------------------------------------------------------

def backtest_pricing(
    prices:       np.ndarray,   # daily spot prices (test period)
    dates:        pd.DatetimeIndex,
    hist_vols:    np.ndarray,   # historical rolling vol at each date
    lstm_vols:    np.ndarray,   # LSTM-forecast vol at each date
    market_prices: np.ndarray,  # observed option prices (ATM call, T≈21/252)
    K_offsets:    np.ndarray | None = None,  # ATM strike = round(S/100)*100 if None
    T:            float = 21 / 252,
    r:            float = 0.065,
    option:       str = "call",
    garch_vols:   np.ndarray | None = None,  # optional GARCH(1,1) forecast vol
) -> pd.DataFrame:
    """
    For each date in the test set, price an ATM call under three vol
    sources and compare to the observed market price.

    Returns a DataFrame with columns:
      date, method, sigma, model_price, market_price, error, pct_error
    """
    rows = []
    n    = len(prices)

    for i in range(n):
        S     = float(prices[i])
        K     = float(K_offsets[i]) if K_offsets is not None else round(S / 100) * 100
        mkt   = float(market_prices[i])

        sig_hist  = float(hist_vols[i])
        sig_lstm  = float(lstm_vols[i])
        sig_impl  = implied_vol(mkt, S, K, T, r, option)
        sig_garch = float(garch_vols[i]) if garch_vols is not None else None

        candidates = [("historical", sig_hist),
                      ("lstm",       sig_lstm),
                      ("implied",    sig_impl)]
        if sig_garch is not None:
            candidates.append(("garch", sig_garch))

        for method, sig in candidates:
            if np.isnan(sig) or sig <= 0:
                continue
            model_p = bsm_price(S, K, T, r, sig, option)
            rows.append({
                "date":         dates[i],
                "method":       method,
                "sigma":        sig,
                "model_price":  model_p,
                "market_price": mkt,
                "error":        model_p - mkt,
                "abs_error":    abs(model_p - mkt),
                "pct_error":    (model_p - mkt) / mkt if mkt > 0 else np.nan,
            })

    df = pd.DataFrame(rows)
    return df


def estimate_vrp(
    prices:        np.ndarray,
    dates,
    lstm_vols:     np.ndarray,
    market_prices: np.ndarray,
    K_offsets:     np.ndarray | None = None,
    T:             float = 21 / 252,
    r:             float = 0.065,
) -> float:
    """
    Estimate the Volatility Risk Premium (VRP) from a held-out dataset.

    VRP = E[sigma_implied − sigma_lstm]

    Under Girsanov's theorem the physical-measure vol (what LSTM forecasts)
    and the risk-neutral vol (what the market prices) are related by the
    market price of volatility risk.  This function estimates that premium
    empirically so we can adjust LSTM test-set predictions toward the
    risk-neutral measure.

    Returns the scalar additive correction: sigma_Q ≈ sigma_P + VRP
    """
    diffs = []
    for i in range(len(prices)):
        S   = float(prices[i])
        K   = float(K_offsets[i]) if K_offsets is not None else round(S / 100) * 100
        mkt = float(market_prices[i])
        sig_impl = implied_vol(mkt, S, K, T, r)
        sig_lstm = float(lstm_vols[i])
        if np.isnan(sig_impl) or sig_impl <= 0 or sig_lstm <= 0:
            continue
        diffs.append(sig_impl - sig_lstm)
    return float(np.mean(diffs)) if diffs else 0.0


def summarise_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate horse-race results by method."""
    return (
        df.groupby("method")
        .agg(
            mean_error=("error",     "mean"),
            std_error= ("error",     "std"),
            mae=       ("abs_error", "mean"),
            mean_pct=  ("pct_error", "mean"),
        )
        .round(6)
    )
