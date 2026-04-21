"""
Black-Scholes-Merton and CRR Binomial pricing utilities.
Used by both Project 1 (RL hedging) and Project 3 (LSTM vol + pricing).
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Core BSM formulas
# ---------------------------------------------------------------------------

def bsm_price(S, K, T, r, sigma, option='call'):
    """Black-Scholes-Merton option price."""
    if T <= 0:
        if option == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_delta(S, K, T, r, sigma, option='call'):
    """First derivative of BSM price w.r.t. S."""
    if T <= 0:
        if option == 'call':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option == 'call' else norm.cdf(d1) - 1.0


def bsm_gamma(S, K, T, r, sigma):
    """Second derivative of BSM price w.r.t. S (same for call and put)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bsm_vega(S, K, T, r, sigma):
    """Sensitivity of BSM price to sigma (same for call and put)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def bsm_theta(S, K, T, r, sigma, option='call'):
    """Time decay of BSM price (per year)."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option == 'call':
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_vol(market_price, S, K, T, r, option='call',
                lo=1e-6, hi=10.0):
    """
    Recover implied vol via Brent's method.
    Returns np.nan if the market price is outside the no-arbitrage bounds.
    """
    try:
        intrinsic = max(S - K, 0) if option == 'call' else max(K - S, 0)
        if market_price <= intrinsic:
            return np.nan
        f = lambda s: bsm_price(S, K, T, r, s, option) - market_price
        return brentq(f, lo, hi, xtol=1e-8)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# CRR Binomial model
# ---------------------------------------------------------------------------

def crr_binomial(S, K, T, r, sigma, N=100, option='call',
                 style='european'):
    """
    Cox-Ross-Rubinstein binomial tree.
    style: 'european' or 'american'
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Terminal stock prices (vectorised)
    j = np.arange(N + 1)
    ST = S * (u ** (N - j)) * (d ** j)

    if option == 'call':
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    disc = np.exp(-r * dt)
    for _ in range(N):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if style == 'american':
            j_inner = np.arange(len(V))
            S_inner = S * (u ** (len(V) - 1 - j_inner)) * (d ** j_inner)
            intrinsic = (np.maximum(S_inner - K, 0.0) if option == 'call'
                         else np.maximum(K - S_inner, 0.0))
            V = np.maximum(V, intrinsic)

    return float(V[0])
