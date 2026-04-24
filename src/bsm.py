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


# ---------------------------------------------------------------------------
# Heston exact pricing  (semi-analytic, Heston 1993)
# ---------------------------------------------------------------------------

def heston_price(
    S: float, K: float, tau: float, r: float,
    V0: float, kappa: float, theta: float, xi: float, rho: float,
    option: str = 'call',
) -> float:
    """
    Exact Heston (1993) European call/put price via numerical integration
    of the characteristic function.

    Slow (~2 ms/call).  Use for evaluation metrics only, not inside training
    loops.  Falls back to BSM(sqrt(V0)) on any numerical failure.
    """
    from scipy.integrate import quad

    if tau < 1e-7 or S <= 0:
        return bsm_price(S, K, max(tau, 0), r, max(np.sqrt(V0), 1e-4), option)

    def _char(phi, j):
        i   = 1j
        u_j = 0.5 if j == 1 else -0.5
        b_j = kappa - rho * xi if j == 1 else kappa
        d   = np.sqrt((rho * xi * i * phi - b_j) ** 2
                      - xi ** 2 * (2 * u_j * i * phi - phi ** 2))
        g   = (b_j - rho * xi * i * phi + d) / (b_j - rho * xi * i * phi - d)
        try:
            exp_d = np.exp(d * tau)
            denom_log = 1 - g * exp_d
            numer_log = 1 - g
            if abs(denom_log) < 1e-12 or abs(numer_log) < 1e-12:
                raise ValueError
            C = (r * i * phi * tau
                 + kappa * theta / xi ** 2
                 * ((b_j - rho * xi * i * phi + d) * tau
                    - 2 * np.log(denom_log / numer_log)))
            D = ((b_j - rho * xi * i * phi + d) / xi ** 2
                 * (1 - exp_d) / (1 - g * exp_d))
        except Exception:
            return 0.0
        return np.exp(C + D * V0 + i * phi * np.log(S))

    def _integrand(phi, j):
        f = _char(phi, j)
        return float(np.real(np.exp(-1j * phi * np.log(K)) * f / (1j * phi)))

    try:
        P1 = 0.5 + quad(lambda p: _integrand(p, 1), 1e-6, 200,
                        limit=200, epsabs=1e-4)[0] / np.pi
        P2 = 0.5 + quad(lambda p: _integrand(p, 2), 1e-6, 200,
                        limit=200, epsabs=1e-4)[0] / np.pi
        call = float(S * P1 - K * np.exp(-r * tau) * P2)
    except Exception:
        return bsm_price(S, K, tau, r, max(np.sqrt(V0), 1e-4), option)

    if option == 'call':
        return max(call, 0.0)
    return max(call - S + K * np.exp(-r * tau), 0.0)


def heston_delta(
    S: float, K: float, tau: float, r: float,
    V0: float, kappa: float, theta: float, xi: float, rho: float,
    eps: float = 0.5,
) -> float:
    """Heston delta via central finite difference on heston_price."""
    if tau < 1e-7:
        return 1.0 if S > K else 0.0
    p_up   = heston_price(S + eps, K, tau, r, V0, kappa, theta, xi, rho)
    p_down = heston_price(S - eps, K, tau, r, V0, kappa, theta, xi, rho)
    return float((p_up - p_down) / (2 * eps))
