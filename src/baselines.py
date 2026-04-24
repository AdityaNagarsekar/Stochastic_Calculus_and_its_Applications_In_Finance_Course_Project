"""
Project 1 — BSM delta-hedging baselines.

Two strategies:
  - BSM daily  : rebalance every step (rebalance_freq=1)
  - BSM weekly : rebalance every 5 steps (rebalance_freq=5)

Returns terminal hedging errors and total transaction costs for
comparison with the RL agent.
"""

import numpy as np
from bsm import bsm_delta, bsm_price, bsm_gamma


def run_bsm_hedge(
    env_params: dict,
    n_episodes: int = 1000,
    rebalance_freq: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate BSM delta-hedging across many GBM paths.

    Parameters
    ----------
    env_params      : dict with keys S0, K, T, r, sigma, n_steps, kappa
    n_episodes      : number of Monte Carlo episodes
    rebalance_freq  : how often (in steps) to rebalance the hedge
    seed            : RNG seed for reproducibility

    Returns
    -------
    terminal_errors : array of shape (n_episodes,) — final PnL of hedged portfolio
    total_costs     : array of shape (n_episodes,) — total transaction costs paid
    """
    rng = np.random.default_rng(seed)

    S0    = env_params['S0']
    K     = env_params['K']
    T     = env_params['T']
    r     = env_params['r']
    sigma = env_params['sigma']
    n     = env_params['n_steps']
    kappa = env_params['kappa']
    dt    = T / n

    terminal_errors = np.empty(n_episodes)
    total_costs     = np.empty(n_episodes)

    for ep in range(n_episodes):
        S       = S0 * rng.uniform(0.90, 1.10)  # same randomisation as env
        delta   = 0.0
        tc_sum  = 0.0

        # Hedge portfolio: start funded at the BSM price of the option
        portfolio = bsm_price(S, K, T, r, sigma)

        for t in range(n):
            tau = max(T - t * dt, 1e-9)

            if t % rebalance_freq == 0:
                new_delta = bsm_delta(S, K, tau, r, sigma)
                tc = kappa * abs(new_delta - delta) * S
                tc_sum += tc
                portfolio -= tc       # transaction costs reduce portfolio value
                delta = new_delta

            # GBM step
            Z     = rng.standard_normal()
            S_new = S * np.exp((r - 0.5 * sigma ** 2) * dt
                               + sigma * np.sqrt(dt) * Z)
            portfolio += delta * (S_new - S)
            S = S_new

        # At expiry: portfolio vs option payoff
        payoff = max(S - K, 0.0)
        terminal_errors[ep] = portfolio - payoff
        total_costs[ep]     = tc_sum

    return terminal_errors, total_costs


def record_baseline_metrics(errors: np.ndarray, costs: np.ndarray,
                             name: str) -> dict:
    """Summarise a set of baseline runs into a metrics dict."""
    return {
        "name":       name,
        "mean_error": float(np.mean(errors)),
        "std_error":  float(np.std(errors)),
        "mae":        float(np.mean(np.abs(errors))),
        "mean_cost":  float(np.mean(costs)),
    }


def run_lstm_bsm_hedge(
    real_prices:  np.ndarray,
    lstm_vols:    np.ndarray,
    S0:   float = 100.0,
    K:    float = 100.0,
    T:    float = 21 / 252,
    r:    float = 0.05,
    kappa: float = 0.001,
    n_steps: int = 21,
    seed:  int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BSM delta hedge using the LSTM-predicted vol as sigma at each step.

    Simulates a GBM path (using the LSTM vol for that step as the
    diffusion coefficient) and hedges with BSM delta computed at the
    same LSTM vol.  This is the critical 'oracle vol' baseline that
    isolates how much better hedging can be when the vol forecast is perfect.

    Parameters
    ----------
    real_prices  : 1-D array of historical closing prices — used only to select
                   which LSTM vol window to use (test-period alignment).
    lstm_vols    : 1-D array of LSTM vol forecasts aligned with real_prices
    S0           : normalised starting spot price (default 100, matching the RL
                   environment).  Raw NIFTY prices must NOT be used here: NIFTY
                   trades at ~15 000 while K=100, making every option deeply
                   ITM, BSM delta ≈ 1, and the first rebalance incurs ~15 × in
                   transaction costs — destroying comparability with other runs.
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # lstm_vols may be a shorter test-set slice; sample only within its bounds.
    # real_prices is indexed by position — we restrict start_i so both arrays
    # have valid n_steps-length windows from the same offset.
    n_lstm = len(lstm_vols)
    n_prices = len(real_prices)
    # Use the tail of real_prices aligned with lstm_vols (test period)
    price_offset = max(0, n_prices - n_lstm)
    aligned_prices = real_prices[price_offset:]
    max_start = min(len(aligned_prices), n_lstm) - n_steps
    if max_start <= 0:
        raise ValueError(
            f"lstm_vols ({n_lstm}) or real_prices tail ({len(aligned_prices)}) "
            f"too short for n_steps={n_steps}"
        )
    n_episodes = min(1000, max_start)

    terminal_errors = np.empty(n_episodes)
    total_costs     = np.empty(n_episodes)

    for ep in range(n_episodes):
        start_i = int(rng.integers(0, max_start))
        # Start at normalised S0 ±10 % to match the RL/BSM env exactly.
        # real_prices[start_i] is only used to select the LSTM vol window.
        S = S0 * rng.uniform(0.90, 1.10)
        sig_path = lstm_vols[start_i: start_i + n_steps]
        sig_path = np.clip(sig_path, 0.03, 1.5)

        portfolio = bsm_price(S, K, T, r, float(sig_path[0]))
        delta     = 0.0
        tc_sum    = 0.0

        for t in range(n_steps):
            tau = max(T - t * dt, 1e-9)
            sig = float(sig_path[t])

            new_delta = bsm_delta(S, K, tau, r, sig)
            tc = kappa * abs(new_delta - delta) * S
            tc_sum    += tc
            portfolio -= tc
            delta      = new_delta

            Z     = rng.standard_normal()
            S_new = S * np.exp((r - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * Z)
            portfolio += delta * (S_new - S)
            S = S_new

        payoff = max(S - K, 0.0)
        terminal_errors[ep] = portfolio - payoff
        total_costs[ep]     = tc_sum

    return terminal_errors, total_costs


def run_bootstrap_hedge(
    real_returns: np.ndarray,
    env_params:   dict,
    n_episodes:   int = 1000,
    seed:         int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    BSM daily hedge evaluated on bootstrapped blocks of real NIFTY returns.

    Each episode samples n_steps consecutive real log-returns (block bootstrap)
    and reconstructs the price path, then hedges with BSM daily using a
    rolling 21-day realised vol estimate as sigma.
    """
    rng  = np.random.default_rng(seed)
    n    = env_params['n_steps']
    K    = env_params['K']
    T    = env_params['T']
    r    = env_params['r']
    kappa = env_params['kappa']
    dt   = T / n
    S0   = env_params['S0']

    n_hist = len(real_returns)
    terminal_errors = np.empty(n_episodes)
    total_costs     = np.empty(n_episodes)

    for ep in range(n_episodes):
        start = rng.integers(21, n_hist - n)   # leave 21 days for vol estimate
        block = real_returns[start: start + n]  # real return block

        # Rolling vol estimate: use preceding 21 days
        sigma = float(np.std(real_returns[start - 21: start]) * np.sqrt(252))
        sigma = max(sigma, 0.03)

        S = S0 * np.exp(float(real_returns[start - 1]))   # start from actual level
        portfolio = bsm_price(S, K, T, r, sigma)
        delta     = 0.0
        tc_sum    = 0.0

        for t in range(n):
            tau = max(T - t * dt, 1e-9)
            new_delta = bsm_delta(S, K, tau, r, sigma)
            tc = kappa * abs(new_delta - delta) * S
            tc_sum    += tc
            portfolio -= tc
            delta      = new_delta

            S_new = S * np.exp(float(block[t]))
            portfolio += delta * (S_new - S)
            S = S_new

        payoff = max(S - K, 0.0)
        terminal_errors[ep] = portfolio - payoff
        total_costs[ep]     = tc_sum

    return terminal_errors, total_costs
