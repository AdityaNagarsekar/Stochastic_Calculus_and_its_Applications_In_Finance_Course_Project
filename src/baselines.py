"""
Project 1 — BSM delta-hedging baselines.

Two strategies:
  - BSM daily  : rebalance every step (rebalance_freq=1)
  - BSM weekly : rebalance every 5 steps (rebalance_freq=5)

Returns terminal hedging errors and total transaction costs for
comparison with the RL agent.
"""

import numpy as np
from bsm import bsm_delta, bsm_price


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
