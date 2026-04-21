"""
Shared backtesting utilities used by both projects.

Provides:
  - evaluate_rl_agent   : run a trained SB3 model on the hedging env
  - plot_pnl_comparison : overlapping terminal PnL histograms
  - plot_hedge_ratios   : average delta path over episodes
  - plot_vol_forecast   : LSTM predicted vs realised vol
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Project 1 — RL evaluation
# ---------------------------------------------------------------------------

def evaluate_rl_agent(
    model,                  # stable_baselines3 model
    env_cls,                # HedgingEnv class
    env_params: dict,
    n_episodes: int = 1000,
    deterministic: bool = True,
) -> dict:
    """
    Roll out the RL model on fresh env episodes.

    Returns a dict with:
      terminal_pnl    : (n_episodes,) final hedge PnL
      episode_tc      : (n_episodes,) total transaction costs
      delta_paths     : list of RL delta sequences per episode
      bsm_delta_paths : list of BSM delta sequences at the same states
    """
    from env import HedgingEnv
    from bsm import bsm_delta as _bsm_delta

    K = env_params.get("K", 100.0)
    T = env_params.get("T", 21 / 252)
    r = env_params.get("r", 0.05)
    sigma = env_params.get("sigma", 0.20)

    env = HedgingEnv(**env_params)
    terminal_pnl    = []
    episode_tc      = []
    delta_paths     = []
    bsm_delta_paths = []

    for _ in range(n_episodes):
        obs, _   = env.reset()
        done     = False
        deltas   = []
        bsm_dels = []

        while not done:
            # Decode state: obs = [S/K, tau/T, delta_prev]
            S_t   = float(obs[0]) * K
            tau_t = max(float(obs[1]) * T, 1e-9)
            bsm_dels.append(_bsm_delta(S_t, K, tau_t, r, sigma))

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            deltas.append(float(action[0]))

        terminal_pnl.append(info.get("terminal_pnl", np.nan))
        episode_tc.append(info.get("episode_tc", np.nan))
        delta_paths.append(deltas)
        bsm_delta_paths.append(bsm_dels)

    return {
        "terminal_pnl":    np.array(terminal_pnl),
        "episode_tc":      np.array(episode_tc),
        "delta_paths":     delta_paths,
        "bsm_delta_paths": bsm_delta_paths,
    }


# ---------------------------------------------------------------------------
# Project 1 — Plots
# ---------------------------------------------------------------------------

def plot_pnl_comparison(results: dict, title: str = "Terminal PnL distribution",
                         save: bool = True):
    """
    Overlapping histogram of terminal PnL for multiple hedging strategies.
    results: dict mapping label → array of terminal PnL values
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, vals in results.items():
        ax.hist(vals, bins=60, alpha=0.55, label=label, density=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Terminal hedging PnL")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "pnl_comparison.png", dpi=150)
    plt.show()


def plot_hedge_ratios(bsm_path: np.ndarray, rl_path: np.ndarray,
                      save: bool = True):
    """
    Average delta path over episodes: RL agent vs BSM delta.
    bsm_path, rl_path: shape (n_steps,) averaged over episodes
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(bsm_path, label="BSM delta", lw=2)
    ax.plot(rl_path,  label="RL agent",  lw=2, ls="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Hedge ratio (delta)")
    ax.set_title("Mean hedge ratio over an episode")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "hedge_ratios.png", dpi=150)
    plt.show()


def plot_cost_comparison(cost_dict: dict, save: bool = True):
    """Bar chart of mean transaction cost per episode across methods."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(cost_dict.keys()), list(cost_dict.values()), width=0.5)
    ax.set_ylabel("Mean transaction cost per episode")
    ax.set_title("Transaction cost comparison")
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "cost_comparison.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Project 3 — Plots
# ---------------------------------------------------------------------------

def plot_vol_forecast(
    dates_test: pd.DatetimeIndex,
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    title: str = "LSTM Vol Forecast vs Realised Vol",
    save: bool = True,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates_test, y_true, label="Realised vol",  lw=1.5)
    ax.plot(dates_test, y_pred, label="LSTM forecast", lw=1.5, ls="--")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "vol_forecast.png", dpi=150)
    plt.show()


def plot_vega_surface(df_vega: pd.DataFrame, save: bool = True):
    """
    Plot Vega curves (∂V/∂σ) vs sigma for each maturity group.
    df_vega must come from pricing_pipeline.vega_sensitivity_analysis.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for T_days, grp in df_vega.groupby("T_days"):
        ax.plot(grp["sigma"], grp["vega"],
                label=f"T = {T_days}d", lw=2)
    ax.set_xlabel("Implied / forecast sigma")
    ax.set_ylabel("Vega (∂V/∂σ)")
    ax.set_title("Vega vs sigma by maturity")
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "vega_surface.png", dpi=150)
    plt.show()


def plot_pricing_error(df_backtest: pd.DataFrame, save: bool = True):
    """Bar chart of MAE across pricing methods."""
    summary = df_backtest.groupby("method")["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    summary.plot.bar(ax=ax, width=0.5)
    ax.set_ylabel("Mean absolute pricing error")
    ax.set_title("Pricing horse race — MAE by vol source")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "pricing_error_bar.png", dpi=150)
    plt.show()
