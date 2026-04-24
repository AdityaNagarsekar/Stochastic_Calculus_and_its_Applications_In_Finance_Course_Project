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
# Shared analysis helpers
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[boot_idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(boot_means, alpha / 2)),
        "ci_high": float(np.quantile(boot_means, 1 - alpha / 2)),
    }


def paired_bootstrap_mae_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test for mean absolute error difference a - b."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size != b_arr.size or a_arr.size == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    diff = np.abs(a_arr) - np.abs(b_arr)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_diff = diff[boot_idx].mean(axis=1)
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.quantile(boot_diff, alpha / 2)),
        "ci_high": float(np.quantile(boot_diff, 1 - alpha / 2)),
        "p_value": float(min(1.0, 2 * min(np.mean(boot_diff <= 0.0), np.mean(boot_diff >= 0.0)))),
    }


def tail_risk_metrics(values: np.ndarray, alpha: float = 0.05) -> dict:
    """Tail-risk summary for terminal PnL or errors."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"q05": np.nan, "q50": np.nan, "q95": np.nan, "var95": np.nan, "cvar95": np.nan, "worst1pct": np.nan}
    var95 = float(np.quantile(arr, alpha))
    tail = arr[arr <= var95]
    cvar95 = float(tail.mean()) if tail.size else var95
    worst_n = max(1, int(np.ceil(0.01 * arr.size)))
    worst1pct = float(np.sort(arr)[:worst_n].mean())
    return {
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "var95": var95,
        "cvar95": cvar95,
        "worst1pct": worst1pct,
    }


def utility_scores(mae: float, mean_tc: float, lambdas=(1.0, 2.0, 5.0)) -> dict:
    """Simple cost-error tradeoff score J = MAE + lambda * TC."""
    return {f"J_lambda_{lam:g}": float(mae + lam * mean_tc) for lam in lambdas}


def turnover_from_delta_paths(delta_paths: list[list[float]]) -> dict:
    """Absolute delta turnover summary across episodes."""
    if not delta_paths:
        return {"mean_turnover": np.nan, "std_turnover": np.nan}
    totals = []
    for path in delta_paths:
        arr = np.asarray(path, dtype=float)
        if arr.size <= 1:
            totals.append(0.0)
        else:
            totals.append(float(np.sum(np.abs(np.diff(arr)))))
    totals = np.asarray(totals, dtype=float)
    return {"mean_turnover": float(totals.mean()), "std_turnover": float(totals.std())}


def regime_slice_metrics(
    values: np.ndarray,
    moneyness: np.ndarray | None = None,
    tau_frac: np.ndarray | None = None,
    sigma_proxy: np.ndarray | None = None,
) -> dict:
    """Slice terminal values by simple regime bins when inputs are available."""
    arr = np.asarray(values, dtype=float)
    out = {}
    if arr.size == 0:
        return out

    def _slice(metric_name: str, group: np.ndarray | None, bins, labels):
        if group is None:
            return
        group_arr = np.asarray(group, dtype=float)
        if group_arr.size != arr.size:
            return
        cats = pd.cut(group_arr, bins=bins, labels=labels, include_lowest=True)
        out[metric_name] = {
            str(label): float(np.mean(np.abs(arr[cats == label])))
            for label in labels
            if np.any(cats == label)
        }

    _slice("moneyness", moneyness, bins=[0.0, 0.95, 1.05, np.inf], labels=["OTM", "ATM", "ITM"])
    _slice("tau", tau_frac, bins=[0.0, 0.25, 0.75, 1.0], labels=["near_expiry", "mid", "early"])
    _slice("sigma", sigma_proxy, bins=[0.0, 0.15, 0.25, np.inf], labels=["low", "mid", "high"])
    return out


def analyse_hedging_results(
    terminal_pnl: np.ndarray,
    episode_tc: np.ndarray,
    delta_paths: list[list[float]] | None = None,
    bsm_delta_paths: list[list[float]] | None = None,
    moneyness: np.ndarray | None = None,
    tau_frac: np.ndarray | None = None,
    sigma_proxy: np.ndarray | None = None,
    lambdas=(1.0, 2.0, 5.0),
    seed: int = 42,
) -> dict:
    """Full summary for a hedging experiment."""
    pnl = np.asarray(terminal_pnl, dtype=float)
    tc = np.asarray(episode_tc, dtype=float)
    mae = np.abs(pnl)
    out = {
        "n_episodes": int(pnl.size),
        "mean_error": float(np.mean(pnl)) if pnl.size else np.nan,
        "std_error": float(np.std(pnl)) if pnl.size else np.nan,
        "mae": float(np.mean(mae)) if mae.size else np.nan,
        "mean_cost": float(np.mean(tc)) if tc.size else np.nan,
        "error_ci95": bootstrap_mean_ci(pnl, seed=seed),
        "mae_ci95": bootstrap_mean_ci(mae, seed=seed),
        "tail_risk": tail_risk_metrics(pnl),
        "utility": utility_scores(float(np.mean(mae)) if mae.size else np.nan,
                                  float(np.mean(tc)) if tc.size else np.nan,
                                  lambdas=lambdas),
        "turnover": turnover_from_delta_paths(delta_paths or []),
        "regimes": regime_slice_metrics(pnl, moneyness=moneyness, tau_frac=tau_frac, sigma_proxy=sigma_proxy),
    }
    if delta_paths is not None and bsm_delta_paths is not None:
        rl = np.concatenate([np.asarray(p, dtype=float) for p in delta_paths if len(p)]) if delta_paths else np.array([])
        bsm = np.concatenate([np.asarray(p, dtype=float) for p in bsm_delta_paths if len(p)]) if bsm_delta_paths else np.array([])
        if rl.size and rl.size == bsm.size:
            out["delta_corr"] = float(np.corrcoef(rl, bsm)[0, 1])
            out["delta_rmse"] = float(np.sqrt(np.mean((rl - bsm) ** 2)))
        else:
            out["delta_corr"] = np.nan
            out["delta_rmse"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Project 1 — RL evaluation
# ---------------------------------------------------------------------------

def evaluate_rl_agent(
    model,                  # stable_baselines3 model
    env_cls,                # HedgingEnv class
    env_params: dict,
    n_episodes: int = 1000,
    deterministic: bool = True,
    sigma_for_bsm: float | None = None,  # if None, use env_params["sigma"]
    seed: int | None = None,
) -> dict:
    """
    Roll out the RL model on fresh env episodes.

    Returns a dict with:
      terminal_pnl    : (n_episodes,) final hedge PnL
      episode_tc      : (n_episodes,) total transaction costs
      delta_paths     : list of RL delta sequences per episode
      bsm_delta_paths : list of BSM delta sequences at the same states
    """
    from bsm import bsm_delta as _bsm_delta

    K = env_params.get("K", 100.0)
    T = env_params.get("T", 21 / 252)
    r = env_params.get("r", 0.05)
    sigma = sigma_for_bsm if sigma_for_bsm is not None \
            else env_params.get("sigma", 0.20)

    env = env_cls(**env_params)
    terminal_pnl    = []
    episode_tc      = []
    delta_paths     = []
    bsm_delta_paths = []
    obs_paths       = []

    for episode_idx in range(n_episodes):
        episode_seed = None if seed is None else seed + episode_idx
        obs, _   = env.reset(seed=episode_seed)
        done     = False
        deltas   = []
        bsm_dels = []
        obs_seq  = []

        while not done:
            obs_seq.append(np.asarray(obs, dtype=float).copy())
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
        obs_paths.append(obs_seq)

    return {
        "terminal_pnl":    np.array(terminal_pnl),
        "episode_tc":      np.array(episode_tc),
        "delta_paths":     delta_paths,
        "bsm_delta_paths": bsm_delta_paths,
        "obs_paths":       obs_paths,
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
