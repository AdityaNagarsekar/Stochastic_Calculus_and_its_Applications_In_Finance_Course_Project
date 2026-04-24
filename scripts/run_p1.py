"""
Project 1 — RL Delta Hedging
Run end-to-end: env check → BSM baselines → PPO train → evaluate → plots

Usage:
    python scripts/run_p1.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from env import HedgingEnv
from baselines import run_bsm_hedge, record_baseline_metrics
from backtest import (
    evaluate_rl_agent,
    analyse_hedging_results,
    paired_bootstrap_mae_diff,
)
from bsm import bsm_delta

RESULTS = Path(__file__).parent.parent / "results"
MODELS  = Path(__file__).parent.parent / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

ENV_PARAMS = dict(S0=100, K=100, T=21/252, r=0.05,
                  sigma=0.20, n_steps=21, kappa=0.001)
N_EPISODES = 1000


# ==========================================================================
# Smoke test
# ==========================================================================
print("=" * 60)
print("SMOKE TEST")
print("=" * 60)

env = HedgingEnv(**ENV_PARAMS)
obs, _ = env.reset(seed=0)
for _ in range(21):
    obs, rew, done, _, _ = env.step(env.action_space.sample())
assert obs.shape == (3,), f"Unexpected obs shape: {obs.shape}"
print(f"Passed. Final obs: {obs}")
print(f"Obs space : {env.observation_space}")
print(f"Act space : {env.action_space}")


# ==========================================================================
# Phase 2 — BSM baselines
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 2 — BSM baselines")
print("=" * 60)

daily_errors,  daily_costs  = run_bsm_hedge(ENV_PARAMS, n_episodes=N_EPISODES, rebalance_freq=1)
weekly_errors, weekly_costs = run_bsm_hedge(ENV_PARAMS, n_episodes=N_EPISODES, rebalance_freq=5)

bsm_daily_metrics  = record_baseline_metrics(daily_errors,  daily_costs,  "BSM daily")
bsm_weekly_metrics = record_baseline_metrics(weekly_errors, weekly_costs, "BSM weekly")

print(json.dumps(bsm_daily_metrics,  indent=2))
print(json.dumps(bsm_weekly_metrics, indent=2))


# ==========================================================================
# Phase 3 — Train PPO agent
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 3 — PPO training  (500k timesteps)")
print("=" * 60)

vec_env  = make_vec_env(lambda: HedgingEnv(**ENV_PARAMS), n_envs=4)
eval_env = HedgingEnv(**ENV_PARAMS)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=str(MODELS),
    log_path=str(RESULTS),
    eval_freq=10_000,
    n_eval_episodes=200,
    deterministic=True,
    verbose=0,
)

model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=1e-4,        # lower LR — 3e-4 caused value_loss spikes
    n_steps=512,               # larger rollout buffer per update
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[128, 64]),  # wider first layer
    verbose=1,
)
model.learn(total_timesteps=500_000, callback=eval_cb)
model.save(str(MODELS / "ppo_hedge_v1"))
print("Saved models/ppo_hedge_v1")


# ==========================================================================
# Phase 3b — Hyperparameter sweep
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 3b — Hyperparameter sweep")
print("=" * 60)

sweep_configs = [
    dict(kappa=0.000, net_arch=[64, 64],  lr=3e-4, name="no_tc"),
    dict(kappa=0.001, net_arch=[64, 64],  lr=3e-4, name="base"),
    dict(kappa=0.001, net_arch=[128, 64], lr=3e-4, name="wider"),
    dict(kappa=0.001, net_arch=[64, 64],  lr=1e-4, name="slow_lr"),
]

for cfg in sweep_configs:
    params = {**ENV_PARAMS, "kappa": cfg["kappa"]}
    venv = make_vec_env(lambda: HedgingEnv(**params), n_envs=4)
    m = PPO(
        "MlpPolicy", venv,
        learning_rate=cfg["lr"],
        n_steps=256, batch_size=64, n_epochs=10,
        policy_kwargs=dict(net_arch=cfg["net_arch"]),
        verbose=0,
    )
    m.learn(total_timesteps=200_000)
    m.save(str(MODELS / f"ppo_{cfg['name']}"))
    print(f"Done: {cfg['name']}")


# ==========================================================================
# Phase 4 — Evaluate RL agent vs baselines
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 4 — Evaluation")
print("=" * 60)

model = PPO.load(str(MODELS / "ppo_hedge_v1"))
rl_results = evaluate_rl_agent(model, HedgingEnv, ENV_PARAMS, n_episodes=N_EPISODES)

rl_pnl   = rl_results["terminal_pnl"]
rl_costs = rl_results["episode_tc"]

rl_metrics = {
    "name":       "RL (PPO)",
    "mean_error": float(np.mean(rl_pnl)),
    "std_error":  float(np.std(rl_pnl)),
    "mae":        float(np.mean(np.abs(rl_pnl))),
    "mean_cost":  float(np.mean(rl_costs)),
}


def _episode_state_means(obs_paths: list[list[np.ndarray]]) -> dict:
    moneyness = []
    tau_frac = []
    sigma = []
    for ep in obs_paths:
        if not ep:
            continue
        arr = np.asarray(ep, dtype=float)
        moneyness.append(float(np.mean(arr[:, 0])))
        tau_frac.append(float(np.mean(arr[:, 1])))
        if arr.shape[1] > 3:
            sigma.append(float(np.mean(arr[:, 3])))
    return {
        "moneyness": np.asarray(moneyness, dtype=float) if moneyness else None,
        "tau_frac": np.asarray(tau_frac, dtype=float) if tau_frac else None,
        "sigma_proxy": np.asarray(sigma, dtype=float) if sigma else None,
    }


def _flatten_episode_traces(values: np.ndarray, obs_paths: list[list[np.ndarray]]) -> dict:
    flat_values = []
    flat_moneyness = []
    flat_tau_frac = []
    flat_sigma = []
    for value, ep in zip(values, obs_paths):
        if not ep:
            continue
        arr = np.asarray(ep, dtype=float)
        steps = arr.shape[0]
        flat_values.append(np.repeat(float(value), steps))
        flat_moneyness.append(arr[:, 0])
        flat_tau_frac.append(arr[:, 1])
        if arr.shape[1] > 3:
            flat_sigma.append(arr[:, 3])
    return {
        "values": np.concatenate(flat_values) if flat_values else np.array([]),
        "moneyness": np.concatenate(flat_moneyness) if flat_moneyness else None,
        "tau_frac": np.concatenate(flat_tau_frac) if flat_tau_frac else None,
        "sigma": np.concatenate(flat_sigma) if flat_sigma else None,
    }


def _seed_robustness(model, env_cls, params: dict, seeds=(0, 17, 42), episodes: int = 200) -> dict:
    rows = []
    for s in seeds:
        res = evaluate_rl_agent(model, env_cls, params, n_episodes=episodes, seed=s)
        rows.append({
            "seed": s,
            "mae": float(np.mean(np.abs(res["terminal_pnl"]))),
            "mean_cost": float(np.mean(res["episode_tc"])),
        })
    maes = np.array([r["mae"] for r in rows], dtype=float)
    costs = np.array([r["mean_cost"] for r in rows], dtype=float)
    return {
        "seeds": rows,
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std()),
        "cost_mean": float(costs.mean()),
        "cost_std": float(costs.std()),
    }


analysis_daily = analyse_hedging_results(daily_errors, daily_costs)
analysis_weekly = analyse_hedging_results(weekly_errors, weekly_costs)
analysis_rl = analyse_hedging_results(
    rl_pnl,
    rl_costs,
    delta_paths=rl_results["delta_paths"],
    bsm_delta_paths=rl_results["bsm_delta_paths"],
    **_episode_state_means(rl_results["obs_paths"]),
)

rl_step_traces = _flatten_episode_traces(rl_pnl, rl_results["obs_paths"])
analysis_rl["regimes_stepwise"] = analyse_hedging_results(
    rl_step_traces["values"],
    np.repeat(rl_costs, [len(ep) for ep in rl_results["obs_paths"]]) if rl_results["obs_paths"] else np.array([]),
    moneyness=rl_step_traces["moneyness"],
    tau_frac=rl_step_traces["tau_frac"],
    sigma_proxy=rl_step_traces["sigma"],
)["regimes"]

analysis_bsm_vs_rl = paired_bootstrap_mae_diff(daily_errors, rl_pnl)
analysis_weekly_vs_daily = paired_bootstrap_mae_diff(daily_errors, weekly_errors)
seed_robustness_rl = _seed_robustness(model, HedgingEnv, ENV_PARAMS)

print("\nResults summary:")
for m in [bsm_daily_metrics, bsm_weekly_metrics, rl_metrics]:
    print(f"  {m['name']:12s}  MAE={m['mae']:.4f}  mean={m['mean_error']:+.4f}  "
          f"std={m['std_error']:.4f}  cost={m['mean_cost']:.4f}")

all_metrics = [bsm_daily_metrics, bsm_weekly_metrics, rl_metrics]
with open(RESULTS / "p1_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print("Saved p1_metrics.json")

extended_p1 = {
    "baselines": {
        "bsm_daily": analysis_daily,
        "bsm_weekly": analysis_weekly,
        "rl_ppo": analysis_rl,
    },
    "paired_bootstrap": {
        "daily_vs_rl": analysis_bsm_vs_rl,
        "daily_vs_weekly": analysis_weekly_vs_daily,
    },
    "seed_robustness": {
        "rl_ppo": seed_robustness_rl,
    },
}


# ==========================================================================
# Phase 5 — Additional baselines
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 5 — Additional baselines (LSTM-BSM delta + bootstrap)")
print("=" * 60)

from baselines import run_lstm_bsm_hedge, run_bootstrap_hedge
from data_pipeline import load_nifty, load_processed

# Load LSTM test predictions and NIFTY data
lstm_preds_path = RESULTS.parent / "data" / "processed" / "lstm_test_preds.npy"
if lstm_preds_path.exists():
    lstm_vols = np.load(lstm_preds_path)
    df_nifty  = load_nifty()
    real_prices = df_nifty["Close"].values
    real_returns = df_nifty["log_ret"].dropna().values

    # LSTM-BSM delta baseline
    lstm_bsm_errors, lstm_bsm_costs = run_lstm_bsm_hedge(
        real_prices=real_prices,
        lstm_vols=lstm_vols,
        S0=ENV_PARAMS["S0"],
        K=ENV_PARAMS["K"], T=ENV_PARAMS["T"],
        r=ENV_PARAMS["r"], kappa=ENV_PARAMS["kappa"],
        n_steps=ENV_PARAMS["n_steps"],
    )
    lstm_bsm_metrics = record_baseline_metrics(
        lstm_bsm_errors, lstm_bsm_costs, "LSTM-guided BSM delta"
    )
    print(json.dumps(lstm_bsm_metrics, indent=2))

    # Bootstrap real-path BSM baseline
    boot_errors, boot_costs = run_bootstrap_hedge(
        real_returns=real_returns,
        env_params=ENV_PARAMS,
    )
    boot_metrics = record_baseline_metrics(boot_errors, boot_costs, "BSM bootstrap (real returns)")
    print(json.dumps(boot_metrics, indent=2))

    analysis_lstm_bsm = analyse_hedging_results(lstm_bsm_errors, lstm_bsm_costs)
    analysis_boot = analyse_hedging_results(boot_errors, boot_costs)

    analysis_bootstrap_vs_lstm = paired_bootstrap_mae_diff(boot_errors, lstm_bsm_errors)

    all_metrics_ext = all_metrics + [lstm_bsm_metrics, boot_metrics]
    with open(RESULTS / "p1_metrics.json", "w") as f:
        json.dump(all_metrics_ext, f, indent=2)
    print("Updated p1_metrics.json with new baselines")

    extended_p1["baselines"]["lstm_bsm"] = analysis_lstm_bsm
    extended_p1["baselines"]["bootstrap_bsm"] = analysis_boot
    extended_p1["paired_bootstrap"]["bootstrap_vs_lstm_bsm"] = analysis_bootstrap_vs_lstm

    with open(RESULTS / "p1_analysis.json", "w") as f:
        json.dump(extended_p1, f, indent=2)
    print("Saved p1_analysis.json")

    # Plot comparison including new baselines
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, vals in [("BSM daily",         daily_errors),
                         ("BSM weekly",        weekly_errors),
                         ("RL (PPO)",          rl_pnl),
                         ("LSTM-guided BSM",   lstm_bsm_errors),
                         ("Bootstrap BSM",     boot_errors)]:
        ax.hist(vals, bins=60, alpha=0.50, label=label, density=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Terminal hedging PnL")
    ax.set_ylabel("Density")
    ax.set_title("Terminal PnL — all baselines including LSTM-guided BSM")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS / "p1_pnl_all_baselines.png", dpi=150)
    plt.close(fig)
    print("Saved p1_pnl_all_baselines.png")
else:
    print("  Skipping LSTM/bootstrap baselines — run run_p3.py first")

# ==========================================================================
# Phase 6 — Domain-randomized RL
# ==========================================================================
print("\n" + "=" * 60)
print("PHASE 6 — Domain-randomized RL  (σ ~ Uniform(0.10, 0.40))")
print("=" * 60)

dr_params = {**ENV_PARAMS, "sigma_range": (0.10, 0.40)}
dr_vec_env = make_vec_env(lambda: HedgingEnv(**dr_params), n_envs=4)
dr_model = PPO(
    "MlpPolicy", dr_vec_env,
    learning_rate=1e-4, n_steps=512, batch_size=128, n_epochs=10, gamma=0.99,
    policy_kwargs=dict(net_arch=[128, 64]),
    verbose=0,
)
dr_model.learn(total_timesteps=300_000)
dr_model.save(str(MODELS / "ppo_domain_random"))
print("Saved models/ppo_domain_random")

dr_results  = evaluate_rl_agent(dr_model, HedgingEnv, ENV_PARAMS, n_episodes=N_EPISODES)
dr_pnl      = dr_results["terminal_pnl"]
dr_costs    = dr_results["episode_tc"]
dr_all_rl   = np.concatenate(dr_results["delta_paths"])
dr_all_bsm  = np.concatenate(dr_results["bsm_delta_paths"])
dr_corr     = float(np.corrcoef(dr_all_bsm, dr_all_rl)[0, 1])
dr_metrics  = {
    "name":       "RL (domain random σ)",
    "mean_error": float(np.mean(dr_pnl)),
    "std_error":  float(np.std(dr_pnl)),
    "mae":        float(np.mean(np.abs(dr_pnl))),
    "mean_cost":  float(np.mean(dr_costs)),
    "bsm_corr":   dr_corr,
}
dr_analysis = analyse_hedging_results(
    dr_pnl,
    dr_costs,
    delta_paths=dr_results["delta_paths"],
    bsm_delta_paths=dr_results["bsm_delta_paths"],
    **_episode_state_means(dr_results["obs_paths"]),
)
dr_step_traces = _flatten_episode_traces(dr_pnl, dr_results["obs_paths"])
dr_analysis["regimes_stepwise"] = analyse_hedging_results(
    dr_step_traces["values"],
    np.repeat(dr_costs, [len(ep) for ep in dr_results["obs_paths"]]) if dr_results["obs_paths"] else np.array([]),
    moneyness=dr_step_traces["moneyness"],
    tau_frac=dr_step_traces["tau_frac"],
    sigma_proxy=dr_step_traces["sigma"],
)["regimes"]
print(f"  Domain-random RL  MAE={dr_metrics['mae']:.4f}  "
      f"corr={dr_corr:.4f}  TC={dr_metrics['mean_cost']:.4f}")

with open(RESULTS / "p1_domain_random_metrics.json", "w") as f:
    json.dump(dr_metrics, f, indent=2)
print("Saved p1_domain_random_metrics.json")

extended_p1["baselines"]["domain_random_rl"] = dr_analysis
with open(RESULTS / "p1_analysis.json", "w") as f:
    json.dump(extended_p1, f, indent=2)
print("Updated p1_analysis.json")


# ==========================================================================
# Plots
# ==========================================================================
print("\n" + "=" * 60)
print("PLOTS")
print("=" * 60)

# --- Terminal PnL distributions ---
fig, ax = plt.subplots(figsize=(9, 5))
for label, vals in [("BSM daily",  daily_errors),
                     ("BSM weekly", weekly_errors),
                     ("RL (PPO)",   rl_pnl)]:
    ax.hist(vals, bins=60, alpha=0.55, label=label, density=True)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_xlabel("Terminal hedging PnL")
ax.set_ylabel("Density")
ax.set_title("Terminal PnL distribution")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS / "p1_pnl_comparison.png", dpi=150)
plt.close(fig)
print("Saved p1_pnl_comparison.png")

# --- Mean hedge ratio over an episode (RL replicates BSM delta) ---
# Use per-episode BSM delta computed at the *actual* S and tau seen by the agent,
# so the comparison is state-aligned rather than a fixed ATM path.
max_ep_len = max(len(p) for p in rl_results["delta_paths"])

def _pad(paths, length):
    return np.array([p + [p[-1]] * (length - len(p)) for p in paths])

rl_path  = _pad(rl_results["delta_paths"],     max_ep_len).mean(axis=0)
bsm_path = _pad(rl_results["bsm_delta_paths"], max_ep_len).mean(axis=0)

# Per-episode step-level residuals for correlation / RMSE
all_rl  = np.concatenate(rl_results["delta_paths"])
all_bsm = np.concatenate(rl_results["bsm_delta_paths"])
corr       = float(np.corrcoef(all_bsm, all_rl)[0, 1])
rmse_delta = float(np.sqrt(np.mean((all_rl - all_bsm) ** 2)))
print(f"  RL vs BSM delta  —  corr={corr:.4f}  RMSE={rmse_delta:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.plot(bsm_path, label="BSM delta (state-aligned mean)", lw=2, color="#4c72b0")
ax.plot(rl_path,  label=f"RL agent (corr={corr:.3f})", lw=2, ls="--", color="#c44e52")
ax.set_ylabel("Hedge ratio (delta)")
ax.set_title("RL Agent Replicates BSM Delta Hedging Strategy")
ax.legend()
ax.set_ylim(0, 1)

ax2 = axes[1]
ax2.bar(range(len(rl_path)), rl_path - bsm_path,
        color="#dd8452", alpha=0.6, width=1.0)
ax2.axhline(0, color="black", lw=0.8, ls="--")
ax2.set_xlabel("Step within episode")
ax2.set_ylabel("RL − BSM")
ax2.set_title(f"Deviation from BSM delta  (RMSE={rmse_delta:.4f})")

plt.tight_layout()
fig.savefig(RESULTS / "p1_hedge_ratios.png", dpi=150)
plt.close(fig)
print("Saved p1_hedge_ratios.png")

# --- Transaction cost comparison ---
cost_dict = {
    "BSM daily":  float(np.mean(daily_costs)),
    "BSM weekly": float(np.mean(weekly_costs)),
    "RL (PPO)":   float(np.mean(rl_costs)),
}
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(list(cost_dict.keys()), list(cost_dict.values()), width=0.5)
ax.set_ylabel("Mean transaction cost per episode")
ax.set_title("Transaction cost comparison")
plt.tight_layout()
fig.savefig(RESULTS / "p1_cost_comparison.png", dpi=150)
plt.close(fig)
print("Saved p1_cost_comparison.png")

# --- Box plot of absolute hedging errors ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot(
    [np.abs(daily_errors), np.abs(weekly_errors), np.abs(rl_pnl)],
    labels=["BSM daily", "BSM weekly", "RL (PPO)"],
    showfliers=False,
)
ax.set_ylabel("Absolute terminal hedging error")
ax.set_title("Hedging error distribution (box plot)")
plt.tight_layout()
fig.savefig(RESULTS / "p1_error_boxplot.png", dpi=150)
plt.close(fig)
print("Saved p1_error_boxplot.png")

print("\nProject 1 complete. All outputs in results/")
