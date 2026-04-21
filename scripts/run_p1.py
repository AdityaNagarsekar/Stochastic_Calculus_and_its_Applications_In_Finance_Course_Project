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
from backtest import evaluate_rl_agent
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

print("\nResults summary:")
for m in [bsm_daily_metrics, bsm_weekly_metrics, rl_metrics]:
    print(f"  {m['name']:12s}  MAE={m['mae']:.4f}  mean={m['mean_error']:+.4f}  "
          f"std={m['std_error']:.4f}  cost={m['mean_cost']:.4f}")

all_metrics = [bsm_daily_metrics, bsm_weekly_metrics, rl_metrics]
with open(RESULTS / "p1_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
print("Saved p1_metrics.json")


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
